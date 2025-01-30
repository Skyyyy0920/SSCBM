import numpy as np
import os
import torch
import torchvision
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict

import random
import logging
from torchvision.models import resnet50
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path
from pytorch_lightning import seed_everything
from torchvision import transforms

#########################################################
# CONCEPT INFORMATION REGARDING CelebA
#########################################################


SELECTED_CONCEPTS = [
    2,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    39,
]

CONCEPT_SEMANTICS = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young',
]


class CelebaDataset(Dataset):
    def __init__(self, ds, labeled_ratio, training,
                 seed=42, transform=None,
                 concept_transform=None, label_transform=None):
        self.ds = ds
        self.transform = transform
        self.concept_transform = concept_transform
        self.label_transform = label_transform
        self.l_choice = defaultdict(bool)

        if training:
            random.seed(seed)
            class_count = defaultdict(int)
            for img_data in self.ds:
                class_count[img_data[1][0].item()] += 1

            labeled_count = defaultdict(int)
            for idx, img_data in enumerate(self.ds):
                class_label = img_data[1][0].item()
                if labeled_count[class_label] < labeled_ratio * class_count[class_label]:
                    self.l_choice[idx] = True
                    labeled_count[class_label] += 1
                else:
                    self.l_choice[idx] = False
        else:
            for idx in range(len(self.ds)):
                self.l_choice[idx] = True

        count = 0
        for key, value in self.l_choice.items():
            if value:
                count += 1
        logging.info(f"actual labeled ratio: {count / len(self.l_choice)}")

        self.neighbor = self.nearest_neighbors_resnet(k=2)

    def nearest_neighbors_resnet(self, k=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet50(pretrained=True).to(device)
        model.eval()
        imgs = []
        for d in self.ds:
            imgs.append(d[0][None, :, :, :])
        imgs_tensor = torch.cat(imgs, dim=0)
        imgs_tensor = imgs_tensor.to(device)
        num_chunks = 10
        chunked_tensors = torch.chunk(imgs_tensor, num_chunks, dim=0)

        features = []
        with torch.no_grad():
            for chunk in tqdm(chunked_tensors):
                features.append(model(chunk))
        features = torch.cat(features, dim=0)
        features = features.detach().cpu().numpy()
        labeled_features = []
        for idx in range(len(features)):
            if self.l_choice[idx]:
                labeled_features.append(features[idx])
        labeled_features = np.array(labeled_features)
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
        nbrs.fit(labeled_features)
        distances, indices = nbrs.kneighbors(features)

        weights = 1.0 / (distances + 1e-6)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        return [{'indices': idx, 'weights': w} for idx, w in zip(indices, weights)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_data = self.ds[idx]
        l = self.l_choice[idx]
        neighbor_info = self.neighbor[idx]
        neighbor_indices = neighbor_info['indices']
        nbr_concepts = []
        # print(len(neighbor_indices))
        for idx in neighbor_indices:
            # What is the attribute_label?
            # print(self.ds[idx][1][1])
            nbr_concepts.append(self.ds[idx][1][1].unsqueeze(0))
        nbr_concepts = torch.concat(nbr_concepts, dim=0)
        nbr_weight = torch.from_numpy(neighbor_info['weights'])

        class_label = img_data[1][0]
        if self.label_transform:
            class_label = self.label_transform(class_label)

        attr_label = img_data[1][1]
        if self.concept_transform is not None:
            attr_label = self.concept_transform(attr_label)

        return img_data[0], class_label, torch.FloatTensor(attr_label), torch.tensor(l), nbr_concepts, nbr_weight


def generate_data(
        config,
        labeled_ratio=0.1,
        seed=42,
        output_dataset_vars=False,
):
    root_dir = './data/CelebA/'
    concept_group_map = None
    seed_everything(seed)
    use_binary_vector_class = config.get('use_binary_vector_class', False)
    if use_binary_vector_class:
        width = config.get('label_binary_width', 5)

        def _binarize(concepts, selected, width):
            result = []
            binary_repr = []
            concepts = concepts[selected]
            for i in range(0, concepts.shape[-1], width):
                binary_repr.append(
                    str(int(np.sum(concepts[i: i + width]) > 0))
                )
            return int("".join(binary_repr), 2)

        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=True,
            target_transform=lambda x: x[0].long() - 1,
            target_type=['attr'],
        )

        concept_freq = np.sum(celeba_train_data.attr.cpu().detach().numpy(), axis=0) / celeba_train_data.attr.shape[0]
        logging.info(f"Concept frequency is: {concept_freq}")
        sorted_concepts = list(map(
            lambda x: x[0],
            sorted(enumerate(np.abs(concept_freq - 0.5)), key=lambda x: x[1]),
        ))
        num_concepts = config.get( 'num_concepts', celeba_train_data.attr.shape[-1]  )
        concept_idxs = sorted_concepts[:num_concepts]
        concept_idxs = sorted(concept_idxs)
        if config.get('num_hidden_concepts', 0):
            num_hidden = config.get('num_hidden_concepts', 0)
            hidden_concepts = sorted(
                sorted_concepts[num_concepts:min((num_concepts + num_hidden), len(sorted_concepts))])
        else:
            hidden_concepts = []
        logging.info(f"Selecting concepts: {concept_idxs}")
        logging.info(f"\tAnd hidden concepts: {hidden_concepts}")
        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(config['image_size']),
                transforms.CenterCrop(config['image_size']),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    _binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_idxs + hidden_concepts),
                        width=width,
                    ),
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ],
            target_type=['identity', 'attr'],
        )
        label_remap = {}
        vals, counts = np.unique(
            list(map(
                lambda x: _binarize(
                    x.cpu().detach().numpy(),
                    selected=(concept_idxs + hidden_concepts),
                    width=width,
                ),
                celeba_train_data.attr
            )),
            return_counts=True,
        )
        for i, label in enumerate(vals):
            label_remap[label] = i

        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(config['image_size']),
                transforms.CenterCrop(config['image_size']),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    label_remap[_binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_idxs + hidden_concepts),
                        width=width,
                    )],
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ],
            target_type=['identity', 'attr'],
        )
        num_classes = len(label_remap)

        # And subsample to reduce its massive size
        factor = config.get('label_dataset_subsample', 1)
        if factor != 1:
            train_idxs = np.random.choice(
                np.arange(0, len(celeba_train_data)),
                replace=False,
                size=len(celeba_train_data) // factor,
            )
            logging.info(f"Subsampling to {len(train_idxs)} elements.")
            celeba_train_data = torch.utils.data.Subset(
                celeba_train_data,
                train_idxs,
            )
    else:
        concept_selection = list(range(0, len(CONCEPT_SEMANTICS)))
        if config.get('selected_concepts', False):
            concept_selection = SELECTED_CONCEPTS
        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=True,
            target_transform=lambda x: x[0].long() - 1,
            target_type=['identity'],
        )
        num_concepts = config.get(
            'num_concepts',
            celeba_train_data.attr.shape[-1],
        )
        vals, counts = np.unique(
            celeba_train_data.identity,
            return_counts=True,
        )
        sorted_labels = list(map(
            lambda x: x[0],
            sorted(zip(vals, counts), key=lambda x: -x[1])
        ))
        logging.info(f"Selecting {config['num_classes']} out of {len(vals)} classes")
        result_dir = config.get('result_dir', None)
        if result_dir:
            Path(result_dir).mkdir(parents=True, exist_ok=True)
            np.save(
                os.path.join(
                    result_dir,
                    f"selected_top_{config['num_classes']}_labels.npy",
                ),
                sorted_labels[:config['num_classes']],
            )
        label_remap = {}
        for i, label in enumerate(sorted_labels[:config['num_classes']]):
            label_remap[label] = i

        # Now reload by transform the labels accordingly
        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(config['image_size']),
                transforms.CenterCrop(config['image_size']),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    label_remap.get(
                        x[0].cpu().detach().item() - 1,
                        config['num_classes']
                    ),
                    dtype=torch.long,
                ),
                x[1][concept_selection].float(),
            ],
            target_type=['identity', 'attr'],
        )
        num_classes = config['num_classes']

        train_idxs = np.where(
            list(map(
                lambda x: x.cpu().detach().item() - 1 in label_remap,
                celeba_train_data.identity
            ))
        )[0]
        celeba_train_data = torch.utils.data.Subset(
            celeba_train_data,
            train_idxs,
        )
    total_samples = len(celeba_train_data)
    train_samples = int(0.7 * total_samples)
    test_samples = int(0.2 * total_samples)
    val_samples = total_samples - test_samples - train_samples
    logging.info(
        f"Data split is: {total_samples} = {train_samples} (train) + "
        f"{test_samples} (test) + {val_samples} (validation)"
    )
    celeba_train_data, celeba_test_data, celeba_val_data = \
        torch.utils.data.random_split(
            celeba_train_data,
            [train_samples, test_samples, val_samples],
        )

    celeba_train_data = CelebaDataset(celeba_train_data, labeled_ratio=labeled_ratio,
                                      training=True, seed=seed)
    celeba_val_data = CelebaDataset(celeba_val_data, labeled_ratio=1., training=False, seed=seed)
    celeba_test_data = CelebaDataset(celeba_test_data, labeled_ratio=1., training=False, seed=seed)

    train_dl = torch.utils.data.DataLoader(
        celeba_train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
    )
    test_dl = torch.utils.data.DataLoader(
        celeba_test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    val_dl = torch.utils.data.DataLoader(
        celeba_val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )

    if config.get('weight_loss', False):
        attribute_count = np.zeros((num_concepts,))
        samples_seen = 0
        for i, (_, (y, c)) in enumerate(train_dl):
            c = c.cpu().detach().numpy()
            attribute_count += np.sum(c, axis=0)
            samples_seen += c.shape[0]
        imbalance = samples_seen / attribute_count - 1
    else:
        imbalance = None
    # if not output_dataset_vars:
    #     return train_dl, val_dl, test_dl, imbalance
    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (num_concepts, len(label_remap), concept_group_map),
    )
