import os
import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from collections import defaultdict
import random
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

N_CONCEPTS = 31
N_CLASSES = 5


class PBCDataset(Dataset):
    def __init__(self, csv_path, image_dir, labeled_ratio, training,
                 seed=42, root_dir='./data/PBC/', path_transform=None, transform=None,
                 concept_transform=None, label_transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.concept_transform = concept_transform
        self.label_transform = label_transform
        self.image_dir = image_dir
        self.root_dir = root_dir
        self.path_transform = path_transform
        self.l_choice = defaultdict(bool)
        self.is_train = 'train' in csv_path

        self.label_map = {
            'Neutrophil': 0, 'Eosinophil': 1, 'Basophil': 2,
            'Monocyte': 3, 'Lymphocyte': 4
        }
        self.concept_columns = ['cell_size', 'cell_shape', 'nucleus_shape',
                                'nuclear_cytoplasmic_ratio', 'chromatin_density',
                                'cytoplasm_vacuole', 'cytoplasm_texture',
                                'cytoplasm_colour', 'granule_type',
                                'granule_colour', 'granularity']
        self.concept_maps = {}
        for column in self.concept_columns:
            unique_values = sorted(self.data[column].unique())
            self.concept_maps[column] = {val: idx for idx, val in enumerate(unique_values)}

        print("Concept maps:")
        for concept, value_map in self.concept_maps.items():
            print(f"{concept}: {value_map}")

        each_class_num = math.ceil(labeled_ratio * len(self.data) / N_CLASSES)
        if training:
            random.seed(seed)
            labeled_count = defaultdict(int)
            for idx, row in self.data.iterrows():
                class_label = row['label']
                if labeled_count[class_label] < each_class_num:
                    self.l_choice[idx] = True
                    labeled_count[class_label] += 1
                else:
                    self.l_choice[idx] = False
        else:
            for idx in range(len(self.data)):
                self.l_choice[idx] = True

        count = 0
        for key, value in self.l_choice.items():
            if value:
                count += 1

        logging.info(f"each class number: {each_class_num}")
        logging.info(f"actual labeled ratio: {count / len(self.l_choice)}")

        neighbor_num = each_class_num if each_class_num <= 2 else 3
        self.neighbor = self.nearest_neighbors_resnet(k=neighbor_num)

    def _get_concept_vector(self, row):
        concept_vector = []
        for concept in self.concept_columns:
            n_classes = len(self.concept_maps[concept])
            current_value = self.concept_maps[concept][row[concept]]
            one_hot = [1.0 if i == current_value else 0.0 for i in range(n_classes)]
            concept_vector.extend(one_hot)
        return np.array(concept_vector)

    def nearest_neighbors_resnet(self, k=3):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet50(pretrained=True).to(device)
        model.eval()

        imgs = []
        for _, row in tqdm(self.data.iterrows(), desc="Processing images"):
            img_path = os.path.join(self.root_dir, row['path'])
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0)
            imgs.append(img_tensor)

        imgs_tensor = torch.cat(imgs, dim=0)
        imgs_tensor = imgs_tensor.to(device)
        num_chunks = 10
        chunked_tensors = torch.chunk(imgs_tensor, num_chunks, dim=0)

        features = []
        with torch.no_grad():
            for chunk in tqdm(chunked_tensors, desc="Extracting features"):
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
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        l = self.l_choice[idx]

        # Get neighbor information
        neighbor_info = self.neighbor[idx]
        neighbor_indices = neighbor_info['indices']
        nbr_concepts = []
        for n_idx in neighbor_indices:
            nbr_concepts.append(self._get_concept_vector(self.data.iloc[n_idx]))
        nbr_concepts = torch.tensor(nbr_concepts)
        nbr_weight = torch.from_numpy(neighbor_info['weights'])

        # Load and process image
        img_path = os.path.join(self.root_dir, row['path'])
        img = Image.open(img_path).convert('RGB')

        # Get class label
        class_label = self.label_map[row['label']]
        if self.label_transform:
            class_label = self.label_transform(class_label)
        if self.transform:
            img = self.transform(img)

        # Get concept labels
        attr_label = self._get_concept_vector(row)
        if self.concept_transform is not None:
            attr_label = self.concept_transform(attr_label)

        return img, class_label, torch.FloatTensor(attr_label), torch.tensor(l), nbr_concepts, nbr_weight


def load_data(
        csv_path,
        batch_size,
        labeled_ratio,
        seed=42,
        training=False,
        image_dir='images',
        resampling=False,
        resol=299,
        root_dir='./data/PBC',
        num_workers=1,
        concept_transform=None,
        label_transform=None,
):
    resized_resol = int(resol * 256 / 224)
    is_training = 'train' in csv_path

    if is_training:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])

    dataset = PBCDataset(
        labeled_ratio=labeled_ratio,
        seed=seed,
        training=training,
        csv_path=csv_path,
        image_dir=image_dir,
        transform=transform,
        root_dir=root_dir,
        concept_transform=concept_transform,
        label_transform=label_transform,
    )

    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return loader


def generate_data(
        config,
        labeled_ratio=0.1,
        seed=42,
):
    root_dir = config['root_dir']
    train_data_path = os.path.join(root_dir, 'PBC_dataset_normal_DIB/pbc_attr_v1_train.csv')
    val_data_path = os.path.join(root_dir, 'PBC_dataset_normal_DIB/pbc_attr_v1_val.csv')
    test_data_path = os.path.join(root_dir, 'PBC_dataset_normal_DIB/pbc_attr_v1_test.csv')

    train_dl = load_data(
        labeled_ratio=labeled_ratio,
        seed=seed,
        csv_path=train_data_path,
        training=True,
        batch_size=config['batch_size'],
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
    )

    val_dl = load_data(
        labeled_ratio=labeled_ratio,
        seed=seed,
        csv_path=val_data_path,
        training=False,
        batch_size=config['batch_size'],
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
    )

    test_dl = load_data(
        labeled_ratio=labeled_ratio,
        seed=seed,
        csv_path=test_data_path,
        training=False,
        batch_size=config['batch_size'],
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
    )

    return train_dl, val_dl, test_dl, None, (N_CONCEPTS, N_CLASSES, None)
