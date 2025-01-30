import os
import math
import torch
import numpy as np
import random
import logging
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
from torchvision.models import resnet50
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader, random_split

N_CONCEPTS = 85
N_CLASSES = 50


class AwA2Dataset(Dataset):
    def __init__(self, ds, labeled_ratio, training,
                 seed=42, transform=None,
                 concept_transform=None, label_transform=None):
        self.ds = ds
        self.transform = transform
        self.concept_transform = concept_transform
        self.label_transform = label_transform
        self.l_choice = defaultdict(bool)

        each_class_num = math.ceil(labeled_ratio * len(self.ds) / N_CLASSES)
        if training:
            random.seed(seed)
            labeled_count = defaultdict(int)
            for idx, img_data in enumerate(self.ds):
                class_label = img_data[1][0].item()
                if labeled_count[class_label] < each_class_num:
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
        logging.info(f"each class number: {each_class_num}")
        logging.info(f"actual labeled ratio: {count / len(self.l_choice)}")

        neighbor_num = each_class_num if each_class_num <= 2 else 3
        self.neighbor = self.nearest_neighbors_resnet(k=neighbor_num)

    def nearest_neighbors_resnet(self, k=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet50(pretrained=True).to(device)
        model.eval()
        imgs = []
        for d in self.ds:
            imgs.append(d[0][None, :, :, :])
        imgs_tensor = torch.cat(imgs, dim=0)
        imgs_tensor = imgs_tensor.to(device)
        num_chunks = 32
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

        return img_data[0], class_label, torch.tensor(attr_label).to(torch.float32), torch.tensor(
            l), nbr_concepts, nbr_weight


class RawAwA(Dataset):
    def __init__(self, img_paths, labels, l2c_dct):
        super().__init__()
        self.img_paths = img_paths
        self.labels = labels
        self.l2c_dct = l2c_dct

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB').resize((299, 299))
        img = np.array(img).transpose()

        cs = self.l2c_dct[label, :]
        return torch.tensor(img / 255.).to(torch.float32), (torch.tensor(label), torch.tensor(cs))


def load_data(data_dir, sample=1, seed=42):
    classes = []
    with open(os.path.join(data_dir, "classes.txt")) as f:
        lines = f.readlines()
        for line in lines:
            classes.append(line.split('\t')[1].strip())
    cls2cls_id_dct = {k: v for v, k in enumerate(classes)}

    with open(os.path.join(data_dir, "predicate-matrix-binary.txt"), 'r') as f:
        lines = f.readlines()
        lines = [line.strip('\t').split(' ') for line in lines]
        lines = [[int(i) for i in line] for line in lines]
        cs_matrix = np.array(lines)

    img_paths = []
    labels = []
    for _cls in classes:
        _cls = _cls.strip()
        dir = os.path.join(data_dir, "JPEGImages", _cls)
        img_names = [img for img in os.listdir(dir) if img.endswith('jpg')]
        img_paths += [os.path.join(dir, img) for img in img_names]
        labels += [cls2cls_id_dct[_cls]] * len(img_names)

    size = len(img_paths)
    print("There are {} images in total.".format(size))
    indices = list(range(size))
    random.seed(seed)
    indices = random.sample(indices, int(size * sample))
    sampled_img_paths, sampled_labels = [], []
    random.shuffle(indices)
    for i in indices:
        sampled_img_paths.append(img_paths[i])
        sampled_labels.append(labels[i])
    train_size = int(int(size * sample) * 0.8)
    val_size = int(int(size * sample) * 0.1)
    train_set = RawAwA(sampled_img_paths[:train_size], sampled_labels[:train_size], cs_matrix)
    val_set = RawAwA(sampled_img_paths[train_size:train_size + val_size],
                     sampled_labels[train_size:train_size + val_size], cs_matrix)
    test_set = RawAwA(sampled_img_paths[train_size + val_size:], sampled_labels[train_size + val_size:], cs_matrix)
    return train_set, val_set, test_set


def generate_data(
        config,
        labeled_ratio=0.1,
        seed=42,
):
    train_data, val_data, test_data = load_data(data_dir=config['root_dir'], seed=seed)

    train_data = AwA2Dataset(train_data, labeled_ratio=labeled_ratio, training=True, seed=seed)
    val_data = AwA2Dataset(val_data, labeled_ratio=1, training=False, seed=seed)
    test_data = AwA2Dataset(test_data, labeled_ratio=1, training=False, seed=seed)

    train_dl = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
    )
    test_dl = torch.utils.data.DataLoader(
        test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    val_dl = torch.utils.data.DataLoader(
        val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )

    # Finally, determine whether we will need to compute the imbalance factors
    if config.get('weight_loss', False):
        attribute_count = np.zeros((N_CONCEPTS,))
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
    return train_dl, val_dl, test_dl, imbalance, (N_CONCEPTS, N_CLASSES, None)
