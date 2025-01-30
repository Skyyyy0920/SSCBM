import os
import math
import glob
import torch
import random
import logging
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

N_CLASSES = 5
N_CONCEPTS = 19


class Derm7ptDataset(Dataset):
    def __init__(self, meta_csv, index_csv, image_dir, labeled_ratio, training,
                 seed=42, root_dir='./data/derm7pt/', transform=None,
                 concept_transform=None, label_transform=None):
        self.file_mapping = self._create_file_mapping(image_dir)
        self.full_meta = pd.read_csv(meta_csv)
        self.split_indexes = pd.read_csv(index_csv)['indexes'].values
        self.data = self.full_meta[self.full_meta['case_num'].isin(self.split_indexes)].copy()
        self.transform = transform
        self.concept_transform = concept_transform
        self.label_transform = label_transform
        self.image_dir = image_dir
        self.root_dir = root_dir
        self.l_choice = defaultdict(bool)
        self.is_train = training

        self.diagnosis_groups = {
            'basal cell carcinoma': 0,
            'blue nevus': 1, 'clark nevus': 1, 'combined nevus': 1,
            'congenital nevus': 1, 'dermal nevus': 1, 'recurrent nevus': 1,
            'reed or spitz nevus': 1,
            'melanoma': 2, 'melanoma (in situ)': 2, 'melanoma (less than 0.76 mm)': 2,
            'melanoma (0.76 to 1.5 mm)': 2, 'melanoma (more than 1.5 mm)': 2,
            'melanoma metastasis': 2,
            'dermatofibroma': 3, 'lentigo': 3, 'melanosis': 3,
            'miscellaneous': 3, 'vascular lesion': 3,
            'seborrheic keratosis': 4
        }

        self.concept_mappings = {
            'pigment_network': {'absent': 0, 'typical': 1, 'atypical': 2},
            'streaks': {'absent': 0, 'regular': 1, 'irregular': 2},
            'pigmentation': {'absent': 0, 'diffuse regular': 1, 'localized regular': 1,
                             'diffuse irregular': 2, 'localized irregular': 2},
            'regression_structures': {'absent': 0, 'blue areas': 1, 'white areas': 1, 'combinations': 1},
            'dots_and_globules': {'absent': 0, 'regular': 1, 'irregular': 2},
            'blue_whitish_veil': {'absent': 0, 'present': 1},
            'vascular_structures': {'absent': 0, 'arborizing': 1, 'comma': 1, 'hairpin': 1,
                                    'within regression': 1, 'wreath': 1, 'dotted': 2, 'linear irregular': 2}
        }

        self.concept_columns = list(self.concept_mappings.keys())

        each_class_num = math.ceil(labeled_ratio * len(self.data) / N_CLASSES)
        if training:
            random.seed(seed)

            labeled_count = defaultdict(int)
            for idx, row in self.data.iterrows():
                class_label = self.diagnosis_groups[row['diagnosis']]
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

    def _create_file_mapping(self, image_dir):
        mapping = {}
        for root, _, files in os.walk(image_dir):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, image_dir)
                    mapping[rel_path.lower()] = rel_path
        return mapping

    def _get_actual_image_path(self, image_name):
        direct_path = os.path.join(self.image_dir, image_name)
        if os.path.exists(direct_path):
            return direct_path

        lower_name = image_name.lower()
        if lower_name in self.file_mapping:
            return os.path.join(self.image_dir, self.file_mapping[lower_name])

        image_dir = Path(self.image_dir)
        base_name = os.path.basename(image_name)
        parent_dir = os.path.dirname(image_name)

        search_dir = image_dir / parent_dir if parent_dir else image_dir

        if search_dir.exists():
            for file in search_dir.iterdir():
                if file.name.lower() == base_name.lower():
                    return str(file)

        raise FileNotFoundError(f"Image not found: {image_name}")

    def _get_concept_vector(self, row):
        concept_vector = []
        for concept, mapping in self.concept_mappings.items():
            value = row[concept]
            n_classes = max(mapping.values()) + 1
            one_hot = [1.0 if i == mapping[value] else 0.0 for i in range(n_classes)]
            concept_vector.extend(one_hot)
        return np.array(concept_vector)

    def _get_total_concept_dims(self):
        total_dims = 0
        for mapping in self.concept_mappings.values():
            total_dims += max(mapping.values()) + 1
        return total_dims

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
            try:
                img_path = self._get_actual_image_path(row['derm'])
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0)
                imgs.append(img_tensor)
            except Exception as e:
                print(f"Error loading image {row['derm']}: {str(e)}")
                img = Image.new('RGB', (224, 224), color='black')
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
        labeled_indices = []
        for idx in range(len(features)):
            if self.l_choice[idx]:
                labeled_features.append(features[idx])
                labeled_indices.append(idx)
        labeled_features = np.array(labeled_features)

        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
        nbrs.fit(labeled_features)
        distances, indices = nbrs.kneighbors(features)

        indices = np.array([labeled_indices[i] for i in indices.flatten()]).reshape(indices.shape)

        weights = 1.0 / (distances + 1e-6)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        return [{'indices': idx, 'weights': w} for idx, w in zip(indices, weights)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        l = self.l_choice[idx]

        neighbor_info = self.neighbor[idx]
        neighbor_indices = neighbor_info['indices']
        nbr_concepts = []
        for n_idx in neighbor_indices:
            nbr_concepts.append(self._get_concept_vector(self.data.iloc[n_idx]))
        nbr_concepts = torch.tensor(nbr_concepts)
        nbr_weight = torch.from_numpy(neighbor_info['weights'])

        try:
            img_path = self._get_actual_image_path(row['derm'])
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {row['derm']}: {str(e)}")
            img = Image.new('RGB', (299, 299), color='black')

        class_label = self.diagnosis_groups[row['diagnosis']]
        if self.label_transform:
            class_label = self.label_transform(class_label)

        if self.transform:
            img = self.transform(img)

        attr_label = self._get_concept_vector(row)
        if self.concept_transform is not None:
            attr_label = self.concept_transform(attr_label)

        return img, class_label, torch.FloatTensor(attr_label), torch.tensor(l), nbr_concepts, nbr_weight


def load_data(
        meta_csv,
        index_csv,
        batch_size,
        labeled_ratio,
        seed=42,
        training=False,
        image_dir='images',
        resampling=False,
        resol=299,
        root_dir='./data/derm7pt',
        num_workers=1,
        concept_transform=None,
        label_transform=None,
):
    if training:
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

    dataset = Derm7ptDataset(
        meta_csv=meta_csv,
        index_csv=index_csv,
        labeled_ratio=labeled_ratio,
        seed=seed,
        training=training,
        image_dir=image_dir,
        transform=transform,
        root_dir=root_dir,
        concept_transform=concept_transform,
        label_transform=label_transform,
    )

    if training:
        drop_last = False  # data is limited
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
    meta_csv = os.path.join(root_dir, 'meta/meta.csv')
    train_index_csv = os.path.join(root_dir, 'meta/train_indexes.csv')
    val_index_csv = os.path.join(root_dir, 'meta/valid_indexes.csv')
    test_index_csv = os.path.join(root_dir, 'meta/test_indexes.csv')

    train_dl = load_data(
        meta_csv=meta_csv,
        index_csv=train_index_csv,
        labeled_ratio=labeled_ratio,
        seed=seed,
        training=True,
        batch_size=config['batch_size'],
        image_dir=os.path.join(root_dir, 'images'),
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
    )

    val_dl = load_data(
        meta_csv=meta_csv,
        index_csv=val_index_csv,
        labeled_ratio=labeled_ratio,
        seed=seed,
        training=False,
        batch_size=config['batch_size'],
        image_dir=os.path.join(root_dir, 'images'),
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
    )

    test_dl = load_data(
        meta_csv=meta_csv,
        index_csv=test_index_csv,
        labeled_ratio=labeled_ratio,
        seed=seed,
        training=False,
        batch_size=config['batch_size'],
        image_dir=os.path.join(root_dir, 'images'),
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
    )

    return train_dl, val_dl, test_dl, None, (N_CONCEPTS, N_CLASSES, None)
