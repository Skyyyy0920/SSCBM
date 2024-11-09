import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


class Derm7ptDataset(Dataset):
    def __init__(self, csv_path, image_dir, labeled_ratio, training,
                 seed=42, root_dir='./data/derm7pt/', transform=None,
                 concept_transform=None, label_transform=None):
        """
        Args:
            csv_path: Path to the CSV metadata file
            image_dir: Directory containing the images
            labeled_ratio: Ratio of labeled data to use
            training: Boolean indicating if this is for training
            seed: Random seed
            root_dir: Root directory of the dataset
            transform: Image transforms
            concept_transform: Transform for concept vectors
            label_transform: Transform for labels
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.concept_transform = concept_transform
        self.label_transform = label_transform
        self.image_dir = image_dir
        self.root_dir = root_dir
        self.l_choice = defaultdict(bool)
        self.is_train = training

        # Define label mappings
        self.label_map = {
            'basal cell carcinoma': 0,
            'nevus': 1,
            'melanoma': 2,
            'DF/LT/MLS/MISC': 3,
            'seborrheic keratosis': 4
        }

        # Define the 7-point criteria as concepts
        self.concept_columns = [
            'pigment_network', 'blue_whitish_veil', 'vascular_structures',
            'pigmentation', 'streaks', 'dots_and_globules', 'regression_structures'
        ]

        # Create concept maps for each feature
        self.concept_maps = {}
        for column in self.concept_columns:
            unique_values = sorted(self.data[column].unique())
            self.concept_maps[column] = {val: idx for idx, val in enumerate(unique_values)}

        if training:
            # Split labeled/unlabeled data maintaining class distribution
            np.random.seed(seed)
            class_count = defaultdict(int)
            for _, row in self.data.iterrows():
                class_count[row['diagnosis']] += 1

            labeled_count = defaultdict(int)
            for idx, row in self.data.iterrows():
                class_label = row['diagnosis']
                if labeled_count[class_label] < labeled_ratio * class_count[class_label]:
                    self.l_choice[idx] = True
                    labeled_count[class_label] += 1
                else:
                    self.l_choice[idx] = False
        else:
            for idx in range(len(self.data)):
                self.l_choice[idx] = True

        # Compute nearest neighbors for semi-supervised learning
        self.neighbor = self.nearest_neighbors_resnet(k=2)

    def _get_concept_vector(self, row):
        """Convert 7-point criteria into one-hot encoded concept vector."""
        concept_vector = []
        for concept in self.concept_columns:
            n_classes = len(self.concept_maps[concept])
            current_value = self.concept_maps[concept][row[concept]]
            one_hot = [1.0 if i == current_value else 0.0 for i in range(n_classes)]
            concept_vector.extend(one_hot)
        return np.array(concept_vector)

    def nearest_neighbors_resnet(self, k=3):
        """Compute nearest neighbors using ResNet50 features."""
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet50(pretrained=True).to(device)
        model.eval()

        # Extract features for all images
        imgs = []
        for _, row in tqdm(self.data.iterrows(), desc="Processing images"):
            img_path = os.path.join(self.image_dir, row['derm'])
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0)
            imgs.append(img_tensor)

        imgs_tensor = torch.cat(imgs, dim=0)
        imgs_tensor = imgs_tensor.to(device)

        # Process in chunks to avoid memory issues
        num_chunks = 10
        chunked_tensors = torch.chunk(imgs_tensor, num_chunks, dim=0)

        features = []
        with torch.no_grad():
            for chunk in tqdm(chunked_tensors, desc="Extracting features"):
                features.append(model(chunk))
        features = torch.cat(features, dim=0)
        features = features.detach().cpu().numpy()

        # Get features for labeled samples only
        labeled_features = []
        for idx in range(len(features)):
            if self.l_choice[idx]:
                labeled_features.append(features[idx])
        labeled_features = np.array(labeled_features)

        # Compute nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
        nbrs.fit(labeled_features)
        distances, indices = nbrs.kneighbors(features)

        # Compute weights based on distances
        weights = 1.0 / (distances + 1e-6)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        return [{'indices': idx, 'weights': w} for idx, w in zip(indices, weights)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
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
        img_path = os.path.join(self.image_dir, row['derm'])
        img = Image.open(img_path).convert('RGB')

        # Get class label
        class_label = self.label_map[row['diagnosis']]
        if self.label_transform:
            class_label = self.label_transform(class_label)

        if self.transform:
            img = self.transform(img)

        # Get concept labels (7-point criteria)
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
        root_dir='./data/derm7pt',
        num_workers=1,
        concept_transform=None,
        label_transform=None,
):
    """Create data loader for the Derm7pt dataset."""
    resized_resol = int(resol * 256 / 224)
    is_training = training

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

    dataset = Derm7ptDataset(
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
    """Generate train, validation and test data loaders."""
    root_dir = config['root_dir']
    train_data_path = os.path.join(root_dir, 'meta/train_meta.csv')
    val_data_path = os.path.join(root_dir, 'meta/valid_meta.csv')
    test_data_path = os.path.join(root_dir, 'meta/test_meta.csv')

    # Number of concepts (total number of possible values across all 7 criteria)
    N_CONCEPTS = 21  # Sum of unique values for each of the 7 criteria
    N_CLASSES = 5  # Number of diagnosis classes

    train_dl = load_data(
        labeled_ratio=labeled_ratio,
        seed=seed,
        csv_path=train_data_path,
        training=True,
        batch_size=config['batch_size'],
        image_dir=os.path.join(root_dir, 'images'),
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
        image_dir=os.path.join(root_dir, 'images'),
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
        image_dir=os.path.join(root_dir, 'images'),
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
    )

    return train_dl, val_dl, test_dl, None, (N_CONCEPTS, N_CLASSES, None)