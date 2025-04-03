import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split, Subset, TensorDataset
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import zipfile
import gzip
import shutil
import urllib.request
from io import BytesIO
import h5py

class DatasetManager:
    """
    Manages loading and preprocessing of multiple datasets for federated learning
    """
    def __init__(self, data_root='./data', num_clients=10, iid=False):
        """
        Initialize the dataset manager
        
        Args:
            data_root: Directory to store datasets
            num_clients: Number of clients to partition data for
            iid: Whether to distribute data in IID fashion (True) or non-IID (False)
        """
        self.data_root = data_root
        self.num_clients = num_clients
        self.iid = iid
        os.makedirs(data_root, exist_ok=True)
        
    def get_dataset(self, dataset_name, transform=None):
        """
        Load and preprocess a specific dataset
        
        Args:
            dataset_name: Name of dataset ('fashion_mnist', 'cifar10', 'har', 'iot', 'chestxray')
            transform: Optional transform to apply to images
            
        Returns:
            train_dataset, test_dataset: Training and test datasets
        """
        if dataset_name.lower() == 'fashion_mnist':
            return self._load_fashion_mnist(transform)
        elif dataset_name.lower() == 'cifar10':
            return self._load_cifar10(transform)
        elif dataset_name.lower() == 'har':
            return self._load_har_dataset()
        elif dataset_name.lower() == 'iot':
            return self._load_industrial_iot()
        elif dataset_name.lower() == 'chestxray':
            return self._load_chestxray(transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _load_fashion_mnist(self, transform=None):
        """Load FashionMNIST dataset"""
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        train_dataset = FashionMNIST(
            root=self.data_root, 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = FashionMNIST(
            root=self.data_root, 
            train=False, 
            download=True, 
            transform=transform
        )
        
        return train_dataset, test_dataset
    
    def _load_cifar10(self, transform=None):
        """Load CIFAR-10 dataset"""
        if transform is None:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transform_test = transform
            
        train_dataset = CIFAR10(
            root=self.data_root, 
            train=True, 
            download=True, 
            transform=transform_train
        )
        
        test_dataset = CIFAR10(
            root=self.data_root, 
            train=False, 
            download=True, 
            transform=transform_test
        )
        
        return train_dataset, test_dataset
    
    def _download_file(self, url, save_path):
        """Download a file from URL to save_path"""
        if os.path.exists(save_path):
            print(f"File already exists: {save_path}")
            return
            
        print(f"Downloading {url} to {save_path}")
        with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
    
    def _load_har_dataset(self):
        """
        Load Human Activity Recognition dataset
        Source: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
        """
        har_dir = os.path.join(self.data_root, 'HAR')
        os.makedirs(har_dir, exist_ok=True)
        
        zip_path = os.path.join(har_dir, 'UCI_HAR_Dataset.zip')
        
        # Download dataset if not present
        if not os.path.exists(os.path.join(har_dir, 'UCI HAR Dataset')):
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
            if not os.path.exists(zip_path):
                self._download_file(url, zip_path)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(har_dir)
                
        # Load data
        train_dir = os.path.join(har_dir, 'UCI HAR Dataset', 'train', 'Inertial Signals')
        test_dir = os.path.join(har_dir, 'UCI HAR Dataset', 'test', 'Inertial Signals')
        
        # Load training data
        X_train = []
        signal_types = ['body_acc_x', 'body_acc_y', 'body_acc_z', 
                       'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                       'total_acc_x', 'total_acc_y', 'total_acc_z']
        
        for signal in signal_types:
            file_path = os.path.join(train_dir, f"{signal}_train.txt")
            signal_data = pd.read_csv(file_path, delim_whitespace=True, header=None).values
            X_train.append(signal_data)
        
        # Stack channels
        X_train = np.dstack(X_train)
        
        # Load training labels
        y_train_path = os.path.join(har_dir, 'UCI HAR Dataset', 'train', 'y_train.txt')
        y_train = pd.read_csv(y_train_path, header=None, delim_whitespace=True).values.ravel() - 1  # Labels from 0-5
        
        # Load test data
        X_test = []
        for signal in signal_types:
            file_path = os.path.join(test_dir, f"{signal}_test.txt")
            signal_data = pd.read_csv(file_path, delim_whitespace=True, header=None).values
            X_test.append(signal_data)
            
        X_test = np.dstack(X_test)
        
        # Load test labels
        y_test_path = os.path.join(har_dir, 'UCI HAR Dataset', 'test', 'y_test.txt')
        y_test = pd.read_csv(y_test_path, header=None, delim_whitespace=True).values.ravel() - 1
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        return train_dataset, test_dataset
    
    def _download_and_extract_iot(self):
        """Download and extract industrial IoT dataset"""
        iot_dir = os.path.join(self.data_root, 'IOT')
        os.makedirs(iot_dir, exist_ok=True)
        
        # This is an example using the SKAB dataset as industrial IoT data
        # https://github.com/waico/SKAB
        
        files = [
            ('https://raw.githubusercontent.com/waico/SKAB/master/SKAB/valve1.csv', 'valve1.csv'),
            ('https://raw.githubusercontent.com/waico/SKAB/master/SKAB/valve2.csv', 'valve2.csv'),
            ('https://raw.githubusercontent.com/waico/SKAB/master/SKAB/other.csv', 'other.csv'),
            ('https://raw.githubusercontent.com/waico/SKAB/master/SKAB/pump.csv', 'pump.csv')
        ]
        
        for url, filename in files:
            file_path = os.path.join(iot_dir, filename)
            if not os.path.exists(file_path):
                self._download_file(url, file_path)
                
        return iot_dir
    
    def _load_industrial_iot(self):
        """Load Industrial IoT dataset"""
        iot_dir = self._download_and_extract_iot()
        
        # Load and combine files
        files = ['valve1.csv', 'valve2.csv', 'other.csv', 'pump.csv']
        dfs = []
        
        for filename in files:
            file_path = os.path.join(iot_dir, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Extract features and labels
        # For IoT anomaly detection: 0 = normal, 1 = anomaly
        features = combined_df.drop(['datetime', 'anomaly', 'changepoint'], axis=1, errors='ignore').values
        labels = combined_df['anomaly'].fillna(0).astype(int).values
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        return train_dataset, test_dataset
    
    def _download_chestxray_sample(self):
        """Download a small sample of ChestX-ray14 dataset (for demonstration)"""
        xray_dir = os.path.join(self.data_root, 'ChestXray')
        os.makedirs(xray_dir, exist_ok=True)
        
        # For demonstration, we'll use a tiny subset of the NIH Chest X-ray dataset
        # In a real implementation, you would download from the official source
        # https://nihcc.app.box.com/v/ChestXray-NIHCC
        
        # This is a simplified version for the example
        # For a real implementation, use the actual ChestX-ray14 dataset
        
        sample_dir = os.path.join(xray_dir, 'sample')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Simulate a small dataset with random data for demonstration
        # In reality, you would download the actual X-ray images
        
        # Generate 100 random grayscale images and labels for demonstration
        np.random.seed(42)
        num_samples = 100
        img_size = 64  # Smaller size for demonstration
        
        # Generate synthetic X-ray-like images (random noise)
        images = np.random.randn(num_samples, img_size, img_size) * 0.1 + 0.5
        images = np.clip(images, 0, 1)
        
        # Generate binary labels for 14 conditions (multi-label classification)
        labels = np.random.randint(0, 2, size=(num_samples, 14))
        
        # Save generated data
        data_file = os.path.join(xray_dir, 'chestxray_sample.npz')
        np.savez(data_file, images=images, labels=labels)
        
        return data_file
    
    def _load_chestxray(self, transform=None):
        """Load ChestX-ray14 dataset (sample for demonstration)"""
        data_file = self._download_chestxray_sample()
        
        # Load generated sample data
        data = np.load(data_file)
        images = data['images']
        labels = data['labels']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
        
        # Create custom datasets
        train_dataset = ChestXrayDataset(X_train, y_train, transform=transform)
        test_dataset = ChestXrayDataset(X_test, y_test, transform=transform)
        
        return train_dataset, test_dataset
    
    def create_client_datasets(self, dataset_name, transform=None):
        """
        Partition dataset into client datasets
        
        Args:
            dataset_name: Name of dataset to load
            transform: Optional transform to apply
            
        Returns:
            client_datasets: List of datasets for each client
            test_dataset: Common test dataset
        """
        train_dataset, test_dataset = self.get_dataset(dataset_name, transform)
        
        if self.iid:
            # IID: randomly shuffle and split
            client_datasets = self._create_iid_clients(train_dataset)
        else:
            # Non-IID: sort by labels and distribute in a biased way
            client_datasets = self._create_non_iid_clients(train_dataset, dataset_name)
            
        return client_datasets, test_dataset
    
    def _create_iid_clients(self, dataset):
        """Partition dataset in IID fashion"""
        num_items = len(dataset)
        items_per_client = num_items // self.num_clients
        client_datasets = []
        
        indices = list(range(num_items))
        np.random.shuffle(indices)
        
        for i in range(self.num_clients):
            start_idx = i * items_per_client
            end_idx = start_idx + items_per_client if i < self.num_clients - 1 else num_items
            client_indices = indices[start_idx:end_idx]
            client_datasets.append(Subset(dataset, client_indices))
            
        return client_datasets
    
    def _create_non_iid_clients(self, dataset, dataset_name):
        """Partition dataset in non-IID fashion"""
        if dataset_name.lower() in ['fashion_mnist', 'cifar10']:
            return self._create_label_skew_clients(dataset)
        elif dataset_name.lower() in ['har', 'iot', 'chestxray']:
            return self._create_feature_skew_clients(dataset)
        else:
            # Default to label skew for unknown datasets
            return self._create_label_skew_clients(dataset)
    
    def _create_label_skew_clients(self, dataset):
        """Create non-IID distribution with label skew"""
        # Group by label
        labels = self._get_dataset_labels(dataset)
        label_indices = {}
        
        for idx, label in enumerate(labels):
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(idx)
            
        # Create Dirichlet distribution for label assignment
        num_classes = len(label_indices)
        client_datasets = [[] for _ in range(self.num_clients)]
        
        # Use Dirichlet distribution to allocate data non-uniformly
        alpha = 0.5  # Lower alpha = more skew
        for k in range(num_classes):
            label_k = list(label_indices.keys())[k]
            idx_k = label_indices[label_k]
            np.random.shuffle(idx_k)
            
            # Generate proportion of each class for each client
            proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))
            proportions = np.array([p * len(idx_k) for p in proportions])
            proportions = proportions.astype(int)
            
            # Fix rounding errors
            proportions[-1] = len(idx_k) - np.sum(proportions[:-1])
            
            # Assign samples to clients
            start_idx = 0
            for i in range(self.num_clients):
                count = proportions[i]
                client_datasets[i].extend(idx_k[start_idx:start_idx + count])
                start_idx += count
                
        # Convert to Subset datasets
        return [Subset(dataset, indices) for indices in client_datasets]
    
    def _create_label_skew_clients(self, dataset):
        """Create non-IID distribution with label skew"""
        # Group by label
        labels = self._get_dataset_labels(dataset)
        label_indices = {}
        
        for idx, label in enumerate(labels):
            label_key = int(label) if torch.is_tensor(label) else label
            if label_key not in label_indices:
                label_indices[label_key] = []
            label_indices[label_key].append(idx)
            
        # Create Dirichlet distribution for label assignment
        num_classes = len(label_indices)
        client_datasets = [[] for _ in range(self.num_clients)]
        
        # Use Dirichlet distribution to allocate data non-uniformly
        alpha = 0.5  # Lower alpha = more skew
        for k in range(num_classes):
            try:
                label_k = list(label_indices.keys())[k]
                idx_k = label_indices[label_k]
                np.random.shuffle(idx_k)
                
                # Generate proportion of each class for each client
                proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))
                proportions = np.array([max(1, int(p * len(idx_k))) for p in proportions])  # Ensure at least 1 sample
                
                # Adjust if sum is too large
                while sum(proportions) > len(idx_k):
                    # Find index with largest proportion and reduce it
                    idx_max = np.argmax(proportions)
                    if proportions[idx_max] > 1:  # Ensure we don't reduce below 1
                        proportions[idx_max] -= 1
                
                # Assign samples to clients
                start_idx = 0
                for i in range(self.num_clients):
                    count = proportions[i]
                    if start_idx + count <= len(idx_k):
                        client_datasets[i].extend(idx_k[start_idx:start_idx + count])
                        start_idx += count
                    
            except Exception as e:
                print(f"Warning: Error distributing class {k}: {e}")
        
        # Ensure each client gets at least one sample
        for i in range(self.num_clients):
            if len(client_datasets[i]) == 0:
                print(f"Warning: Client {i} received no samples. Assigning fallback samples.")
                # Find a client with more samples and take one
                for j in range(self.num_clients):
                    if len(client_datasets[j]) > 1:
                        client_datasets[i].append(client_datasets[j].pop())
                        break
        
        # Convert to Subset datasets
        return [Subset(dataset, indices) for indices in client_datasets]
    
    def _get_dataset_labels(self, dataset):
        """Extract labels from dataset"""
        if hasattr(dataset, 'targets'):
            return dataset.targets
        elif hasattr(dataset, 'tensors'):  # TensorDataset
            return dataset.tensors[1].numpy()
        else:
            # Extract labels manually by iterating (slow)
            return [y for _, y in [dataset[i] for i in range(len(dataset))]]


class ChestXrayDataset(Dataset):
    """Custom dataset for ChestX-ray14"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        return image, torch.FloatTensor(label)  # Multi-label classification