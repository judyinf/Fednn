import torch
from torch.utils.data import Dataset
import numpy as np
import os


class HARDataset(Dataset):
    def __init__(self, root, train: bool = True, transform=None):
        """
        Args:
            root (str): Root directory containing the train and test folders.
            train (bool): If True, loads data from the train folder, else from the test folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.train = train
        self.transform = transform

        # Determine the folder based on train/test mode
        folder = "train" if self.train else "test"
        self.data_dir = os.path.join(self.root, folder)

        # Load data and labels
        self.data, self.labels = self.load_data()

    def load_data(self):
        """
        Load feature and label data from files.
        """
        # Feature file names
        feature_files = [
            "body_acc_x_{}.txt", "body_acc_y_{}.txt", "body_acc_z_{}.txt",
            "body_gyro_x_{}.txt", "body_gyro_y_{}.txt", "body_gyro_z_{}.txt",
            "total_acc_x_{}.txt", "total_acc_y_{}.txt", "total_acc_z_{}.txt"
        ]

        # Replace 'train' or 'test' in file names based on mode
        suffix = "train" if self.train else "test"
        feature_files = [file.format(suffix) for file in feature_files]

        # Load features
        data = []
        for file in feature_files:
            file_path = os.path.join(self.data_dir, "Inertial Signals", file)
            feature_data = np.loadtxt(file_path)  # Shape: (num_samples, 128)
            data.append(feature_data)
        data = np.stack(data, axis=-1)  # Shape: (num_samples, 128, 9)

        # Load labels
        labels_file = os.path.join(self.data_dir, f"y_{suffix}.txt")
        labels = np.loadtxt(labels_file).astype(int)  # Shape: (num_samples,)

        return data, labels

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.
        """
        # Get features and labels for the given index
        sample = self.data[idx]
        label = self.labels[idx]

        # Apply transformations if specified
        if self.transform:
            sample = self.transform(sample)

        return sample, label
