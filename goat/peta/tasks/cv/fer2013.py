import io
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets import load_dataset


class CustomFER2013Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = Image.open(io.BytesIO(sample["img_bytes"])).convert("L")  # Convert to PIL image
        label = sample["labels"]

        if self.transform:
            image = self.transform(image)

        return image, label


class FER2013:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=16,
    ):

        # location = os.path.join("~/data", "FER2013")

        # Load the FER2013 dataset using Hugging Face datasets library
        fer2013 = load_dataset("Jeneral/fer-2013", split="train")

        # Instantiate the custom PyTorch training dataset
        self.train_dataset = CustomFER2013Dataset(fer2013, transform=preprocess)

        # Use PyTorch DataLoader to create an iterator over training batches
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # Load the FER2013 test dataset using Hugging Face datasets library
        fer2013_test = load_dataset("Jeneral/fer-2013", split="test")

        # Instantiate the custom PyTorch test dataset
        self.test_dataset = CustomFER2013Dataset(fer2013_test, transform=preprocess)

        # Use PyTorch DataLoader to create an iterator over test batches
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = [
            ["angry"],
            ["disgusted"],
            ["fearful"],
            ["happy", "smiling"],
            ["sad", "depressed"],
            ["surprised", "shocked", "spooked"],
            ["neutral", "bored"],
        ]
