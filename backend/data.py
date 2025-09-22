# src/data.py - Updated version with ImageFolder support

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision.transforms.functional as TF
from collections import Counter
import warnings

# Handle truncated images gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_path(p: str) -> bool:
    """Check if path has valid image extension"""
    return Path(p).suffix.lower() in IMG_EXTS


class ImageFolderDataset(Dataset):
    """
    Dataset that works with ImageFolder structure:
    data_root/
        train/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
        val/
            class1/
            class2/
        test/
            class1/
            class2/
    """

    def __init__(self,
                 root_dir: str,
                 transform: Optional[A.Compose] = None,
                 quick_validation: bool = True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.quick_validation = quick_validation

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        # Scan for images and build dataset
        self.samples = []
        self.class_to_idx = {}
        self._scan_directory()

        if quick_validation:
            self._quick_validate_images()
        else:
            self._validate_images()

        print(f"Dataset initialized with {len(self.samples)} valid images across {len(self.class_to_idx)} classes")

    def _scan_directory(self):
        """Scan directory structure and build class mapping"""
        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        class_dirs = sorted(class_dirs, key=lambda x: x.name)

        # Build class to index mapping
        self.class_to_idx = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}

        # Collect all image paths
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            for img_path in class_dir.iterdir():
                if img_path.is_file() and is_image_path(str(img_path)):
                    self.samples.append((str(img_path), class_idx, class_dir.name))

    def _quick_validate_images(self):
        """Quick validation - check file existence and size"""
        valid_samples = []
        print(f"Quick validation of {len(self.samples)} images...")

        for idx, (img_path, class_idx, class_name) in enumerate(self.samples):
            if idx % 1000 == 0:
                print(f"Validated {idx}/{len(self.samples)} images...")

            path_obj = Path(img_path)
            if not path_obj.exists():
                continue

            try:
                file_size = path_obj.stat().st_size
                if file_size == 0 or file_size > 50 * 1024 * 1024:  # Skip empty or >50MB
                    continue
                valid_samples.append((img_path, class_idx, class_name))
            except OSError:
                continue

        dropped = len(self.samples) - len(valid_samples)
        if dropped > 0:
            print(f"Warning: {dropped} images were dropped during quick validation")

        self.samples = valid_samples

    def _validate_images(self):
        """Full validation - actually open images"""
        valid_samples = []
        print(f"Full validation of {len(self.samples)} images...")

        for idx, (img_path, class_idx, class_name) in enumerate(self.samples):
            if idx % 100 == 0:
                print(f"Validated {idx}/{len(self.samples)} images...")

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if w <= 0 or h <= 0 or max(w, h) > 4096:
                        continue
                    valid_samples.append((img_path, class_idx, class_name))
            except Exception:
                continue

        dropped = len(self.samples) - len(valid_samples)
        if dropped > 0:
            print(f"Warning: {dropped} images were invalid and dropped")

        self.samples = valid_samples

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        class_counts = Counter([class_idx for _, class_idx, _ in self.samples])
        n_classes = len(self.class_to_idx)
        total = len(self.samples)

        weights = torch.ones(n_classes, dtype=torch.float32)
        for i in range(n_classes):
            cnt = class_counts.get(i, 0)
            if cnt == 0:
                weights[i] = 0.0
            else:
                weights[i] = total / (n_classes * cnt)

        return weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        try:
            img_path, label, class_name = self.samples[idx]

            # Load and convert image
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_array = np.array(img)

            # Apply transforms
            if self.transform:
                try:
                    augmented = self.transform(image=img_array)
                    img_tensor = augmented['image']
                except Exception as e:
                    warnings.warn(f"Transform failed for {img_path}: {e}")
                    img_tensor = TF.to_tensor(Image.fromarray(img_array))
            else:
                img_tensor = TF.to_tensor(Image.fromarray(img_array))

            return img_tensor, label, str(img_path)

        except Exception as e:
            warnings.warn(f"Failed to load image at index {idx}: {e}")
            dummy_tensor = torch.zeros((3, 224, 224))
            return dummy_tensor, 0, "corrupted_image"


class ImageFolderDataModule:
    """Data module for ImageFolder structure"""

    def __init__(self,
                 data_root: str,
                 image_size: int = 224,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 augmentation_level: str = "medium",
                 quick_validation: bool = True,
                 use_weighted_sampling: bool = False):

        self.root = Path(data_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_level = augmentation_level
        self.quick_validation = quick_validation
        self.use_weighted_sampling = use_weighted_sampling

        # Paths to train/val/test directories
        self.train_dir = self.root / "images" / "train"
        self.val_dir = self.root / "images" / "val"
        self.test_dir = self.root / "images" / "test"

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.class_to_idx = None
        self.n_classes = None

    def _get_train_transforms(self) -> A.Compose:
        """Get training data augmentations"""
        if self.augmentation_level == "light":
            transforms = [
                A.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        elif self.augmentation_level == "medium":
            transforms = [
                A.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.6, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=25, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
                A.GaussianBlur(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        else:  # heavy
            transforms = [
                A.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.5, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=35, p=0.6),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03, p=0.7),
                A.GaussianBlur(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]

        return A.Compose(transforms)

    def _get_val_transforms(self) -> A.Compose:
        """Get validation/test transforms"""
        return A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def setup(self):
        """Setup datasets"""
        try:
            print("Setting up ImageFolder data module...")

            # Load training data first to get class mappings
            if not self.train_dir.exists():
                raise FileNotFoundError(f"Training directory not found: {self.train_dir}")

            self.train_ds = ImageFolderDataset(
                self.train_dir,
                transform=self._get_train_transforms(),
                quick_validation=self.quick_validation
            )

            # Get class information from training set
            self.class_to_idx = self.train_ds.class_to_idx
            self.n_classes = len(self.class_to_idx)

            # Load validation data
            if self.val_dir.exists():
                self.val_ds = ImageFolderDataset(
                    self.val_dir,
                    transform=self._get_val_transforms(),
                    quick_validation=self.quick_validation
                )
            else:
                print("Warning: Validation directory not found")
                self.val_ds = None

            # Load test data
            if self.test_dir.exists():
                self.test_ds = ImageFolderDataset(
                    self.test_dir,
                    transform=self._get_val_transforms(),
                    quick_validation=self.quick_validation
                )
            else:
                print("Warning: Test directory not found")
                self.test_ds = None

            train_size = len(self.train_ds) if self.train_ds else 0
            val_size = len(self.val_ds) if self.val_ds else 0
            test_size = len(self.test_ds) if self.test_ds else 0

            print(f"Setup complete: {train_size} train, {val_size} val, {test_size} test")
            print(f"Number of classes: {self.n_classes}")

        except Exception as e:
            raise RuntimeError(f"Failed to setup ImageFolder datasets: {e}")

        return self

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Validation dataset not found or not initialized.")

        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("Test dataset not found or not initialized.")

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )
