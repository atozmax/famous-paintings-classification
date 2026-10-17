import os
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.labels import ARTISTS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class NumericImageFolder(datasets.ImageFolder):
    """Class index is the numeric folder name, not a string sort of that name."""

    def find_classes(self, directory):
        classes = [
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and entry.name.isdigit()
        ]
        classes.sort(key=int)
        return classes, {name: int(name) for name in classes}


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]
        return self.transform(image), label


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(248),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize(248),
            transforms.RandomAffine(
                degrees=8,
                scale=(0.94, 1.06),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def list_images(folder):
    folder = Path(folder)
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_loaders(base_dataset, train_idx, val_idx, batch_size, num_workers):
    train_set = TransformSubset(Subset(base_dataset, train_idx), train_transform())
    val_set = TransformSubset(Subset(base_dataset, val_idx), eval_transform())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def artist_name(class_id):
    return ARTISTS[int(class_id)]
