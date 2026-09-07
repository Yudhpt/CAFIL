"""Shared image transform helpers for Stage 2 training."""

from __future__ import annotations

from torchvision import transforms as T
from torchvision.transforms import RandAugment


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 224):
    return T.Compose(
        [
            T.RandomResizedCrop(int(image_size), scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def build_strong_train_transform(image_size: int = 224):
    return T.Compose(
        [
            T.RandomResizedCrop(int(image_size), scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            RandAugment(num_ops=2, magnitude=10),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def make_train_transform(image_size: int = 224, aug_type: str = "std"):
    if str(aug_type).strip().lower() == "strong":
        return build_strong_train_transform(image_size=int(image_size))
    return build_train_transform(image_size=int(image_size))
