from typing import Tuple

import PIL

import torch
import torchvision.transforms as transforms

from .datasets import DATASET_STATS, SUPPORTED_DATASETS
from .gaussian_blur import GaussianBlur


class SimCLRDataTransform:

    """Applies augmentations to sample two times, as described in SimCLR paper"""

    def __init__(self, transform: transforms.Compose):
        self.transform = transform

    def __call__(self, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class ContrastiveAugmentor:

    """Applies augmentation for contrastive learning, as in SimCLR paper"""

    def __init__(self, dataset: str, input_size: Tuple[int, int, int]):
        """
        Args:
            dataset: dataset to apply augmentations to

            input_size: input image size

        Raises:
            ValueError: if specified dataset is unsupported
        """

        if dataset not in SUPPORTED_DATASETS:
            raise ValueError('Unsupported dataset')

        stats = DATASET_STATS[dataset]

        h, w = input_size[:2]
        size = (h, w)
        blur_kernel_size = 2 * int(.05 * h) + 1
        color = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

        augmentations = transforms.Compose([
            transforms.Resize(size, interpolation=PIL.Image.LANCZOS),
            transforms.RandomResizedCrop(size=size, interpolation=PIL.Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=blur_kernel_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=stats['mean'], std=stats['std'])
        ])
        self._augmentations = SimCLRDataTransform(augmentations)

    def __call__(self, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._augmentations(sample)


class ValidAugmentor:

    """Applies augmentation for validation and testing"""

    def __init__(self, dataset: str, input_size: Tuple[int, int, int]):
        """
        Args:
            dataset: dataset to apply augmentations to

            input_size: input image size

        Raises:
            ValueError: if specified dataset is unsupported
        """

        if dataset not in SUPPORTED_DATASETS:
            raise ValueError('Unsupported dataset')

        stats = DATASET_STATS[dataset]

        h, w = input_size[:2]
        size = (h, w)

        self._augmentations = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean=stats['mean'], std=stats['std'])
        ])

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return self._augmentations(sample)


class PatchAugmentor:

    """Applies augmentations to patch"""

    def __init__(self, input_size: Tuple[int, int, int]):
        """
        Args:
            input_size: input image size
        """

        h, w = input_size[:2]
        size = (h, w)

        self._augmentations = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor()
        ])

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return self._augmentations(sample)
