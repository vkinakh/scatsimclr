from typing import Tuple

import PIL
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets

from .gaussian_blur import GaussianBlur
from .dataset_stats import DATASET_STATS
from .cifar_20 import CIFAR20


class SimCLRDataTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class UnsupervisedDatasetWrapper:
    """Dataset wrapper for unsupervised image classification"""

    DATASETS = ['stl10', 'cifar10', 'cifar20']

    def __init__(self,
                 batch_size: int,
                 valid_size: float,
                 input_size: Tuple[int, int, int],
                 dataset: str):
        """

        Args:
            batch_size: batch size to use in train and validation data loaders

            valid_size: percentage of the data to be used in validation set. Should be in range (0, 1)

            input_size: input size of the image. Should be Tuple (H, W, C), H - height, W - width, C - channels

            dataset: dataset to use. Available datasets are in DATASET
        """

        if batch_size <= 0:
            raise ValueError('Incorrect `batch_size` value. `batch_size` should be > 0')

        if valid_size < 0 or valid_size >= 1:
            raise ValueError('Incorrect `valid_size`. `valid_size` should be in range (0, 1)')

        if dataset not in self.DATASETS:
            raise ValueError(f'Incorrect `dataset`. Possible options: [{", ".join(self.DATASETS)}]')

        if len(input_size) != 3:
            raise ValueError('Incorrect `input_size`. `input_size` should be tuple (H, W, C)')

        self._batch_size = batch_size
        self._valid_size = valid_size
        self._input_size = input_size
        self._dataset = dataset

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        data_augmentations = SimCLRDataTransform(self._get_augmentations())

        if self._dataset == 'stl10':
            train_dataset = datasets.STL10('./data', split='train+unlabeled',
                                           download=True,
                                           transform=data_augmentations)
        elif self._dataset == 'cifar10':
            train_dataset = datasets.CIFAR10('./data', train=True,
                                             transform=data_augmentations,
                                             download=True)
        elif self._dataset == 'cifar20':
            train_dataset = CIFAR20('./data/cifar-20', train=True,
                                    transform=data_augmentations,
                                    download=True)
        else:
            raise ValueError('Incorrect dataset')

        train_loader, valid_loader = self._get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_train_validation_data_loaders(self, train_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self._valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size,
                                  sampler=train_sampler, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self._batch_size,
                                  sampler=valid_sampler, drop_last=True)
        return train_loader, valid_loader

    def _get_augmentations(self) -> transforms.Compose:
        stats = DATASET_STATS[self._dataset]

        color = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

        # here it's assumed that height == width
        h, w = self._input_size[:2]
        blur_kernel_size = 2 * int(.05 * h) + 1

        size = (h, w)
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
        return augmentations
