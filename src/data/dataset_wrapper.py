from typing import Tuple

import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from .datasets import SUPPORTED_DATASETS, get_dataset
from .augmentor import ContrastiveAugmentor


class UnsupervisedDatasetWrapper:
    """Dataset wrapper for unsupervised image classification"""

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

        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f'Incorrect `dataset`. Possible options: [{", ".join(SUPPORTED_DATASETS)}]')

        if len(input_size) != 3:
            raise ValueError('Incorrect `input_size`. `input_size` should be tuple (H, W, C)')

        self._batch_size = batch_size
        self._valid_size = valid_size
        self._input_size = input_size
        self._dataset = dataset

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        data_augmentations = ContrastiveAugmentor(self._dataset, self._input_size)

        dataset = get_dataset(self._dataset, True, data_augmentations, True, True)
        train_loader, valid_loader = self._get_train_validation_data_loaders(dataset)
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
