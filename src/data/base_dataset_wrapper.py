from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from .datasets import SUPPORTED_DATASETS


class BaseDatasetWrapper(ABC):

    """Base dataset wrapper"""

    def __init__(self,
                 batch_size: int,
                 valid_size: float,
                 input_size: Tuple[int, int, int],
                 dataset: str):
        """
        Args:
            batch_size: batch size to use in dataloader

            valid_size: percentage of the data, to be used

            input_size: input image size

            dataset: dataset to use

        Raises:
            ValueError: if `dataset` is not supported

            ValueError: if `input_size` is incorrect

            ValueError: if `batch_size` if negative

            ValueError: if `valid_size` is not in range (0, 1)
        """

        if dataset not in SUPPORTED_DATASETS:
            raise ValueError('Unsupported dataset')

        if len(input_size) != 3:
            raise ValueError('Input size should be in form (H, W, C)')

        if batch_size <= 0:
            raise ValueError('Incorrect `batch_size` value. It should be positive')

        if valid_size <= 0 or valid_size >= 1:
            raise ValueError('Incorrect `valid_size`. It should be in range (0, 1)')

        self._batch_size = batch_size
        self._valid_size = valid_size
        self._input_size = input_size
        self._dataset = self.get_dataset(dataset)

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Returns: train and valid dataloaders for specified dataset
        """

        n = len(self._dataset)
        indices = list(range(n))
        np.random.shuffle(indices)

        split = int(np.floor(self._valid_size * n))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(self._dataset, batch_size=self._batch_size, sampler=train_sampler, num_workers=4,
                                  drop_last=True)
        valid_loader = DataLoader(self._dataset, batch_size=self._batch_size, sampler=valid_sampler, num_workers=4,
                                  drop_last=True)
        return train_loader, valid_loader

    @abstractmethod
    def get_dataset(self, dataset: str) -> Dataset:
        pass
