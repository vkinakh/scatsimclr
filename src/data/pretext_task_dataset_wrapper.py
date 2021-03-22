from typing import Tuple

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.data import RotationDataset, JigsawDataset
from src.data.datasets import SUPPORTED_DATASETS
from src.data import ContrastiveAugmentor


class PretextTaskDatasetWrapper:
    """Dataset wrapper for unsupervised image classification"""

    def __init__(self, dataset: str,
                 input_size: Tuple[int, int, int],
                 batch_size: int,
                 valid_size: float,
                 jigsaw: bool,
                 rotation: bool):

        """
        Args:
            dataset: dataset to use. Available datasets are in SUPPORTED_DATASETS

            input_size: input size of the image. Should be Tuple (H, W, C), H - height, W - width, C - channels

            batch_size: batch size to use in train and validation data loaders

            valid_size: percentage of the data to be used in validation set. Should be in range (0, 1)

            jigsaw: if True, jigsaw pretext task is used

            rotation: is True, jigsaw pretext task is used

        Raises:
            ValueError: If both `jigsaw` and `rotation` or none are selected

            ValueError: If `batch_size` is negative

            ValueError: If `valid_size` is not in range (0, 1)

            ValueError: If `dataset` is not in SUPPORTED_DATASETS

            ValueError: If `input_size` is not (H, W, C)
        """

        if sum((jigsaw, rotation)) > 1 or sum((jigsaw, rotation)) == 0:
            raise ValueError('Only one pretext task should be used at the time')

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

        transform = ContrastiveAugmentor(dataset, input_size)

        if jigsaw:
            self._dataset = JigsawDataset(dataset=dataset, input_size=input_size,
                                          transform=transform)
        elif rotation:
            self._dataset = RotationDataset(dataset=dataset, input_size=input_size,
                                            transform=transform)

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        n = len(self._dataset)
        indices = list(range(n))
        np.random.shuffle(indices)

        split = int(np.floor(self._valid_size * n))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(self._dataset, batch_size=self._batch_size, sampler=train_sampler)
        valid_loader = DataLoader(self._dataset, batch_size=self._batch_size, sampler=valid_sampler)
        return train_loader, valid_loader
