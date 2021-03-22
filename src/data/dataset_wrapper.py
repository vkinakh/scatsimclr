from typing import Tuple

from torch.utils.data import Dataset

from .datasets import get_dataset
from .augmentor import ContrastiveAugmentor
from .base_dataset_wrapper import BaseDatasetWrapper


class UnsupervisedDatasetWrapper(BaseDatasetWrapper):
    """Dataset wrapper for unsupervised image classification"""

    def __init__(self,
                 batch_size: int,
                 valid_size: float,
                 input_shape: Tuple[int, int, int],
                 dataset: str):
        """
        Args:
            batch_size: batch size to use in train and validation data loaders

            valid_size: percentage of the data to be used in validation set. Should be in range (0, 1)

            input_shape: input size of the image. Should be Tuple (H, W, C), H - height, W - width, C - channels

            dataset: dataset to use. Available datasets are in SUPPORTED_DATASETS
        """

        super().__init__(batch_size, valid_size, input_shape, dataset)

    def get_dataset(self, dataset: str) -> Dataset:
        data_augmentations = ContrastiveAugmentor(dataset, self._input_size)
        return get_dataset(dataset, True, data_augmentations, True, True)
