from typing import Tuple

from torch.utils.data import Dataset

from src.data import RotationDataset, JigsawDataset
from src.data import ContrastiveAugmentor
from .base_dataset_wrapper import BaseDatasetWrapper


class PretextTaskDatasetWrapper(BaseDatasetWrapper):

    """Dataset wrapper for pretext task learning"""

    def __init__(self,
                 batch_size: int,
                 valid_size: float,
                 input_size: Tuple[int, int, int],
                 dataset: str,
                 jigsaw: bool,
                 rotation: bool):
        """
        Args:
            batch_size: batch size to use in train and validation data loaders

            valid_size: percentage of the data to be used in validation set. Should be in range (0, 1)

            input_size: input size of the image. Should be Tuple (H, W, C), H - height, W - width, C - channels

            dataset: dataset to use. Available datasets are in SUPPORTED_DATASETS

            jigsaw: if True, jigsaw pretext task is used

            rotation: if True, jigsaw pretext task is used

        Raises:
            ValueError: If both `jigsaw` and `rotation` or none are selected
        """

        if sum((jigsaw, rotation)) > 1 or sum((jigsaw, rotation)) == 0:
            raise ValueError('Only one pretext task should be used at the time')

        self._jigsaw = jigsaw
        self._rotation = rotation

        super().__init__(batch_size, valid_size, input_size, dataset)

    def get_dataset(self, dataset: str) -> Dataset:
        transform = ContrastiveAugmentor(dataset, self._input_size)

        if self._jigsaw:
            return JigsawDataset(dataset=dataset, input_size=self._input_size, transform=transform)
        elif self._rotation:
            return RotationDataset(dataset=dataset, input_size=self._input_size, transform=transform)
