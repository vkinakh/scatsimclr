from typing import Tuple
import math

import numpy as np

import torch
from torch.utils.data import Dataset

from .datasets import SUPPORTED_DATASETS, get_dataset
from .augmentor import PatchAugmentor


def retrive_permutations(classes: int) -> np.ndarray:
    all_perm = np.load('./data/permutations_%d.npy' % (classes))
    if all_perm.min() == 1:
        all_perm = all_perm - 1

    return all_perm


class JigsawDataset(Dataset):
    """Dataset with jigsaw transformation"""

    def __init__(self, dataset: str, input_size: Tuple[int, int, int],
                 transform=None,
                 train: bool = True,
                 permutations: int = 35):
        """

        Args:
            dataset: dataset to use

            input_size: input image size

            transform: transforms to apply

            train: if True, train dataset is loaded

            permutations: number of permutations
        """

        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f'Unsupported dataset. `dataset` should be in [{", ".join(SUPPORTED_DATASETS)}]')

        if len(input_size) != 3:
            raise ValueError('Incorrect `input_size`. It should be (H, W, C)')

        self._transform = transform
        self._dataset = get_dataset(dataset, train, download=True, unlabeled=True)
        self._transform_patch = PatchAugmentor(input_size)

        self._permutations = retrive_permutations(permutations)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        img, _ = self._dataset[item]
        patches, order = self._get_patches(img)

        img_aug1, img_aug2 = self._transform(img)
        return img_aug1, img_aug2, patches, order

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_patches(self, img: torch.Tensor) -> Tuple[torch.Tensor, int]:
        img_tr = img

        s = float(img_tr.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = int(n / 3)
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([math.ceil(c[1] - a), math.ceil(c[0] - a), int(c[1] + a), int(c[0] + a)]).astype(int)
            tile = img_tr.crop(c.tolist())
            tile = self._transform_patch(tile)
            tiles[n] = tile

        order = np.random.randint(len(self._permutations))
        data = [tiles[self._permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)
        return data, int(order)
