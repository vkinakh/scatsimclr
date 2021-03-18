from typing import Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import cv2

from .datasets import SUPPORTED_DATASETS, get_dataset


class RotationDataset(Dataset):
    """Dataset with rotation transformation"""

    def __init__(self, dataset: str, input_size: Tuple[int, int, int],
                 transform=None,
                 train: bool = True):

        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f'Unsupported dataset. `dataset` should be in [{", ".join(SUPPORTED_DATASETS)}]')

        if len(input_size) != 3:
            raise ValueError('Incorrect `input_size`. It should be (H, W, C)')

        self._dataset = get_dataset(dataset, train, True, True)
        self._transform = transform
        h, w = input_size[:2]
        self._transform_patch = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor()
        ])

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img, _ = self._dataset[item]

        rotated_imgs = torch.stack([
            self._transform(img),
            self._transform(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
            self._transform(cv2.rotate(img, cv2.ROTATE_180)),
            self._transform(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
        ], dim=0)
        rotation_labels = torch.LongTensor([0, 1, 2, 3])

        img_aug1 = self._transform(img)
        img_aug2 = self._transform(img)
        return img_aug1, img_aug2, rotated_imgs, rotation_labels

    def __len__(self) -> int:
        return len(self._dataset)
