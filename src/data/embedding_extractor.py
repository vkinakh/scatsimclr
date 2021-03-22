from typing import Tuple
from enum import IntEnum

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import get_dataset, SUPPORTED_DATASETS
from .augmentor import ValidAugmentor


class EmbeddingType(IntEnum):
    """Type of embedding

    H - feature space used as features
    Z - projection space used as features
    CONCAT - concatenation of H and Z space used as features
    """

    H = 1
    Z = 2
    CONCAT = 3


class EmbeddingExtractor:

    """Extracts embeddings from images using model"""

    def __init__(self, model: nn.Module, device: str,
                 dataset: str, input_size: Tuple[int, int, int],
                 batch_size: int, *, embedding_type: EmbeddingType = EmbeddingType.H):
        """
        Args:
            model: model to compute embeddings

            device: device to load data

            dataset: dataset to compute embeddings

            input_size: input image size

            batch_size: batch size

            embedding_type: type of embeddings to compute. See EmbeddingType enumeration
        """

        if dataset not in SUPPORTED_DATASETS:
            raise ValueError('Unsupported dataset')

        if len(input_size) != 3:
            raise ValueError('Incorrect `input_size`. It should be (H, W, C)')

        self._model = model
        self._device = device
        self._dataset = dataset
        self._batch_size = batch_size
        self._embedding_type = embedding_type
        self._transform = ValidAugmentor(self._dataset, input_size)

    def get_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes embedding features, that will be used for classification and other downstream tasks

        Returns:
            Tuple: train features, train labels, test features, test labels
        """

        train_dataset = get_dataset(dataset=self._dataset, train=True,
                                    transform=self._transform, download=True)
        test_dataset = get_dataset(dataset=self._dataset, train=False,
                                   transform=self._transform, download=True)

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self._batch_size, drop_last=True)

        train_features, train_labels = self._compute_embeddings(train_loader)
        test_features, test_labels = self._compute_embeddings(test_loader)
        return train_features, train_labels, test_features, test_labels

    def _compute_embeddings(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self._device)

            labels.extend(batch_y)

            h, z = self._model(batch_x)

            if self._embedding_type == EmbeddingType.H:
                features.extend(h.cpu().detach().numpy())
            elif self._embedding_type == EmbeddingType.Z:
                features.extend(z.cpu().detach().numpy())
            elif self._embedding_type == EmbeddingType.CONCAT:
                f = torch.cat((h, z), 1)
                features.extend(f.cpu().detach().numpy())

        features = np.array(features)
        labels = np.array(labels)
        return features, labels
