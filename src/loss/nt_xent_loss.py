import torch
import torch.nn as nn
import numpy as np


class NTXentLoss(nn.Module):

    """Normalized Temperature-scaled Cross Entropy Loss used in SimCLR"""

    def __init__(self, device: str, batch_size: int, temperature: float, use_cosine_similarity: bool):
        super(NTXentLoss, self).__init__()
        self._batch_size = batch_size
        self._temperature = temperature
        self._device = device
        self._softmax = nn.Softmax(dim=-1)
        self._mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self._similarity_function = self._get_similarity_function(use_cosine_similarity)
        self._criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity: bool):
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self) -> torch.Tensor:
        diag = np.eye(2 * self._batch_size)
        l1 = np.eye((2 * self._batch_size), 2 * self._batch_size, k=-self._batch_size)
        l2 = np.eye((2 * self._batch_size), 2 * self._batch_size, k=self._batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self._device)

    @staticmethod
    def _dot_simililarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis: torch.Tensor, zjs: torch.Tensor) -> torch.Tensor:
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self._similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self._batch_size)
        r_pos = torch.diag(similarity_matrix, -self._batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self._batch_size, 1)

        negatives = similarity_matrix[self._mask_samples_from_same_repr].view(2 * self._batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self._temperature

        labels = torch.zeros(2 * self._batch_size).to(self._device).long()
        loss = self._criterion(logits, labels)

        return loss / (2 * self._batch_size)
