import torch
import torch.nn as nn


class LogisticRegression(nn.Module):

    """One layer linear classifier"""

    def __init__(self, n_features: int, n_classes: int):
        """
        Args:
            n_features: number of input features

            n_classes: number of output classes
        """

        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
