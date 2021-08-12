from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models import LogisticRegression


class LogisticRegressionEvaluator:

    """Logistic regression evaluator. Trains one-layer linear classifier, to classify embeddings"""

    def __init__(self, n_features: int, n_classes: int, device: str, batch_size: int):
        """
        Args:
            n_features: number of input features

            n_classes: number of classes

            device: device

            batch_size: batch size to use when training
        """

        self._model = LogisticRegression(n_features, n_classes).to(device)
        self._scaler = StandardScaler()
        self._device = device
        self._batch_size = batch_size

    def run_evaluation(self, train_data: np.ndarray, train_labels: np.ndarray,
                       test_data: np.ndarray, test_labels: np.ndarray,
                       epochs: int) -> float:
        """Runs evaluation

        Args:
            train_data: training vectors

            train_labels: training labels

            test_data: test vectors

            test_labels: test labels

            epochs: number of training epochs

        Returns:
            float: test accuracy
        """

        # standard dataset, create dataloaders
        train_data, test_data = self._standard_dataset(train_data, test_data)
        train = TensorDataset(torch.from_numpy(train_data),
                              torch.from_numpy(train_labels).type(torch.long))
        train_loader = DataLoader(train, batch_size=self._batch_size, shuffle=False)

        test = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels).type(torch.long))
        test_loader = DataLoader(test, batch_size=self._batch_size, shuffle=False)

        weight_decay = 1e-4
        lr = 3e-4

        optimizer = torch.optim.Adam(self._model.parameters(), lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # train model
        for e in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self._device), batch_y.to(self._device)

                optimizer.zero_grad()
                logits = self._model(batch_x)
                loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()

        accuracy = self._eval(test_loader)
        return accuracy

    def _standard_dataset(self, train_data: np.ndarray,
                          test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._scaler.fit(train_data)
        train_data = self._scaler.transform(train_data)
        test_data = self._scaler.transform(test_data)
        return train_data, test_data

    def _eval(self, loader: DataLoader) -> float:
        correct = 0
        total = 0

        with torch.no_grad():
            self._model.eval()

            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self._device), batch_y.to(self._device)
                logits = self._model(batch_x)

                predicted = torch.argmax(logits, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            final_acc = correct / total
            self._model.train()
            return final_acc
