from typing import NoReturn
from pathlib import Path
import shutil

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.loss import NTXentLoss
from src.data import UnsupervisedDatasetWrapper
from src.models import ResNetSimCLR, ScatSimCLR
from src.evaluation import LogisticRegressionEvaluator
from src.data import EmbeddingExtractor
from src.data.datasets import NUM_CLASSES


def get_device() -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def save_config_file(model_checkpoints_folder: Path) -> NoReturn:
    model_checkpoints_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy('./config.yaml', model_checkpoints_folder / 'config.yaml')


class ScatSimCLRTrainer:

    EMBEDDINGS_MODELS = ['resnet18', 'resnet50', 'scatsimclr8', 'scatsimclr12', 'scatsimclr16', 'scatsimclr30']

    def __init__(self, config):
        self._config = config
        self._device = get_device()
        self._writer = SummaryWriter()

        self._device = get_device()
        self._nt_xent_criterion = NTXentLoss(self._device, config['batch_size'], **config['loss'])

    def _step(self, model: nn.Module, xis: torch.Tensor, xjs: torch.Tensor) -> torch.Tensor:
        xis = xis.to(self._device)
        xjs = xjs.to(self._device)

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self._nt_xent_criterion(zis, zjs)
        return loss

    def _get_embeddings_model(self, model_name: str) -> nn.Module:

        if model_name not in self.EMBEDDINGS_MODELS:
            raise ValueError('Unsupported model')

        if 'resnet' in model_name:
            return ResNetSimCLR(**self._config['model'])

        if 'scatsimclr' in model_name:
            blocks = int(model_name[10:])
            input_size = eval(self._config['dataset']['input_shape'])

            return ScatSimCLR(**self._config['model'], input_size=input_size, res_blocks=blocks)

    def train(self) -> NoReturn:

        # load dataset
        dataset_wrapper = UnsupervisedDatasetWrapper(batch_size=self._config['batch_size'],
                                                     **self._config['dataset'])
        train_loader, valid_loader = dataset_wrapper.get_data_loaders()

        # create and if needed load model
        model = self._get_embeddings_model(self._config['model']['base_model'])
        model = self._load_weights(model)

        # create optimizer and sheduler
        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self._config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        # create checkpoint and save
        checkpoint_folder = Path(self._writer.log_dir) / 'checkpoints'
        save_config_file(checkpoint_folder)

        n_iter = 0
        valid_n_iter = 0
        test_n_iter = 0
        best_valid_loss = np.inf
        best_classification_acc = -1

        for epoch_counter in range(1, self._config['epochs'] + 1):

            for (xis, xjs), _ in train_loader:
                optimizer.zero_grad()
                loss = self._step(model, xis, xjs)

                if n_iter % self._config['log_every_n_steps'] == 0:
                    self._writer.add_scalar('train_loss', loss, global_step=n_iter)
                loss.backward()

                optimizer.step()
                n_iter += 1

            # validate
            if epoch_counter % self._config['validate_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    model_path = checkpoint_folder / 'model.pth'
                    torch.save(model.state_dict(), model_path)

                self._writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # run evaluation
            if epoch_counter % self._config['eval_every_n_epochs'] == 0 or epoch_counter == 1 or \
                    epoch_counter == self._config['epochs']:
                cls_accuracy = self._test_classification(model)

                if cls_accuracy > best_classification_acc:
                    best_classification_acc = cls_accuracy
                    model_path = checkpoint_folder / f'model_{epoch_counter}.pth'
                    torch.save(model.state_dict(), model_path)
                    self._writer.add_scalar('classification_accuracy', cls_accuracy, global_step=test_n_iter)
                test_n_iter += 1

            # run scheduler
            if epoch_counter >= 10:
                scheduler.step()
            self._writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

        model_path = checkpoint_folder / 'model_final.pth'
        torch.save(model.state_dict(), model_path)

    def _load_weights(self, model: nn.Module) -> nn.Module:
        checkpoints_folder = Path('./runs') / f"{self._config['fine_tune_from']}/checkpoints"

        if checkpoints_folder.exists():
            state_dict = torch.load(checkpoints_folder / 'model_final.pth')
            model.load_state_dict(state_dict)
        else:
            print('Pre-trained weights not found. Training from scratch.')
        return model

    def _validate(self, model: nn.Module, valid_loader: DataLoader) -> float:
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs), _ in valid_loader:
                loss = self._step(model, xis, xjs, )
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss

    def _test_classification(self, model: nn.Module) -> float:
        """Tests classification with extracted features

        Args:
            model: CLR model to use to compute embeddings for classification

        Returns:
            float: classification accuracy
        """

        model.eval()

        input_size = eval(self._config['dataset']['input_size'])
        dataset = self._config['dataset']['dataset']
        epochs = 100

        extractor = EmbeddingExtractor(model, device=self._device,
                                       dataset=dataset,
                                       input_size=input_size, batch_size=self._config['batch_size'])
        train_data, train_labels, test_data, test_labels = extractor.get_features()

        evaluator = LogisticRegressionEvaluator(n_features=train_data.shape[1],
                                                n_classes=NUM_CLASSES[dataset],
                                                device=self._device, batch_size=64)
        accuracy = evaluator.run_evaluation(test_data, train_labels, test_data, test_labels, epochs)
        return accuracy
