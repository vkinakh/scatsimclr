from typing import NoReturn, Dict
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import UnsupervisedDatasetWrapper
from src.trainer import BaseTrainer
from src.trainer.base_trainer import save_config_file


def get_device() -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


class ScatSimCLRTrainer(BaseTrainer):

    def __init__(self, config: Dict):
        super(ScatSimCLRTrainer, self).__init__(config)

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

    def train(self) -> NoReturn:

        # load dataset
        dataset_wrapper = UnsupervisedDatasetWrapper(batch_size=self._config['batch_size'],
                                                     input_shape=eval(self._config['dataset']['input_shape']),
                                                     valid_size=self._config['dataset']['valid_size'],
                                                     dataset=self._config['dataset']['dataset'])
        train_loader, valid_loader = dataset_wrapper.get_data_loaders()

        # create and if needed load model
        model = self._get_embeddings_model(self._config['model']['base_model'])
        model = self._load_weights(model)
        model.to(self._device)

        # create optimizer and scheduler
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
        # checkpoints_folder = Path('./runs') / f"{self._config['fine_tune_from']}/checkpoints"
        checkpoints_file = Path(self._config['fine_tune_from'])

        #if checkpoints_folder.exists():
        if checkpoints_file.exists():
            state_dict = torch.load(checkpoints_file)  # torch.load(checkpoints_folder / 'model_final.pth')
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
