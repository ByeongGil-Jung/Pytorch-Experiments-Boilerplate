from pathlib import Path
import os
import pickle
import time

import torch

from dataset.factory import DatasetModule
from domain.base import Module
from domain.metadata import Metadata
from model.factory import ModelModule
from properties import APPLICATION_PROPERTIES
from logger import logger
from utils import Utils

time.time()


class TrainerBase(Module):

    def __init__(self, metadata: Metadata, model_module: ModelModule, dataset_module: DatasetModule, hparams, _logger=logger, *args, **kwargs):
        super(TrainerBase, self).__init__(*args, **kwargs)
        self.metadata = metadata
        self.name = f"{self.metadata.dataset_name}_{self.metadata.model_name}_trainer"
        self.model_module = model_module
        self.dataset_module = dataset_module
        self.hparams = hparams

        self.model = self.model_module.model

        self.train_dataloader = self.dataset_module.train_dataloader
        self.val_dataloader = self.dataset_module.val_dataloader
        self.test_dataloader = self.dataset_module.test_dataloader

        self.logger = _logger
        self.device = self.arg.device

        # Set environments
        self.tqdm = None
        self.is_plot_showed = False
        self.tqdm_disable = False

        self.set_tqdm_env(tqdm_env=metadata.arg.env)

        # Save
        if self.arg.is_saved:
            self.create_model_directory()

        # Set model configuration
        self.model.to(self.device)
        logger.info(f"Model set to '{self.device}'")

        self.best_model_state_dict = dict()

    def set_tqdm_env(self, tqdm_env):
        tqdm_env_dict = Utils.get_tqdm_env_dict(tqdm_env=tqdm_env)

        self.tqdm = tqdm_env_dict["tqdm"]
        self.is_plot_showed = tqdm_env_dict["tqdm_disable"]
        self.tqdm_disable = tqdm_env_dict["is_plot_showed"]

    def create_model_directory(self):
        Path(self.metadata.model_dir_path).mkdir(parents=True, exist_ok=True)

    def load_best_model(self):
        best_model_file_path = self.metadata.get_best_model_file_path()

        if os.path.isfile(best_model_file_path):
            self.model.load_state_dict(torch.load(best_model_file_path, map_location=APPLICATION_PROPERTIES.DEVICE_CPU))
            self.model.to(self.device)

            logger.info(f"Succeed to load best model, device: '{self.device}'")
        else:
            logger.error(f"Failed to load best model, file not exist")

    def get_entire_record_file(self):
        entire_record_file = None
        entire_record_file_path = self.metadata.get_entire_record_file_path()

        if os.path.isfile(entire_record_file_path):
            with open(entire_record_file_path, "rb") as f:
                entire_record_file = pickle.load(f)

            logger.info(f"Succeed to get entire record file")
        else:
            logger.error(f"Failed to get entire record file, file not exist")

        return entire_record_file

    def train(self, *args, **kwargs):
        pass

    def validate(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
