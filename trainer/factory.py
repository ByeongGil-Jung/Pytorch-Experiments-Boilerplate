from torch import optim
from torch.nn import functional as F
import torch

from dataset.factory import DatasetModule
from domain.base import Module, Hyperparameters
from domain.metadata import Metadata
from model.factory import ModelModule
from logger import logger
from trainer.base import TrainerBase
from trainer.cnn_custom_trainer import CNNCustomTrainer


class TrainerModule(Module):

    def __init__(self, metadata: Metadata, model_module: ModelModule, dataset_module: DatasetModule, *args, **kwargs):
        super(TrainerModule, self).__init__(*args, **kwargs)
        self.metadata = metadata
        self.model_module = model_module
        self.dataset_module = dataset_module

        self.trainer: TrainerBase = None

        # Create
        self.create()

    def create(self):
        # trainer_factory = cls(model_name=model_name)
        metadata = self.metadata
        model_name = self.metadata.model_name
        model_module = self.model_module
        dataset_module = self.dataset_module

        trainer = None

        if model_name == "cnn_custom":
            trainer = CNNCustomTrainer(
                metadata=metadata,
                model_module=model_module,
                dataset_module=dataset_module,
                hparams=TrainerModule.get_hyperparameters(model_name=model_name),
                **self.arg
            )
        elif model_name == "model_1":
            pass

        # Set
        self.trainer = trainer

        logger.info(f"Trainer selected : '{trainer}'")

        return self

    """
    @TODO
    Move & Modify this method
    """
    @classmethod
    def get_hyperparameters(cls, model_name):
        hyperparameters = None

        if model_name == "cnn_custom":
            hyperparameters = Hyperparameters(
                optimizer_cls=optim.Adam,
                criterion=F.binary_cross_entropy,
                n_epoch=5,
                lr=1e-3,
                hypothesis_threshold=0.5,
                weight_decay=0,
                # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        elif model_name == "model_1":
            pass

        return hyperparameters

    def do(self, mode):
        logger.info(f"Start to {mode}")
        result_dict = dict()

        if mode == "train":
            result_dict = self.trainer.train()
        elif mode == "inference":
            result_dict = self.trainer.predict()

        logger.info(f"Completed to {mode}")
        return result_dict
