import os

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.kaggle_casting_dataset import KaggleCastingDataset
from domain.base import Module
from domain.metadata import Metadata
from logger import logger
from properties import APPLICATION_PROPERTIES


class DatasetModule(Module):

    def __init__(self, metadata: Metadata, *args, **kwargs):
        super(DatasetModule, self).__init__(*args, **kwargs)
        self.metadata = metadata

        self.name = self.metadata.dataset_name

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        # Create
        self.create()

    def create(self):
        train_dataset = None
        val_dataset = None
        test_dataset = None

        train_dataloader = None
        val_dataloader = None
        test_dataloader = None

        data_name = self.name

        if data_name == "kaggle_casting_data":
            train_dataset = KaggleCastingDataset(
                root=os.path.join(APPLICATION_PROPERTIES.DATA_DIRECTORY_PATH, "kaggle_casting_data", "train"),
                transform=transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor()
                ])
            )
            test_dataset = KaggleCastingDataset(
                root=os.path.join(APPLICATION_PROPERTIES.DATA_DIRECTORY_PATH, "kaggle_casting_data", "test"),
                transform=transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor()
                ])
            )

            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=32,
                shuffle=True
            )
            test_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=32,
                shuffle=False
            )
        elif data_name == "data_1":
            pass

        # Set
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset if val_dataset else test_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader if val_dataloader else test_dataloader
        self.test_dataloader = test_dataloader

        logger.info(f"Data selected : '{data_name}'")

        return self
