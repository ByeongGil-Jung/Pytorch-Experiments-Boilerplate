import os

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.kaggle_casting_dataset import KaggleCastingDataset
from logger import logger
from properties import APPLICATION_PROPERTIES


class DatasetFactory(object):

    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    @classmethod
    def create(cls, data_name):
        dataset_factory = cls()

        train_dataset = None
        val_dataset = None
        test_dataset = None

        train_dataloader = None
        val_dataloader = None
        test_dataloader = None

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
        dataset_factory.train_dataset = train_dataset
        dataset_factory.val_dataset = val_dataset
        dataset_factory.test_dataset = test_dataset

        dataset_factory.train_dataloader = train_dataloader
        dataset_factory.val_dataloader = val_dataloader
        dataset_factory.test_dataloader = test_dataloader

        logger.info(f"Data selected : '{data_name}'")
        return dataset_factory
