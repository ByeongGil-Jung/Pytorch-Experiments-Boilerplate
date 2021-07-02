from logger import logger

from domain.base import Module
from domain.metadata import Metadata
from model.cnn_custom import CNNCustom


class ModelModule(Module):

    def __init__(self, metadata: Metadata, *args, **kwargs):
        super(ModelModule, self).__init__(*args, **kwargs)
        self.metadata = metadata
        self.name = self.metadata.model_name

        self.model = None

        # Create
        self.create()

    def create(self):
        model_name = self.name
        model = None

        if model_name == "cnn_custom":
            model = CNNCustom(fc_input=19600)
        elif model_name == "model_1":
            pass

        # Set
        self.model = model

        logger.info(f"Model selected :\n'{model}'")

        return self
