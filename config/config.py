import os

from domain.base import Domain, Yaml
from properties import APPLICATION_PROPERTIES


class Config(Domain):

    def __init__(self, file_name, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.file_name = file_name

    @property
    def file_path(self):
        return os.path.join(APPLICATION_PROPERTIES.CONFIG_DIRECTORY_PATH, self.file_name)

    @property
    def yaml(self):
        return Yaml(path=self.file_path)

    @property
    def params(self):
        return self.yaml.to_hyperparameters()
