import os

from domain.base import Module
from properties import APPLICATION_PROPERTIES


class Metadata(Module):

    def __init__(self, *args, **kwargs):
        super(Metadata, self).__init__(*args, **kwargs)
        self.name = self.arg.representation
        self.model_name = self.arg.model_name
        self.dataset_name = self.arg.dataset_name
        self.version = self.arg.version

        self.model_home_dir_path = os.path.join(APPLICATION_PROPERTIES.MODEL_DIRECTORY_PATH, "models")
        self.model_dir_name = f"{self.name}_{self.version}"
        self.model_dir_path = os.path.join(self.model_home_dir_path, self.model_dir_name)

        self.model_ext = ".pt"
        self.record_ext = ".pkl"
        self.best_model_file_name = f"{self.model_name}_best_model{self.model_ext}"
        self.entire_record_file_name = f"{self.model_name}_entire_record{self.record_ext}"

    def get_record_file_path_list(self):
        record_file_path_list = list()

        if os.path.isdir(self.model_dir_path):
            for file_name in os.listdir(self.model_dir_path):
                file_basename, file_ext = os.path.splitext(file_name)

                if file_ext == self.record_ext and file_basename.split("_")[-1].isdigit():
                    record_file_path_list.append(os.path.join(self.model_dir_path, file_name))

        return record_file_path_list

    def get_model_file_path_list(self):
        model_file_path_list = list()

        if os.path.isdir(self.model_dir_path):
            for file_name in os.listdir(self.model_dir_path):
                file_basename, file_ext = os.path.splitext(file_name)

                if file_ext == self.model_ext and file_basename.split("_")[-1].isdigit():
                    model_file_path_list.append(os.path.join(self.model_dir_path, file_name))

        return model_file_path_list

    def create_model_file_name(self, epoch):
        return f"{self.model_name}_epoch_{epoch}{self.model_ext}"

    def create_record_file_name(self, epoch):
        return f"{self.model_name}_record_epoch_{epoch}{self.record_ext}"

    def get_save_model_file_path(self, epoch):
        return os.path.join(self.model_dir_path, self.create_model_file_name(epoch=epoch))

    def get_save_record_file_path(self, epoch):
        return os.path.join(self.model_dir_path, self.create_record_file_name(epoch=epoch))

    def get_best_model_file_path(self):
        return os.path.join(self.model_dir_path, self.best_model_file_name)

    def get_entire_record_file_path(self):
        return os.path.join(self.model_dir_path, self.entire_record_file_name)
