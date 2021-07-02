import argparse

from config.config import Config
from dataset.factory import DatasetModule
from domain.metadata import Metadata
from logger import logger
from model.factory import ModelModule
from trainer.factory import TrainerModule


def main(args):
    mode = args.mode.lower()
    config_file_name = args.config.lower()

    # Get Parameters
    params = Config(file_name=config_file_name).params
    logger.info(f"Parameter information :\n{params}")

    metadata_params = params.metadata
    dataset_params = params.dataset
    model_params = params.model
    trainer_params = params.trainer

    # Metadata Controller
    metadata = Metadata(**metadata_params)

    # Dataset Controller
    dataset_module = DatasetModule(metadata=metadata, **dataset_params)

    # Model Controller
    model_module = ModelModule(metadata=metadata, **model_params)

    # Trainer Controller
    trainer_module = TrainerModule(
        metadata=metadata,
        model_module=model_module,
        dataset_module=dataset_module,
        **trainer_params
    )

    result_dict = trainer_module.do(mode=mode)

    print(result_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pytorch Project Template [Byeonggil Jung (Korea Univ, AIR Lab)]")
    parser.add_argument("--mode", required=False, default="train", help="Select the mode, train | inference")
    parser.add_argument("--config", required=True, help="Select the config file")
    args = parser.parse_args()

    logger.info(f"Selected parameters : {args}")

    main(args=args)
