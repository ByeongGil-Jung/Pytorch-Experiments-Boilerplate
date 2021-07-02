<div align="center">    
 
# Your Project Name     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
Pytorch project template for experiments and application (Description ...)    

## How to run   
First, install dependencies   
```bash
# Clone project   
git clone https://github.com/ByeongGil-Jung/Pytorch-Project-Template.git

# Install project   
cd Pytorch-Project-Template
pip install -e .   
pip install -r requirements.txt
 ```    
If you want to modify configurations, modify below YAML files. 
 ```bash
# Modify below configuration YAML files        
cd config
```    
Next, navigate to any file and run it.   
 ```bash
# Run module (example: mnist as your main contribution)   
python main.py --mode "train" --config "kaggle_casting_data-cnn_custom-0.yaml"    
```

## Example code
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
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
```

### Citation   
```
@article{Byeonggil Jung,
  title={Title},
  author={Team},
  journal={Location},
  year={Year}
}
```   
