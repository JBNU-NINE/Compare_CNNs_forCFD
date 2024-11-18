from dataclasses import dataclass
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import dataclasses
import inspect
from typing import List
import torch
import json


@dataclass
class Config:
    pass


@dataclass
class EarlyStoppingConfig(Config):
    callback_type: str = "EarlyStopping"
    status: bool = True
    patience: int = 1000
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class ModelCheckpointConfig(Config):
    callback_type: str = "ModelCheckpoint"
    status: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
    dirpath: str = None
    filename: str = "{epoch}-{val_loss:.2f}"


@dataclass
class ModelConfig(Config):
    """
    DEVICE_ORDER only for Sangam's Machine:
    0-4: 2080TI
    5: A100 80GB
    6: 2080TI
    7: A6000
    """

    model_name: str = "lstm_autoencoder"
    test_device = "cuda:0"


@dataclass
class TrainingConfig(Config):
    batch_size: int = 16
    epochs: int = 5000
    learning_rate: float = 0.0001
    train_data_range: tuple = (0, 800)
    val_data_range: tuple = (800, 900)
    test_data_range: tuple = (900, 1000)
    device: str = "gpu"
    devices: tuple = tuple([0])
    optimizer: str = "adam"
    loss: str = "mse"
    dataset_names: tuple = ("t_arr.npy", "ux_arr.npy", "uy_arr.npy")
    log_dir: str = "logs/"
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    model_checkpoint: ModelCheckpointConfig = ModelCheckpointConfig(
        dirpath=os.path.join(log_dir + f"{ModelConfig.model_name}", "checkpoints")
    )
    generate_samples: bool = True
    regressive_padding: int = 1  # Equivalent to 1 padding
    y_type: str = "diff"

def without_keys(d, keys):
    return {k:v for k, v in d.items() if k not in keys}
def save_all_config(path):
    with open(path, "w") as f:
        json.dump(
            {
                "TrainingConfig": without_keys(TrainingConfig().__dict__, ["early_stopping", "model_checkpoint"]),
                "ModelConfig": ModelConfig().__dict__,
            },
            f, 
            indent = 4
        )
if __name__ == "__main__":
    default_config = TrainingConfig()
    default_model_config = ModelConfig()
    print(default_config)
    print(default_config.early_stopping)
    print(default_config.model_checkpoint)
    print(default_config.model_checkpoint.monitor)
