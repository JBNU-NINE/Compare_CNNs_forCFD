from parser import TrainingConfig, ModelConfig, Config, save_all_config
import torch
# torch random seed 
torch.manual_seed(0)
from models import LitLSTMUNetSmall
from temporal_utils.dataloader import TemporalDataset, FluidDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from datetime import datetime

from models import LitLSTMUNetSmall
import logging 
logging.basicConfig(level=logging.INFO)
log_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


training_config = TrainingConfig()
model_config = ModelConfig()
model_config.model_name = "lstm_autoencoder"
data_path = "/home/ubuntu/ml-convection/dataset"
def get_ckpt_path(model_name, log_version):
    ckpt_dir = os.path.join("logs", model_name, log_version, "checkpoints")
    filename = os.listdir(ckpt_dir)[0]
    if filename.endswith(".ckpt"):
        return os.path.join(ckpt_dir, filename)
    raise Exception("No ckpt file found")

def create_callbacks():
    log_dir = training_config.log_dir
    callbacks = []
    if not os.path.exists(log_dir):
        logging.critical(f"Log Directory {log_dir} does not exist")
        os.makedirs(log_dir)
    logging.info(f"Find Logs at {log_dir}")
    logger = pl.loggers.TensorBoardLogger(
            "logs/", name=model_config.model_name, version=log_version
        )
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            monitor = training_config.model_checkpoint.monitor,
            mode = training_config.model_checkpoint.mode,
        )
    )
    callbacks.append(
        pl.callbacks.EarlyStopping(
            monitor = training_config.early_stopping.monitor,
            patience = training_config.early_stopping.patience,
            mode = training_config.early_stopping.mode
        )
    )
    # saving config 
    logging.info(f"Saving Config at {log_dir}")
    return callbacks, logger
    



if __name__ == "__main__":
    fluid_dataset = FluidDataset(data_path, training_config)
    train_dataset = TemporalDataset(training_config, fluid_dataset, "train")
    val_dataset = TemporalDataset(training_config, fluid_dataset, "val")
    logging.info(f"Train Dataset Length: {len(train_dataset)}")
    logging.info(f"Val Dataset Length: {len(val_dataset)}")

    logging.info(f"Creating Model with values: {fluid_dataset.min_value}, {fluid_dataset.max_value}")
    model = LitLSTMUNetSmall(fluid_dataset.min_value, fluid_dataset.max_value)
    callbacks_arr, logger = create_callbacks()
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers = 5)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers = 5)


    trainer = pl.Trainer(
        accelerator = training_config.device,
        devices = training_config.devices,
        max_epochs = training_config.epochs,
        callbacks = callbacks_arr,
        logger = logger,
        log_every_n_steps=5,
    )
    logging.info("Training model")
    logging.info(f"Save all config")
    config_path = os.path.join(training_config.log_dir, model_config.model_name, log_version)
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    save_all_config(os.path.join(config_path, "config.json"))
    trainer.fit(model, train_loader, val_loader)