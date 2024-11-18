from parser import TrainingConfig, ModelConfig, Config
import torch
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


training_config = TrainingConfig()
model_config = ModelConfig()
model_config.model_name = "lstm_autoencoder"
data_path = "/home/ubuntu/ml-convection/dataset"

if __name__ == "__main__":
    fluid_dataset = FluidDataset(data_path, training_config)
    train_dataset = TemporalDataset(training_config, fluid_dataset, "train")
    val_dataset = TemporalDataset(training_config, fluid_dataset, "val")
    x, y = train_dataset[0]
    logging.info(f"Train Dataset Length: {len(train_dataset)}")
    logging.info(f"Val Dataset Length: {len(val_dataset)}")

    logging.info(f"Creating Model with values: {fluid_dataset.min_value}, {fluid_dataset.max_value}")