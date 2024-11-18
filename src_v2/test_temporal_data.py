import pytest
from parser import TrainingConfig, ModelConfig, Config
import torch
from datetime import datetime
import os
from temporal_utils.dataloader import FluidDataset, TemporalDataset
import numpy as np

training_config = TrainingConfig()
model_config = ModelConfig()
data_path = "/home/sangam/workspace/sangam/python_practice/ml-convection/dataset"
data_shape = (10000, 3, 200, 200)


def test_data_path():
    assert os.path.exists(data_path), "Data Path does not exist"


def test_fluid_dataset():
    fluid_dataset = FluidDataset(data_path, training_config)
    assert (
        fluid_dataset.dataset_arr.numpy().shape == data_shape
    ), "Some mistake in fluid Dataset Shape"
    print(fluid_dataset.dataset_arr.shape)

    max_value = np.max(fluid_dataset.dataset_arr.numpy())
    min_value = np.min(fluid_dataset.dataset_arr.numpy())
    assert max_value == 1.0, "Max Value is not 1.0 (Not normalized)"
    assert min_value == 0.0, "Min Value is not 0.0 (Not Normalized)"


def test_temporal_dataset():
    data_type = "train"
    fluid_dataset = FluidDataset(data_path, training_config)
    temporal_dataset = TemporalDataset(training_config, fluid_dataset, data_type)
    if data_type == "train":
        assert (
            temporal_dataset.dataset_range == training_config.train_data_range
        ), "Train Data Range is not correct"


def test_dataloader():
    data_type = "train"
    fluid_dataset = FluidDataset(data_path, training_config)
    temporal_dataset = TemporalDataset(training_config, fluid_dataset, data_type)
    dataloader = torch.utils.data.DataLoader(
        temporal_dataset,
        batch_size=1,
    )
    for x, y in dataloader:
        for x_curr in x[0]:
            assert not torch.all(x_curr == y[0]), "X and Y are same"
        break
