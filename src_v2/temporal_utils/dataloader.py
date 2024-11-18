from torch.utils.data import DataLoader
import os
import numpy as np
import torch


class FluidDataset:
    min_value = []
    max_value = []
    mean_value = []

    def __init__(self, dataset_path, training_config):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        self.dataset_path = dataset_path
        self.data_names = training_config.dataset_names
        self.set_required_data()

    def set_required_data(self):
        data_names = self.data_names
        data_arr = []
        for data_file in data_names:
            data_path = os.path.join(self.dataset_path, data_file)
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Data file {data_file} not found at {data_path}"
                )
            data_np = self.normalize_data(np.load(data_path).astype("float32"))
            data_arr.append(data_np)

        dataset_arr = np.stack([d for d in data_arr], axis=1)
        self.dataset_arr = torch.from_numpy(dataset_arr)

    def normalize_data(self, x):
        min_value = np.min(x)
        max_value = np.max(x)
        # Save the min and max values for later use
        self.min_value.append(min_value)
        self.max_value.append(max_value)
        self.mean_value.append(np.mean(x))
        normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return normalized_x


class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, training_config, fluid_dataset, data_mode="test"):
        assert data_mode in ["train", "val", "test"], "Invalid Data Mode Given"
        self.data_mode = data_mode
        self.train_data_range = training_config.train_data_range
        self.val_data_range = training_config.val_data_range
        self.test_data_range = training_config.test_data_range
        self.dataset_range = self.get_data_range()
        self.fluid_dataset = fluid_dataset.dataset_arr[
            self.dataset_range[0] : self.dataset_range[1]
        ]
        self.data_names = training_config.dataset_names
        self.batch_size = training_config.batch_size
        self.y_type = training_config.y_type
        self.sequential_size = 5

    def __len__(self):
        data_range = self.get_data_range()
        data_length = data_range[1] - data_range[0]
        return data_length - self.sequential_size

    def get_data_range(self):
        if self.data_mode == "train":
            return self.train_data_range
        elif self.data_mode == "val":
            return self.val_data_range
        elif self.data_mode == "test":
            return self.test_data_range

    def __getitem__(self, idx):
        """
        Note, this is not for difference calculation.
        It just returns the sequential data and the next data.
        """
        x = self.fluid_dataset[idx : idx + self.sequential_size]
        y = self.fluid_dataset[idx + self.sequential_size]
        if self.y_type == "diff":
            y = y - x[-1]
        return x, y

    def __repr__(self):
        return f"TemporalDataLoader({self.fluid_dataset.dataset_path}, {self.train_data_range}, {self.val_data_range}, {self.test_data_range}, {self.data_names}, {self.batch_size}, {self.y_type})"
