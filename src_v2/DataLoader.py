from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from parser import TrainingConfig


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CustomDataLoader:
    min_value = []
    max_value = []
    mean_value = []

    def __init__(self, dataset_path, training_config):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        self.dataset_path = dataset_path
        self.train_data_range = training_config.train_data_range
        self.val_data_range = training_config.val_data_range
        self.test_data_range = training_config.test_data_range
        self.data_names = training_config.dataset_names
        self.batch_size = training_config.batch_size
        self.y_type = training_config.y_type

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

    def get_data_range(self, data_range):
        x_arr = self.dataset_arr[data_range[0] : data_range[1]]
        y_arr = self.dataset_arr[data_range[0] + 1 : data_range[1] + 1]
        if self.y_type == "diff":
            y_arr = y_arr - x_arr
        return x_arr, y_arr

    def get_data(self, data_type):
        if data_type not in ["train", "val", "test"]:
            raise ValueError(f"Invalid data type {data_type}")

        if data_type == "train":
            data_range = self.train_data_range
        elif data_type == "val":
            data_range = self.val_data_range
        else:
            data_range = self.test_data_range
        dataset = Dataset(*self.get_data_range(data_range))
        if data_type == "test":
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
        else:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return loader


if __name__ == "__main__":
    data_path = "/home/ubuntu/ml-convection/dataset"
    default_config = TrainingConfig()
    dataloader = CustomDataLoader(
        dataset_path=data_path, training_config=default_config
    )
    dataloader.set_required_data()
    loader = dataloader.get_data("train")
    for x, y in loader:
        print(x.shape, y.shape)
        print(x.dtype, y.dtype)
        break
