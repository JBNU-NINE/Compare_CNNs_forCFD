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
from plot_utils import PlotUtils


training_config = TrainingConfig()
model_config = ModelConfig()
data_path = "/home/ubuntu/ml-convection/dataset"

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    dirpath="media/",
    filename="lstm_checkpoint",
    save_top_k=1,
    mode="min",
)
early_stopping_callback = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, mode="min"
)


def unnormalize_y(unnorm_arr, min_value, max_value):
    unnorm_arr[:, 0, :, :] = (
        unnorm_arr[:, 0, :, :] * (max_value[0] - min_value[0]) + min_value[0]
    )
    unnorm_arr[:, 1, :, :] = (
        unnorm_arr[:, 1, :, :] * (max_value[1] - min_value[1]) + min_value[1]
    )
    unnorm_arr[:, 2, :, :] = (
        unnorm_arr[:, 2, :, :] * (max_value[2] - min_value[2]) + min_value[2]
    )
    return unnorm_arr


def get_predictions(model, dataloader):
    model.eval()
    predictions_arr = []
    test_y_arr = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(model.device)
            y_hat = model.model(x)
            y_hat = y_hat.detach().cpu().numpy()
            predictions_arr.extend(y_hat)
            test_y_arr.extend(y.detach().cpu().numpy())
    return predictions_arr, test_y_arr


def get_max_ae_arr(predicted_y, test_y):
    max_ae_arr = []
    for i in range(predicted_y.shape[0]):
        max_ae_arr.append(torch.max(torch.abs(predicted_y[i] - test_y[i])).item())
    return max_ae_arr


def plot_max_ae(predicted_y, test_y, channel_name):
    mae_arr = get_max_ae_arr(predicted_y, test_y)
    plt.figure()
    plt.title(channel_name)
    plt.plot(mae_arr)
    # TODO: This is hack, get proper saving path on future
    save_path = "media/plots/"
    plt.savefig(os.path.join(save_path, f"{channel_name}_max_ae.png"))


def calculate_errors(predicted_y, test_y):
    mae_dict = defaultdict(lambda: 0)
    max_ae_dict = defaultdict(lambda: 0)
    dataset_channel_names = training_config.dataset_names
    dataset_channel_names = [name.split(".")[0] for name in dataset_channel_names]
    # Split predicted_y and test_y_arr into individual channels
    predicted_y_arr = torch.split(predicted_y, 1, dim=1)
    test_y_arr = torch.split(test_y, 1, dim=1)
    for i in range(len(dataset_channel_names)):
        mae = F.l1_loss(predicted_y_arr[i], test_y_arr[i])
        max_ae = torch.max(torch.abs(predicted_y_arr[i] - test_y_arr[i]))
        curr_mae = float(mae.detach().cpu().numpy())
        if curr_mae > mae_dict[dataset_channel_names[i]]:
            mae_dict[dataset_channel_names[i]] = curr_mae
        curr_max_ae = float(max_ae.detach().cpu().numpy())
        if curr_max_ae > max_ae_dict[dataset_channel_names[i]]:
            max_ae_dict[dataset_channel_names[i]] = curr_max_ae

        # Plotting Accuracy:
        plot_utils.plot_accuracy(predicted_y_arr[i], test_y_arr[i],f"sequential_{dataset_channel_names[i]}")

        plot_utils.save_animation(
            np.squeeze(predicted_y_arr[i].detach().cpu().numpy()),
            f"sequential_{dataset_channel_names[i]}",
        )

    return {"MAE": mae_dict, "Max AE": max_ae_dict}

log_version = "/home/ubuntu/ml-convection/src_v2/logs/lstm_autoencoder/2024-03-14_22-15-15"
plot_utils = PlotUtils(log_version)

if __name__ == "__main__":
    fluid_dataset = FluidDataset(data_path, training_config)
    temporal_dataset = TemporalDataset(training_config, fluid_dataset, "test")
    print(f" Total Dataset Length {len(temporal_dataset)}")
    dataloader = DataLoader(
        temporal_dataset,
        batch_size=training_config.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    model = LitLSTMUNetSmall.load_from_checkpoint(
        f"{log_version}/checkpoints/epoch=2399-step=120000.ckpt",
        min_value=fluid_dataset.min_value,
        max_value=fluid_dataset.max_value,
    )
    predictions_arr, test_y_arr = get_predictions(model, dataloader)
    predictions_arr = torch.tensor(np.array(predictions_arr))
    test_y_arr = torch.tensor(np.array(test_y_arr))
    print(f"Loader Values: {fluid_dataset.min_value, fluid_dataset.max_value}")
    print(f"Predictions Shape: {predictions_arr.shape}")
    print(f"Before Unnormalization")
    print(torch.max(predictions_arr), torch.min(predictions_arr))
    print(torch.max(test_y_arr), torch.min(test_y_arr))
    predictions_arr = unnormalize_y(
        predictions_arr, fluid_dataset.min_value, fluid_dataset.max_value
    )
    test_y_arr = unnormalize_y(
        test_y_arr, fluid_dataset.min_value, fluid_dataset.max_value
    )
    print(f"After Unnormalization")
    print(torch.max(predictions_arr), torch.min(predictions_arr))
    print(torch.max(test_y_arr), torch.min(test_y_arr))
    my_dict = calculate_errors(predictions_arr, test_y_arr)
    print(my_dict)
