from parser import TrainingConfig, ModelConfig, Config
import torch
from DataLoader import CustomDataLoader
from models import LitLSTMUNetSmall
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from plot_utils import PlotUtils
from temporal_utils.dataloader import TemporalDataset, FluidDataset
from torch.utils.data import DataLoader
import tqdm


training_config = TrainingConfig()
model_config = ModelConfig()
data_path = "/home/ubuntu/ml-convection/dataset"


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

def get_regressive_predictions(model, dataset):
    model.eval()
    n_samples = len(dataset)
    print(f"Total Samples: {n_samples}")
    x = torch.tensor(dataset[0][0])
    x = x.unsqueeze(0)
    test_y_arr = np.zeros((n_samples, 3, 200, 200))
    predictions_arr = np.zeros((n_samples, 3, 200, 200))
    with torch.no_grad():
        for i in tqdm.trange(n_samples):
            x = x.to(model.device)
            y_hat = model.model(x)
            y_true = dataset[i][0][-1].detach().cpu() + dataset[i][1]
            y_pred = x[0][-1] + y_hat[0]
            predictions_arr[i] = y_pred.detach().cpu().numpy()
            test_y_arr[i] = y_true.detach().cpu().numpy()
            x_list = x.squeeze(0).tolist()
            x_list.append(torch.tensor(y_pred).squeeze(0))
            x_list.pop(0)
            x_list = torch.tensor(x_list)
            x = x_list.unsqueeze(0)
    predictions_arr = torch.tensor(predictions_arr)
    test_y_arr = torch.tensor(test_y_arr)
    print(f"Predictions Shape: {predictions_arr.shape}")
    print(f"Test Y Shape: {test_y_arr.shape}")
    return predictions_arr, test_y_arr

def get_regressive_padding_predictions(model, dataset, pad_size = 1):
    model.eval()
    n_samples = len(dataset)
    print(f"Total Samples: {n_samples}")
    with torch.no_grad():
        for i in range(n_samples):
            x = x.to(model.device)
            y_hat = model.model(x)
            break



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
        plot_utils.plot_accuracy(predicted_y_arr[i], test_y_arr[i], f"regressive_{dataset_channel_names[i]}")
        # plot_utils.save_animation(
        #     np.squeeze(np.abs(predicted_y_arr[i].detach().cpu().numpy()) - np.abs(test_y_arr[i].detach().cpu().numpy())) ,
        #     f"regressive_contour_{dataset_channel_names[i]}",
        # )
        print("Shapes: ", predicted_y_arr[i].shape, test_y_arr[i].shape)
        # plot_utils.save_animation(
        #     np.squeeze(predicted_y_arr[i].detach().cpu().numpy()),
        #     f"regressive_{dataset_channel_names[i]}",
        # )
        # plot_utils.save_animation(
        #     np.squeeze(np.abs(predicted_y_arr[i].detach().cpu().numpy()) - np.abs(test_y_arr[i].detach().cpu().numpy())) ,
        #     f"regressive_contour_{dataset_channel_names[i]}",
        # )
    plot_utils.calculate_residual()
    return {"MAE": mae_dict, "Max AE": max_ae_dict}

log_version = "/home/ubuntu/ml-convection/src_v2/logs/lstm_autoencoder/diff_paper"
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
        "/home/ubuntu/ml-convection/src_v2/logs/lstm_autoencoder/diff_paper/checkpoints/epoch=4789-step=239500.ckpt",
        min_value=fluid_dataset.min_value,
        max_value=fluid_dataset.max_value,
    )
    predictions_arr, test_y_arr = get_regressive_predictions(model, temporal_dataset)
    # predictions_arr, test_y_arr = get_predictions(model, dataloader)
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
