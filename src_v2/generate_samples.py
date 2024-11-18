import torch
from models import LitAutoEncoder
import parser
from ModelConfig import ModelSelector
from DataLoader import CustomDataLoader
import torch.nn.functional as F
from plot_utils import PlotUtils
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class GenerateSamples:
    def __init__(self, training_config, model_config, data_path, log_version):
        self.data_path = data_path
        self.training_config = training_config
        self.model_config = model_config
        self.log_version = log_version
        self.__load_dataset()

    def __load_dataset(self):
        self.loader = CustomDataLoader(self.data_path, self.training_config)
        self.loader.set_required_data()
        if self.model_config.model_name == "unet":
            ## TODO: This is a hack, fix this but
            ## For now, we will have a padding of 56 pixels on each side
            print(
                "Applying Hack, padding by 28 pixels every side \n Please remove it later"
            )

            # print(f"The mean value of the dataset is: {self.loader.mean_value}")
            self.loader.dataset_arr = F.pad(
                self.loader.dataset_arr,
                (28, 28, 28, 28),
                mode="constant",
                value=0,
            )

    def set_model(self, checkpoint_path):
        self.model = (
            ModelSelector(self.model_config)
            .get_model(self.model_config.model_name)
            .load_from_checkpoint(
                checkpoint_path,
                min_value=self.loader.min_value,
                max_value=self.loader.max_value,
                map_location=self.model_config.test_device,
            )
        )

    def calculate_errors(self, predicted_y, test_y, create_gif):
        if create_gif:
            plot_utils = PlotUtils(self.log_version)
        mae_dict = defaultdict(lambda: 0)
        max_ae_dict = defaultdict(lambda: 0)
        dataset_channel_names = self.training_config.dataset_names
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
            plot_utils.plot_accuracy(
                predicted_y_arr[i],
                test_y_arr[i],
                f"{dataset_channel_names[i]}",
            )
            if create_gif:
                plot_utils.save_animation(
                    np.squeeze(predicted_y_arr[i].detach().cpu().numpy()),
                    f"{dataset_channel_names[i]}",
                )
            # if create_gif and generation_type == "sequential":
            #     plot_utils.save_animation(
            #         np.squeeze(test_y_arr[i].detach().cpu().numpy()),
            #         f"{dataset_channel_names[i]}_true",
            #     )
        plot_utils.calculate_residual()
        return {"MAE": mae_dict, "Max AE": max_ae_dict}

    @staticmethod
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

    @torch.no_grad()
    def __sequential_generation(
        self, starting_index, n_samples, create_gif, generation_type
    ):
        # TODO: this function can infer the generation type from the function names itself
        test_x_arr, test_y_arr = self.loader.get_data_range(
            (starting_index, starting_index + n_samples)
        )
        autoencoder = self.model.model
        test_x_arr = torch.tensor(test_x_arr).to(self.model.device)
        test_y_arr = torch.tensor(test_y_arr).to(self.model.device)
        predicted_y = autoencoder(test_x_arr)
        predicted_y_unnorm = predicted_y.detach().clone()
        predicted_y_unnorm = GenerateSamples.unnormalize_y(
            predicted_y_unnorm, self.loader.min_value, self.loader.max_value
        )

        test_y_arr_unnorm = test_y_arr.detach().clone()
        test_y_arr_unnorm = GenerateSamples.unnormalize_y(
            test_y_arr_unnorm, self.loader.min_value, self.loader.max_value
        )
        errors_dict_unnorm = self.calculate_errors(
            predicted_y_unnorm,
            test_y_arr_unnorm,
            create_gif,
        )
        print("Unnormalized Error: ", errors_dict_unnorm)
        with open(
            os.path.join(
                "logs",
                self.model_config.model_name,
                self.log_version,
                "training_config.txt",
            ),
            "a",
        ) as f:
            f.write("Sequential Generation\n")
            f.write(f"Unnormalized Error: {errors_dict_unnorm}\n")

    @torch.no_grad()
    def __regressive_generation_padding(
        self, starting_index, n_samples, create_gif, generation_type, padding
    ):
        """
        This function will also do a regressive generation, but there are a few differences and once
        this function is complete, will be merged with the __regressive_generation function.
        The differences are:
        1. This function Will take an input of (200x200) pixels; say X
        2. This function Will output (200x200) pixels; say Y
        3. For X_1 which should have been Y_0, we will:
            - Remove padding from X_1/Y_0, let's say for now 2 pixels.
            - Now, If we remove 2 pixels from either side, the image should be 196x196.
            - Now, we will pad the image with 2 pixels from original image or Y_0_true.
        4. So, if padding is 0, then this function should be identical to __regressive_generation
        """
        test_x_arr, test_y_arr = self.loader.get_data_range(
            (starting_index, starting_index + n_samples)
        )
        autoencoder = self.model.model
        predicted_samples = torch.zeros_like(test_y_arr).to(self.model.device)
        print(predicted_samples.shape)
        test_x_arr = torch.tensor(test_x_arr).to(self.model.device)
        test_y_arr = torch.tensor(test_y_arr).to(self.model.device)

        curr_sample = test_x_arr[0]
        for i in range(n_samples):
            curr_sample = curr_sample.unsqueeze(0)
            predicted_y = autoencoder(curr_sample)
            predicted_y = predicted_y.squeeze(0)
            # All these things to do because apparently torch doesn't allow in-place operations
            temp_sample = test_y_arr[i].detach().cpu().numpy()
            temp_sample[:, padding:-padding, padding:-padding] = (
                predicted_y.detach()
                .cpu()
                .numpy()[:, padding:-padding, padding:-padding]
            )

            curr_sample = torch.tensor(temp_sample).to(self.model.device)
            predicted_samples[i] = curr_sample

        predicted_samples_unnorm = predicted_samples.detach().clone()
        predicted_samples_unnorm = GenerateSamples.unnormalize_y(
            predicted_samples_unnorm, self.loader.min_value, self.loader.max_value
        )
        test_y_arr_unnorm = test_y_arr.detach().clone()
        test_y_arr_unnorm = GenerateSamples.unnormalize_y(
            test_y_arr_unnorm, self.loader.min_value, self.loader.max_value
        )
        errors_dict_unnorm = self.calculate_errors(
            predicted_samples_unnorm,
            test_y_arr_unnorm,
            create_gif,
        )
        print("Unnormalized Error: ", errors_dict_unnorm)
        with open(
            os.path.join(
                "logs",
                self.model_config.model_name,
                self.log_version,
                "training_config.txt",
            ),
            "a",
        ) as f:
            f.write(f"Regressive Generation with Padding: {padding}\n")
            f.write(f"Unnormalized Error: {errors_dict_unnorm}\n")

    @torch.no_grad()
    def __regressive_generation(
        self, starting_index, n_samples, create_gif, generation_type
    ):
        # TODO: this function can infer the generation type from the function names itself
        test_x_arr, test_y_arr = self.loader.get_data_range(
            (starting_index, starting_index + n_samples)
        )
        autoencoder = self.model.model
        predicted_samples = torch.zeros_like(test_y_arr).to(self.model.device)
        print(predicted_samples.shape)
        test_x_arr = torch.tensor(test_x_arr).to(self.model.device)
        test_y_arr = torch.tensor(test_y_arr).to(self.model.device)

        curr_sample = test_x_arr[0]
        for i in range(n_samples):
            curr_sample = curr_sample.unsqueeze(0)
            predicted_y = autoencoder(curr_sample)
            predicted_y = predicted_y.squeeze(0)
            predicted_samples[i] = predicted_y
            curr_sample = curr_sample.squeeze(0)
            curr_sample = predicted_y + curr_sample

        predicted_samples_unnorm = predicted_samples.detach().clone()
        predicted_samples_unnorm = GenerateSamples.unnormalize_y(
            predicted_samples_unnorm, self.loader.min_value, self.loader.max_value
        )
        test_y_arr_unnorm = test_y_arr.detach().clone()
        test_y_arr_unnorm = GenerateSamples.unnormalize_y(
            test_y_arr_unnorm, self.loader.min_value, self.loader.max_value
        )
        errors_dict_unnorm = self.calculate_errors(
            predicted_samples_unnorm,
            test_y_arr_unnorm,
            create_gif,
        )
        print("Unnormalized Error: ", errors_dict_unnorm)
        with open(
            os.path.join(
                "logs",
                self.model_config.model_name,
                self.log_version,
                "training_config.txt",
            ),
            "a",
        ) as f:
            f.write("Regressive Generation\n")
            f.write(f"Unnormalized Error: {errors_dict_unnorm}\n")

    def generate_samples(
        self,
        generation_type,
        starting_index,
        n_samples,
        create_gif=False,
        padding=1,
    ):
        allowed_generation_types = ["sequential", "regressive", "regressive_padding"]
        if generation_type not in allowed_generation_types:
            raise ValueError(
                f"generation_type should be one of {allowed_generation_types}"
            )
        if generation_type == "sequential":
            self.__sequential_generation(
                starting_index, n_samples, create_gif, generation_type
            )
        elif generation_type == "regressive":
            self.__regressive_generation(
                starting_index, n_samples, create_gif, generation_type
            )
        elif generation_type == "regressive_padding":
            self.__regressive_generation_padding(
                starting_index, n_samples, create_gif, generation_type, padding=padding
            )


def get_ckpt_path(model_name, log_version):
    ckpt_dir = os.path.join("logs", model_name, log_version, "checkpoints")
    filename = os.listdir(ckpt_dir)[0]
    if filename.endswith(".ckpt"):
        return os.path.join(ckpt_dir, filename)
    raise Exception("No ckpt file found")


if __name__ == "__main__":
    log_version = "version_0"
    model_path = get_ckpt_path("unet_small", log_version)
    sample_generator = GenerateSamples(
        parser.TrainingConfig(),
        parser.ModelConfig(),
        "/home/ubuntu/ml-convection/dataset",
        log_version,
    )
    sample_generator.set_model(model_path)
    # sample_generator.generate_samples("sequential", 0, 100, True)
    # sample_generator.generate_samples("regressive", 0, 100, True)
    sample_generator.generate_samples("sequential", 900, 100, True)
    sample_generator.generate_samples("regressive", 900, 100, True)
    sample_generator.generate_samples("regressive_padding", 900, 100, True, padding=1)
