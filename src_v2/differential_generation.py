from generate_samples import GenerateSamples, get_ckpt_path
import parser
import torch

log_version = "2024-01-04_00-30-50"
model_type = parser.ModelConfig().model_name
model_path = get_ckpt_path(model_type, log_version)
dataset_path = "/home/sangam/workspace/sangam/python_practice/ml-convection/dataset"


starting_index = parser.TrainingConfig().test_data_range[0]
n_samples = parser.TrainingConfig().test_data_range[1] - starting_index
crete_gif = True
generation_type = "sequential"


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


def regressive_padding_generation(
    sample_generator, test_x_arr, test_y_arr, autoencoder, pad_size=1
):
    test_x_arr = torch.tensor(test_x_arr).to(sample_generator.model.device)
    test_y_arr = torch.tensor(test_y_arr).to(sample_generator.model.device)
    curr_sample = test_x_arr[0]
    predicted_sample_arr = torch.zeros_like(test_y_arr).to(
        sample_generator.model.device
    )
    autoencoder.eval()
    with torch.no_grad():
        for i in range(n_samples):
            curr_sample = curr_sample.unsqueeze(0)
            predicted_y = autoencoder(curr_sample)
            predicted_y = predicted_y.squeeze(0)
            curr_sample = curr_sample.squeeze(0)
            temp_sample = curr_sample + test_y_arr[i]
            curr_sample = curr_sample + predicted_y
            temp_sample[:, pad_size:-pad_size, pad_size:-pad_size] = curr_sample[
                :, pad_size:-pad_size, pad_size:-pad_size
            ]
            print(temp_sample[:, pad_size:-pad_size, pad_size:-pad_size].shape)
            predicted_sample_arr[i] = temp_sample
            curr_sample = temp_sample
        return predicted_sample_arr


def regressive_generation(sample_generator, test_x_arr, test_y_arr, autoencoder):
    test_x_arr = torch.tensor(test_x_arr).to(sample_generator.model.device)
    test_y_arr = torch.tensor(test_y_arr).to(sample_generator.model.device)
    curr_sample = test_x_arr[0]
    predicted_sample_arr = torch.zeros_like(test_y_arr).to(
        sample_generator.model.device
    )
    autoencoder.eval()
    with torch.no_grad():
        for i in range(n_samples):
            curr_sample = curr_sample.unsqueeze(0)
            predicted_y = autoencoder(curr_sample)
            predicted_y = predicted_y.squeeze(0)
            curr_sample = curr_sample.squeeze(0)
            curr_sample = curr_sample + predicted_y
            predicted_sample_arr[i] = curr_sample
        return predicted_sample_arr


def sequential_generation(sample_generator, test_x_arr, test_y_arr, autoencoder):
    test_x_arr = torch.tensor(test_x_arr).to(sample_generator.model.device)
    test_y_arr = torch.tensor(test_y_arr).to(sample_generator.model.device)
    predicted_samples_arr = torch.zeros_like(test_y_arr).to(
        sample_generator.model.device
    )
    autoencoder.eval()
    with torch.no_grad():
        for i in range(n_samples):
            curr_sample = test_x_arr[i]
            curr_sample = curr_sample.unsqueeze(0)
            predicted_y = autoencoder(curr_sample)
            predicted_samples_arr[i] = predicted_y.squeeze(0)
        return predicted_samples_arr


def generate_samples(
    sample_generator, test_x_arr, test_y_arr, autoencoder, generation_type
):
    assert generation_type in ["sequential", "regressive", "regressive_padding"]
    if generation_type == "sequential":
        predicted_samples_arr = sequential_generation(
            sample_generator, test_x_arr, test_y_arr, autoencoder
        )
    elif generation_type == "regressive":
        predicted_samples_arr = regressive_generation(
            sample_generator, test_x_arr, test_y_arr, autoencoder
        )
    else:
        padding = 1
        if model_type == "unet":
            padding = 29  # Because Unet is larger (256,256)
        predicted_samples_arr = regressive_padding_generation(
            sample_generator, test_x_arr, test_y_arr, autoencoder, padding
        )

    return predicted_samples_arr


if __name__ == "__main__":
    sample_generator = GenerateSamples(
        parser.TrainingConfig(),
        parser.ModelConfig(),
        dataset_path,
        log_version,
    )
    sample_generator.set_model(model_path)
    test_x_arr, test_y_arr = sample_generator.loader.get_data_range(
        (starting_index, starting_index + n_samples)
    )
    print(test_x_arr.shape, test_y_arr.shape)
    autoencoder = sample_generator.model.model
    predicted_samples_arr = generate_samples(
        sample_generator, test_x_arr, test_y_arr, autoencoder, generation_type
    )
    predicted_samples_arr = unnormalize_y(
        predicted_samples_arr,
        sample_generator.loader.min_value,
        sample_generator.loader.max_value,
    )
    # We need to this because:
    # In sequential, we can calculate errors according to difference
    # but in others, we need to get the actual values.
    if generation_type in ["regressive", "regressive_padding"]:
        test_y_arr, _ = sample_generator.loader.get_data_range(
            (starting_index + 1, starting_index + n_samples + 1)
        )
    test_y_arr = unnormalize_y(
        test_y_arr, sample_generator.loader.min_value, sample_generator.loader.max_value
    )

    prediction_dict = sample_generator.calculate_errors(
        predicted_samples_arr.detach().cpu(),
        test_y_arr.detach().cpu(),
        True,
        generation_type,
    )
    print(prediction_dict)
