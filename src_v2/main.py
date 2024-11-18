import os

from parser import TrainingConfig, ModelConfig
from ModelConfig import ModelSelector, CustomModelTrainer
from generate_samples import GenerateSamples
import torch
import gc


def get_ckpt_path(model_name, log_version):
    ckpt_dir = os.path.join("logs", model_name, log_version, "checkpoints")
    filename = os.listdir(ckpt_dir)[0]
    if filename.endswith(".ckpt"):
        return os.path.join(ckpt_dir, filename)
    raise Exception("No ckpt file found")


def main():
    default_config = TrainingConfig()
    default_model_config = ModelConfig()
    modelSelector = ModelSelector(default_model_config)
    model = modelSelector.get_model(default_model_config.model_name)
    model_trainer = CustomModelTrainer(
        default_config,
        default_model_config,
        "/home/ubuntu/ml-convection/dataset",
    )
    model_trainer.train_model()
    model_trainer.test_model()
    model_ckpt_path = get_ckpt_path(
        default_model_config.model_name, model_trainer.log_version
    )
    log_version = model_trainer.log_version
    del model_trainer
    del modelSelector
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------_TESTING-----------------------------------
    print(f"Loading Best weights from {model_ckpt_path}")
    if default_config.generate_samples:
        sample_generator = GenerateSamples(
            default_config,
            default_model_config,
            "/home/sangam/workspace/sangam/python_practice/ml-convection/dataset",
            log_version,
        )
        sample_generator.set_model(model_ckpt_path)
        # sample_generator.generate_samples(
        #     "sequential",
        #     default_config.test_data_range[0],
        #     default_config.test_data_range[1] - default_config.test_data_range[0],
        #     create_gif=True,
        # )
        sample_generator.generate_samples(
            "regressive",
            default_config.test_data_range[0],
            default_config.test_data_range[1] - default_config.test_data_range[0],
            create_gif=True,
        )
        sample_generator.generate_samples(
            "regressive_padding",
            default_config.test_data_range[0],
            default_config.test_data_range[1] - default_config.test_data_range[0],
            create_gif=True,
            padding=default_config.regressive_padding,
        )


if __name__ == "__main__":
    main()
