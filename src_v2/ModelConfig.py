from models import LitAutoEncoder, LitUNet, LitUNetSmall
from parser import TrainingConfig, ModelConfig, Config
from DataLoader import CustomDataLoader
import pytorch_lightning as pl
import os
from datetime import datetime
import json


class ModelSelector:
    model_dict = {
        "autoencoder": LitAutoEncoder,
        "unet": LitUNet,
        "unet_small": LitUNetSmall,
    }

    def __init__(self, *model_config):
        self._models = {}
        for model in model_config:
            self.add_model(model.model_name)

    def add_model(self, model_name):
        self._models[model_name] = model_name

    def __str__(self):
        return f"ModelSelector({self._models})"

    def get_model(self, model_name):
        model = self.model_dict.get(model_name)
        return model


class CustomModelTrainer:
    def __init__(self, training_config, model_config, data_path):
        self.training_config = training_config
        self.model_config = model_config
        self.data_path = data_path
        self.callbacks = []
        self.__load_dataset()
        self.__get_model()
        self.__set_logger(self.training_config.log_dir)
        self.__set_earlystopping(training_config.early_stopping)
        self.__set_modelcheckpoint(training_config.model_checkpoint)

    def __set_modelcheckpoint(self, model_checkpoint):
        if model_checkpoint.status:
            self.callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    monitor=model_checkpoint.monitor,
                    mode=model_checkpoint.mode,
                )
            )

    def __set_earlystopping(self, early_stopping):
        if early_stopping.status:
            self.callbacks.append(
                pl.callbacks.EarlyStopping(
                    monitor=early_stopping.monitor,
                    patience=early_stopping.patience,
                    mode=early_stopping.mode,
                )
            )

    def __load_dataset(self):
        self.loader = CustomDataLoader(self.data_path, self.training_config)
        self.loader.set_required_data()

    def __get_model(self):
        self.model = ModelSelector(self.model_config).get_model(
            self.model_config.model_name
        )(self.loader.min_value, self.loader.max_value)

    def __set_logger(self, log_dir):
        if not os.path.exists(log_dir):
            print(f"Creating log directory, Currently Doesn't exist {log_dir}")
            os.makedirs(log_dir)
        else:
            print(f"Log directory already exists {log_dir}, Overwriting")

        # Saving Training Config Too:
        self.log_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logger = pl.loggers.TensorBoardLogger(
            "logs/", name=self.model_config.model_name, version=self.log_version
        )

    def save_class(self, obj, path):
        if isinstance(obj, Config):
            for key, value in obj.__dict__.items():
                if isinstance(value, Config):
                    self.save_class(value, path)
                else:
                    with open(path, "a") as f:
                        f.write(f"{key} : {value}\n")

    def save_training_config(self):
        training_config_path = os.path.join(
            self.training_config.log_dir,
            self.model_config.model_name,
            self.log_version,
            "training_config.txt",
        )
        self.save_class(self.training_config, training_config_path)

    def train_model(self, save_config=True):
        train_loader = self.loader.get_data("train")
        val_loader = self.loader.get_data("val")
        self.trainer = pl.Trainer(
            accelerator=self.training_config.device,
            devices=self.training_config.devices,
            max_epochs=self.training_config.epochs,
            logger=self.logger,
            callbacks=self.callbacks,
            log_every_n_steps=5,  # added because of warning and to see logs for training epoch
        )
        self.trainer.fit(self.model, train_loader, val_loader)
        if save_config:
            self.save_training_config()

    def test_model(self):
        test_loader = self.loader.get_data("test")
        self.trainer.test(self.model, test_loader, ckpt_path="best")


if __name__ == "__main__":
    default_config = TrainingConfig()
    default_model_config = ModelConfig()
    # modelSelector = ModelSelector(default_model_config)
    # model = modelSelector.get_model(default_model_config.model_name)
    model_trainer = CustomModelTrainer(
        default_config,
        default_model_config,
        "/home/sangam/workspace/sangam/python_practice/ml-convection/dataset",
    )
    model_trainer.train_model()
