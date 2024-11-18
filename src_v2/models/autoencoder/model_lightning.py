from .model import AutoEncoder
import pytorch_lightning as pl
import torch.nn.functional as F
import torch


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.model = AutoEncoder()
        self.min_value = min_value
        self.max_value = max_value

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)
        loss = F.mse_loss(x, y)
        self.log(
            "train_losss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)
        loss = F.mse_loss(x, y)
        ######################LOGGING#############################
        self.log("val_loss", loss, on_epoch=True)
        mae = F.l1_loss(x, y)
        self.log("val_mae", mae, on_epoch=True)
        max_ae = torch.max(torch.abs(x - y))
        self.log("val_max_ae", max_ae, on_epoch=True)

        # if batch_idx % 100 == 0:
        #     self.logger.experiment.add_images(
        #         "val_images", x, global_step=self.global_step
        #     )

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)
        loss = F.mse_loss(x, y)
        self.log("test_loss", loss, on_epoch=True)
        # De normalize x and y before calculating MAE
        x = x * (self.max_value - self.min_value) + self.min_value
        y = y * (self.max_value - self.min_value) + self.min_value
        mae = F.l1_loss(x, y)
        self.log("test mae (Denormalized)", mae, on_epoch=True)
        max_ae = torch.max(torch.abs(x - y))
        self.log("test max_ae (Denormalized)", max_ae, on_epoch=True)
        # get all 3 channels from the image
        # TODO: Proper Testing individual channels and logging them.

        self.logger.experiment.add_images(
            "test_images", y, global_step=self.global_step
        )
        self.logger.experiment.add_images(
            "test_images_predicted", x, global_step=self.global_step
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
