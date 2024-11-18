import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from .model import UNetSmall


class LitLSTMUNetSmall(pl.LightningModule):
    def __init__(self, min_value, max_value, to_pad=0):
        super().__init__()
        self.model = UNetSmall(in_channels=3, out_channels=3, init_features=32)
        self.min_value = min_value
        self.max_value = max_value
        self.to_pad = to_pad
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        ## Add padding of 56 pixels on the batch
        x, y = batch
        ## add padding to x
        opt = self.optimizers()
        x = self.model(x)
        loss = F.mse_loss(x, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.log(f"Learning Rate", self.scheduler.get_last_lr()[0])
        self.scheduler.step()
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
        # x = x * (self.max_value - self.min_value) + self.min_value
        # y = y * (self.max_value - self.min_value) + self.min_value
        #### Separate the channels and denormalize
        x[:, 0, :, :] = (
            x[:, 0, :, :] * (self.max_value[0] - self.min_value[0]) + self.min_value[0]
        )
        x[:, 1, :, :] = (
            x[:, 1, :, :] * (self.max_value[1] - self.min_value[1]) + self.min_value[1]
        )
        x[:, 2, :, :] = (
            x[:, 2, :, :] * (self.max_value[2] - self.min_value[2]) + self.min_value[2]
        )
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = 1e-5)
        return optimizer
