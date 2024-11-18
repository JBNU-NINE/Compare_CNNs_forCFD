import torch
import torch.nn as nn
from .convlstm import ConvLSTMCell, ConvLSTM
from collections import OrderedDict

import torch
import torch.nn as nn


class UNetSmall(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetSmall, self).__init__()

        features = init_features
        self.encoder1 = UNetSmall._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetSmall._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetSmall._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNetSmall._block(
            features * 4, features * 8, name="bottleneck"
        )
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNetSmall._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNetSmall._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNetSmall._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.conv_lstm = ConvLSTM(
            input_dim=3,
            hidden_dim=[32, 32, 3],
            kernel_size=(3, 3),
            num_layers=3,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )

    def forward(self, x):
        _, x = self.conv_lstm(x)
        enc1 = self.encoder1(x[0][1])
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


if __name__ == "__main__":
    # Batch Size, Time Steps, Channels, Height, Width
    x = torch.rand((32, 8, 3, 200, 200))
    # channels = 10
    # model = ConvLSTM(
    #     input_dim=channels,
    #     hidden_dim=[32, 32, 3],
    #     kernel_size=(3, 3),
    #     num_layers=3,
    #     batch_first=True,
    #     bias=True,
    #     return_all_layers=False,
    # )
    # _, last_states = model(x)
    # h = last_states[0][0]
    # print(h.shape)
    model = UNetSmall(3, 3, 32)
    out = model(x)
    print(out.shape)
