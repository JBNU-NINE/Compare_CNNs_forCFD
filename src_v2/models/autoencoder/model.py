import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_conv_block(filter_in, filter_out, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=filter_in,
            out_channels=filter_out,
            kernel_size=kernel_size,
            stride=1,
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class Encoder(nn.Module):
    def __init__(self, input_size=(3, 200, 200), latent_dim=128):
        super().__init__()
        self.input_size = input_size
        self.conv_block_1 = get_conv_block(3, 32, 3)
        self.conv_block_2 = get_conv_block(32, 32, 3)
        self.conv_block_3 = get_conv_block(32, 32, 3)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        return x


def get_deconv_block(in_channels=32, out_channels=32, stride=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
        ),
        nn.ReLU(),
    )


class Decoder(nn.Module):
    def __init__(self, input_size=(32, 23, 23)):
        super(Decoder, self).__init__()
        self.deconv_block_1 = get_deconv_block()
        self.deconv_block_2 = get_deconv_block(stride=1)
        self.deconv_block_3 = get_deconv_block(out_channels=12)
        self.out_deconv_block = nn.ConvTranspose2d(
            in_channels=12, out_channels=3, kernel_size=3, stride=1
        )

    def forward(self, x):
        x = self.deconv_block_1(x)
        x = self.deconv_block_2(x)
        x = self.deconv_block_3(x)
        x = self.out_deconv_block(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size=(3, 200, 200), hidden_size=1024):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    # encoder_test = Encoder()
    # # Torch expects (batch_size, channels, height, width) but tensorflow expects (batch_size, height, width, channels)
    # sample_data = np.random.rand(1, 3, 200, 200)
    # output = encoder_test(torch.from_numpy(sample_data).float())
    # print(output.shape)

    autoencoder_test = AutoEncoder((3, 200, 200), 128)
    sample_data = np.random.rand(20, 3, 200, 200)
    output = autoencoder_test(torch.from_numpy(sample_data).float())
    print(output.shape)
