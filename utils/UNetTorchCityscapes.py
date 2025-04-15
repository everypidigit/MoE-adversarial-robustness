import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.encoder1 = conv_block(in_channels, 64)     # 128x256
        self.encoder2 = conv_block(64, 128)             # 64x128
        self.encoder3 = conv_block(128, 256)            # 32x64

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.middle = conv_block(256, 512)              # 16x32

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )   # 32x64
        self.decoder3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )   # 64x128
        self.decoder2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )    # 128x256
        self.decoder1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)              # [B, 64, 128, 256]
        enc2 = self.encoder2(self.pool(enc1))  # [B, 128, 64, 128]
        enc3 = self.encoder3(self.pool(enc2))  # [B, 256, 32, 64]

        # Bottleneck
        mid = self.middle(self.pool(enc3))     # [B, 512, 16, 32]

        # Decoder
        dec3 = self.decoder3(
            torch.cat(
                [self.up3(mid), enc3],
                dim=1
            )
        )   # [B, 256, 32, 64]
        dec2 = self.decoder2(
            torch.cat(
                [self.up2(dec3), enc2],
                dim=1
            )
        )  # [B, 128, 64, 128]
        dec1 = self.decoder1(
            torch.cat(
                [self.up1(dec2), enc1],
                dim=1
            )
        )  # [B, 64, 128, 256]

        return self.final(dec1)  # [B, 3, 128, 256]
