import torch
import torch.nn as nn
from collections import OrderedDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class _UNet(nn.Module):
    '''
    '''

    def __init__(self, in_channels=3, latent_channels=64, out_channels=1, features=[32,32,32]):
        super(_UNet, self).__init__()

        if isinstance(features, int):
            features = [features] * 3

        self.inputs = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.encoder1_1 = _UNet._block(in_channels=in_channels, out_channels=features[0], name="enc1_1")
        self.encoder1_2 = _UNet._block(in_channels=features[0], out_channels=features[0], name="enc1_2")
        self.encoder1_3 = _UNet._block(in_channels=features[0], out_channels=features[0], name="enc1_3")
        self.encoder2_1 = _UNet._block(in_channels=features[0], out_channels=features[1], name="enc2_1")
        self.encoder2_2 = _UNet._block(in_channels=features[1], out_channels=features[1], name="enc2_2")
        self.encoder2_3 = _UNet._block(in_channels=features[1], out_channels=features[1], name="enc2_3")
        self.encoder3_1 = _UNet._block(in_channels=features[1], out_channels=features[2], name="enc3_1")
        self.encoder3_2 = _UNet._block(in_channels=features[2], out_channels=features[2], name="enc3_2")
        self.encoder3_3 = _UNet._block(in_channels=features[2], out_channels=features[2], name="enc3_3")
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck1 = nn.Conv2d(in_channels=features[2], out_channels=latent_channels, kernel_size=3, padding=1)
        self.bottleneck2 = nn.Conv2d(in_channels=latent_channels, out_channels=features[2], kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder3_3 = _UNet._block(in_channels=features[2] * 2, out_channels=features[2], name="dec3_3")
        self.decoder3_2 = _UNet._block(in_channels=features[2], out_channels=features[2], name="dec3_2")
        self.decoder3_1 = _UNet._block(in_channels=features[2], out_channels=features[2], name="dec3_1")
        self.decoder2_3 = _UNet._block(in_channels=features[1] * 2, out_channels=features[1], name="dec2_3")
        self.decoder2_2 = _UNet._block(in_channels=features[1], out_channels=features[1], name="dec2_2")
        self.decoder2_1 = _UNet._block(in_channels=features[1], out_channels=features[1], name="dec2_1")
        self.decoder1_3 = _UNet._block(in_channels=features[0] * 2, out_channels=features[0], name="dec1_3")
        self.decoder1_2 = _UNet._block(in_channels=features[0], out_channels=features[0], name="dec1_2")
        self.decoder1_1 = _UNet._block(in_channels=features[0], out_channels=features[0], name="dec1_1")
        self.outputs = nn.Conv2d(in_channels=features[0], out_channels=out_channels, padding=1)

    def forward(self, x):
        x = self.inputs(x)
        x = self.encoder1_1(x)
        x = self.encoder1_2(x)
        skip1 = self.encoder1_3(x)
        x = self.pool(skip1)
        x = self.encoder2_1(x)
        x = self.encoder2_2(x)
        skip2 = self.encoder2_3(x)
        x = self.pool(skip2)
        x = self.encoder3_1(x)
        x = self.encoder3_2(x)
        skip3 = self.encoder3_3(x)
        x = self.pool(skip3)

        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        x = self.upsample(x)
        x = self.decoder3_3(torch.cat((x, skip3), dim=1))
        x = self.decoder3_2(x)
        x = self.decoder3_1(x)
        x = self.upsample(x)
        x = self.decoder2_3(torch.cat((x, skip2), dim=1))
        x = self.decoder2_2(x)
        x = self.decoder2_1(x)
        x = self.upsample(x)
        x = self.decoder1_3(torch.cat((x, skip1), dim=1))
        x = self.decoder1_2(x)
        x = self.decoder1_1(x)
        map = torch.sigmoid(self.outputs(x))
        return map

    @staticmethod
    def _block(in_channels, out_channels, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "relu", nn.ReLU(inplace=True))
                ]
            )
        )