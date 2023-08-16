import torch
import torch.nn as nn
from collections import OrderedDict

from layers.cbam_attention import CBAMBlock, SpatialAttention
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class CBAM(nn.Module):
    '''
    '''

    def __init__(self, in_channels=3, latent_channels=64, out_channels=1, features=[32,32,32]):
        super(CBAM, self).__init__()

        if isinstance(features, int):
            features = [features] * 3

        self.spatial_mask = SpatialAttention(kernel_size=7)
        self.inputs = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.encoder1_1 = CBAM._block(in_channels=in_channels, out_channels=features[0], cbam_kernel=7, name="enc1_1")
        self.encoder1_2 = CBAM._block(in_channels=features[0], out_channels=features[0], cbam_kernel=7, name="enc1_2")
        self.encoder1_3 = CBAM._block(in_channels=features[0], out_channels=features[0], cbam_kernel=7, name="enc1_3")
        self.encoder2_1 = CBAM._block(in_channels=features[0], out_channels=features[1], cbam_kernel=5, name="enc2_1")
        self.encoder2_2 = CBAM._block(in_channels=features[1], out_channels=features[1], cbam_kernel=5, name="enc2_2")
        self.encoder2_3 = CBAM._block(in_channels=features[1], out_channels=features[1], cbam_kernel=5, name="enc2_3")
        self.encoder3_1 = CBAM._block(in_channels=features[1], out_channels=features[2], cbam_kernel=3, name="enc3_1")
        self.encoder3_2 = CBAM._block(in_channels=features[2], out_channels=features[2], cbam_kernel=3, name="enc3_2")
        self.encoder3_3 = CBAM._block(in_channels=features[2], out_channels=features[2], cbam_kernel=3, name="enc3_3")
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck1 = nn.Conv2d(in_channels=features[2], out_channels=latent_channels, kernel_size=3, padding=1)
        self.bottleneck2 = nn.Conv2d(in_channels=latent_channels, out_channels=features[2], kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder3_3 = CBAM._block(in_channels=features[2] * 2, out_channels=features[2], cbam_kernel=3, name="dec3_3")
        self.decoder3_2 = CBAM._block(in_channels=features[2], out_channels=features[2], cbam_kernel=3, name="dec3_2")
        self.decoder3_1 = CBAM._block(in_channels=features[2], out_channels=features[2], cbam_kernel=3, name="dec3_1")
        self.decoder2_3 = CBAM._block(in_channels=features[1] * 2, out_channels=features[1], cbam_kernel=3, name="dec2_3")
        self.decoder2_2 = CBAM._block(in_channels=features[1], out_channels=features[1], cbam_kernel=3, name="dec2_2")
        self.decoder2_1 = CBAM._block(in_channels=features[1], out_channels=features[1], cbam_kernel=3, name="dec2_1")
        self.decoder1_3 = CBAM._block(in_channels=features[0] * 2, out_channels=features[0], cbam_kernel=3, name="dec1_3")
        self.decoder1_2 = CBAM._block(in_channels=features[0], out_channels=features[0], cbam_kernel=3, name="dec1_2")
        self.decoder1_1 = CBAM._block(in_channels=features[0], out_channels=features[0], cbam_kernel=3, name="dec1_1")
        self.outputs = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        input_mask = self.spatial_mask(x)
        input_conv = self.inputs(x)
        inputs = input_mask * input_conv
        x = x + inputs
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
    def _block(in_channels, out_channels, cbam_kernel, name):
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
                    (
                        name + "cbam",
                        CBAMBlock(
                          channel=out_channels,
                          reduction=out_channels//4,
                          kernel_size=cbam_kernel,
                        ),
                    ),
                    (name + "norm", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "relu", nn.ReLU(inplace=True))
                ]
            )
        )