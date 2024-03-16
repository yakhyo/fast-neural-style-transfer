import torch
from typing import List


class TransformerNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        filters: List[int] = [3, 32, 64, 128]

        # Convolutional Layers
        self.conv1 = ConvLayer(filters[0], filters[1], kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(filters[1], affine=True)

        self.conv2 = ConvLayer(filters[1], filters[2], kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(filters[2], affine=True)

        self.conv3 = ConvLayer(filters[2], filters[3], kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(filters[3], affine=True)

        # Residual Layers
        self.res1 = ResidualBlock(filters[3])
        self.res2 = ResidualBlock(filters[3])
        self.res3 = ResidualBlock(filters[3])
        self.res4 = ResidualBlock(filters[3])
        self.res5 = ResidualBlock(filters[3])

        # Upsampling Layers
        self.upsample_conv1 = ConvLayer(filters[3], filters[2], kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(filters[2], affine=True)

        self.upsample_conv2 = ConvLayer(filters[2], filters[1], kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(filters[1], affine=True)

        self.upsample_conv3 = ConvLayer(filters[1], filters[0], kernel_size=9, stride=1)

        # Non-linearity
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        y = self.relu(self.in4(self.upsample_conv1(y)))
        y = self.relu(self.in5(self.upsample_conv2(y)))

        y = self.upsample_conv3(y)

        return y


class ConvLayer(torch.nn.Module):
    """[Upsample] Convolutional Layer with Reflection Padding

    [Upsample the input and then] does a convolution. This method gives better results
    compared to ConvTranspose2d. <http://distill.pub/2016/deconv-checkerboard/>
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        upsample: int = None
    ) -> None:
        super().__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        if self.upsample is not None:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """Residual Block"""

    def __init__(self, planes: int) -> None:
        super().__init__()
        self.conv1 = ConvLayer(in_channels=planes, out_channels=planes, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(num_features=planes, affine=True)

        self.conv2 = ConvLayer(in_channels=planes, out_channels=planes, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(num_features=planes, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out + residual
