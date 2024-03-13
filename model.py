from collections import namedtuple

import torch
from torchvision import models
from typing import List, Tuple


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


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad: bool = False) -> None:
        super().__init__()
        # VGG16 pretrained model features
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple:
        h = self.slice1(x)
        relu1_2 = h

        h = self.slice2(h)
        relu2_2 = h

        h = self.slice3(h)
        relu3_3 = h

        h = self.slice4(h)
        relu4_3 = h

        outputs = namedtuple("Outputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = outputs(relu1_2, relu2_2, relu3_3, relu4_3)

        return out
