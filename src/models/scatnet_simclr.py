from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kymatio.torch import Scattering2D


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ScatSimCLR(nn.Module):

    N_RES_BLOCKS = [8, 12, 16, 30, 45]

    INPLANES = {
        8: 32,
        12: 256,
        16: 256,
        30: 256,
        45: 256
    }

    def __init__(self, J: int, L: int, input_size: Tuple[int, int, int], res_blocks: int, out_dim: int):

        """
        Args:
            J: ScatNet scale parameter

            L: ScatNet rotation parameter

            input_size: input image size. It should be (H, W, C)

            res_blocks: number of ResBlocks in adaptor network

            out_dim: output dimension of the projection space

        Raises:
            ValueError: if `J` parameter is < 1

            ValueError: if `L` parameter is < 1

            ValueError: if `input_size` is incorrect shape

            ValueError: if `res_blocks` is not supported
        """

        super(ScatSimCLR, self).__init__()

        if J < 1:
            raise ValueError('Incorrect `J` parameter')

        if L < 1:
            raise ValueError('Incorrect `L` parameter')

        if len(input_size) != 3:
            raise ValueError('`input_size` parameter should be (H, W, C)')

        if res_blocks not in self.N_RES_BLOCKS:
            raise ValueError(f'Incorrect `res_blocks` parameter. Is should be in:'
                             f'[{", ".join(self.N_RES_BLOCKS)}]')

        self._J = J
        self._L = L

        self._res_blocks = res_blocks
        self._out_dim = out_dim

        # get image height, width and channels
        h, w, c = input_size
        # ScatNet is applied for each image channel separately
        self._num_scatnet_channels = c * ((L * L * J * (J - 1)) // 2 + L * J + 1)

        # max order is always 2 - maximum possible
        self._scatnet = Scattering2D(J=J, shape=(h, w), L=L, max_order=2)

        # batch size, which is applied to ScatNet features
        self._scatnet_bn = nn.BatchNorm2d(self._num_scatnet_channels)

        # pool size in adapter network
        self._pool_size = 4

        self.inplanes = self.INPLANES[self._res_blocks]
        # adapter network
        self._adapter_network = self._create_adapter_network()

        # linear layers
        num_ftrs = 128 * self._pool_size ** 2
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _create_adapter_network(self) -> nn.Module:
        ichannels = 32 if self._res_blocks == 8 else 256

        adapter_layers = [
            # initial conv
            conv3x3(self._num_scatnet_channels, ichannels),
            nn.BatchNorm2d(ichannels),
            nn.ReLU(True),
        ]

        if self._res_blocks == 8:

            adapter_layers.extend([
                # ResBlocks
                self._make_layer(ResBlock, 64, 4),
                self._make_layer(ResBlock, 128, 4),
            ])

        elif self._res_blocks == 12:

            adapter_layers.extend([
                # ResBlocks
                self._make_layer(ResBlock, 128, 4),
                self._make_layer(ResBlock, 64, 4),
                self._make_layer(ResBlock, 128, 4),
            ])

        elif self._res_blocks == 16:

            adapter_layers.extend([
                # ResBlocks
                self._make_layer(ResBlock, 128, 6),
                self._make_layer(ResBlock, 64, 4),
                self._make_layer(ResBlock, 128, 6)
            ])

        elif self._res_blocks == 30:

            adapter_layers.extend([
                # ResBlocks
                self._make_layer(ResBlock, 128, 30)
            ])

        elif self._res_blocks == 45:

            adapter_layers.extend([
                # ResBlocks
                self._make_layer(ResBlock, 128, 15),
                self._make_layer(ResBlock, 64, 15),
                self._make_layer(ResBlock, 128, 15)
            ])

        adapter_layers.append(nn.AdaptiveAvgPool2d(self._pool_size))
        return nn.Sequential(*adapter_layers)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scatnet = self._scatnet(x).squeeze(1)

        B, C, FN, H, W = scatnet.size()
        scatnet = scatnet.view(B, C * FN, H, W)

        h = self._adapter_network(scatnet)
        h = h.view(h.size(0), -1)

        z = self.l2(F.relu(self.l1(h)))
        return h, z
