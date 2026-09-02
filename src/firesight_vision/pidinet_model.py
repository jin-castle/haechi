from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CSAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        return x * self.sigmoid(y)


class CDCM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            dilation=5,
            padding=5,
            bias=False,
        )
        self.conv2_2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            dilation=7,
            padding=7,
            bias=False,
        )
        self.conv2_3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            dilation=9,
            padding=9,
            bias=False,
        )
        self.conv2_4 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            dilation=11,
            padding=11,
            bias=False,
        )
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.relu1(x))
        return (
            self.conv2_1(x)
            + self.conv2_2(x)
            + self.conv2_3(x)
            + self.conv2_4(x)
        )


class MapReduce(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PDCBlockConverted(nn.Module):
    def __init__(
        self,
        pdc: str,
        inplane: int,
        ouplane: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        kernel_size = 5 if pdc == "rd" else 3
        self.conv1 = nn.Conv2d(
            inplane,
            inplane,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inplane,
            bias=False,
        )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            inplane,
            ouplane,
            kernel_size=1,
            padding=0,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv2(self.relu2(self.conv1(x)))
        if self.stride > 1:
            x = self.shortcut(x)
        return y + x


class PiDiNet(nn.Module):
    def __init__(
        self,
        inplane: int,
        pdcs: tuple[str, ...],
        dil: int | None,
        sa: bool,
    ) -> None:
        super().__init__()
        self.sa = sa
        self.dil = dil
        self.fuseplanes = [inplane, inplane * 2, inplane * 4, inplane * 4]
        self.init_block = nn.Conv2d(
            3,
            inplane,
            kernel_size=5 if pdcs[0] == "rd" else 3,
            padding=2 if pdcs[0] == "rd" else 1,
            bias=False,
        )
        block_class = PDCBlockConverted

        self.block1_1 = block_class(pdcs[1], inplane, inplane)
        self.block1_2 = block_class(pdcs[2], inplane, inplane)
        self.block1_3 = block_class(pdcs[3], inplane, inplane)

        inplane = self.fuseplanes[0]
        self.block2_1 = block_class(pdcs[4], inplane, self.fuseplanes[1], stride=2)
        self.block2_2 = block_class(pdcs[5], self.fuseplanes[1], self.fuseplanes[1])
        self.block2_3 = block_class(pdcs[6], self.fuseplanes[1], self.fuseplanes[1])
        self.block2_4 = block_class(pdcs[7], self.fuseplanes[1], self.fuseplanes[1])

        self.block3_1 = block_class(
            pdcs[8], self.fuseplanes[1], self.fuseplanes[2], stride=2
        )
        self.block3_2 = block_class(pdcs[9], self.fuseplanes[2], self.fuseplanes[2])
        self.block3_3 = block_class(pdcs[10], self.fuseplanes[2], self.fuseplanes[2])
        self.block3_4 = block_class(pdcs[11], self.fuseplanes[2], self.fuseplanes[2])

        self.block4_1 = block_class(
            pdcs[12], self.fuseplanes[2], self.fuseplanes[3], stride=2
        )
        self.block4_2 = block_class(pdcs[13], self.fuseplanes[3], self.fuseplanes[3])
        self.block4_3 = block_class(pdcs[14], self.fuseplanes[3], self.fuseplanes[3])
        self.block4_4 = block_class(pdcs[15], self.fuseplanes[3], self.fuseplanes[3])

        self.conv_reduces = nn.ModuleList()
        if sa and dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for channels in self.fuseplanes:
                self.dilations.append(CDCM(channels, dil))
                self.attentions.append(CSAM(dil))
                self.conv_reduces.append(MapReduce(dil))
        elif sa:
            self.attentions = nn.ModuleList()
            for channels in self.fuseplanes:
                self.attentions.append(CSAM(channels))
                self.conv_reduces.append(MapReduce(channels))
        elif dil is not None:
            self.dilations = nn.ModuleList()
            for channels in self.fuseplanes:
                self.dilations.append(CDCM(channels, dil))
                self.conv_reduces.append(MapReduce(dil))
        else:
            for channels in self.fuseplanes:
                self.conv_reduces.append(MapReduce(channels))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1)
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        height, width = x.shape[2:]
        x = self.init_block(x)
        x1 = self.block1_3(self.block1_2(self.block1_1(x)))
        x2 = self.block2_4(
            self.block2_3(self.block2_2(self.block2_1(x1))),
        )
        x3 = self.block3_4(
            self.block3_3(self.block3_2(self.block3_1(x2))),
        )
        x4 = self.block4_4(
            self.block4_3(self.block4_2(self.block4_1(x3))),
        )

        features = [x1, x2, x3, x4]
        fused: list[torch.Tensor] = []
        if self.sa and self.dil is not None:
            fused = [
                attention(dilation(feature))
                for attention, dilation, feature in zip(
                    self.attentions,
                    self.dilations,
                    features,
                    strict=True,
                )
            ]
        elif self.sa:
            fused = [
                attention(feature)
                for attention, feature in zip(
                    self.attentions,
                    features,
                    strict=True,
                )
            ]
        elif self.dil is not None:
            fused = [
                dilation(feature)
                for dilation, feature in zip(
                    self.dilations,
                    features,
                    strict=True,
                )
            ]
        else:
            fused = features

        outputs = [
            F.interpolate(
                reducer(feature),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            for reducer, feature in zip(self.conv_reduces, fused, strict=True)
        ]
        outputs.append(self.classifier(torch.cat(outputs, dim=1)))
        return [torch.sigmoid(output) for output in outputs]


CARV4_OPERATIONS: tuple[str, ...] = (
    "cd",
    "ad",
    "rd",
    "cv",
    "cd",
    "ad",
    "rd",
    "cv",
    "cd",
    "ad",
    "rd",
    "cv",
    "cd",
    "ad",
    "rd",
    "cv",
)
