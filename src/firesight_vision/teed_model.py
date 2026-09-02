from __future__ import annotations

import torch
from torch import nn


def _smish(input_tensor: torch.Tensor) -> torch.Tensor:
    return input_tensor * torch.tanh(torch.log1p(torch.sigmoid(input_tensor)))


class _Smish(nn.Module):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return _smish(input_tensor)


def _weight_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.xavier_normal_(module.weight, gain=1.0)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__()
        self.add_module(
            "conv1",
            nn.Conv2d(
                input_features,
                output_features,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True,
            ),
        )
        self.add_module("smish1", _Smish())
        self.add_module(
            "conv2",
            nn.Conv2d(
                output_features,
                output_features,
                kernel_size=3,
                stride=1,
                bias=True,
            ),
        )

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = inputs
        new_features = super().forward(_smish(first))
        return 0.5 * (new_features + second), second


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        num_layers: int,
        input_features: int,
        output_features: int,
    ) -> None:
        super().__init__()
        for index in range(num_layers):
            self.add_module(
                f"denselayer{index + 1}",
                _DenseLayer(input_features, output_features),
            )
            input_features = output_features


class _UpConvBlock(nn.Module):
    def __init__(self, input_features: int, up_scale: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        all_pads = [0, 0, 1, 3, 7]
        kernel_size = 2**up_scale
        pad = all_pads[up_scale]
        for index in range(up_scale):
            output_features = 1 if index == up_scale - 1 else 16
            layers.extend(
                [
                    nn.Conv2d(input_features, output_features, 1),
                    _Smish(),
                    nn.ConvTranspose2d(
                        output_features,
                        output_features,
                        kernel_size,
                        stride=2,
                        padding=pad,
                    ),
                ],
            )
            input_features = output_features
        self.features = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.features(input_tensor)


class _SingleConvBlock(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        stride: int,
        *,
        use_activation: bool = False,
    ) -> None:
        super().__init__()
        self.use_activation = use_activation
        self.conv = nn.Conv2d(
            input_features,
            output_features,
            1,
            stride=stride,
            bias=True,
        )
        if use_activation:
            self.smish = _Smish()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.conv(input_tensor)
        if self.use_activation:
            output = self.smish(output)
        return output


class _DoubleConvBlock(nn.Module):
    def __init__(
        self,
        input_features: int,
        middle_features: int,
        output_features: int | None = None,
        stride: int = 1,
        *,
        use_activation: bool = True,
    ) -> None:
        super().__init__()
        self.use_activation = use_activation
        if output_features is None:
            output_features = middle_features
        self.conv1 = nn.Conv2d(
            input_features,
            middle_features,
            3,
            padding=1,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(middle_features, output_features, 3, padding=1)
        self.smish = _Smish()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.smish(self.conv1(input_tensor))
        output = self.conv2(output)
        if self.use_activation:
            output = self.smish(output)
        return output


class _DoubleFusion(nn.Module):
    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.DWconv1 = nn.Conv2d(
            input_features,
            input_features * 8,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=input_features,
        )
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(
            24,
            24 * 1,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=24,
        )
        self.AF = _Smish()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        attention = self.PSconv1(self.DWconv1(self.AF(input_tensor)))
        attention_2 = self.PSconv1(self.DWconv2(self.AF(attention)))
        return _smish((attention_2 + attention).sum(1).unsqueeze(1))


class TED(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_1 = _DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = _DoubleConvBlock(16, 32, use_activation=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = _SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = _SingleConvBlock(32, 48, 1)
        self.up_block_1 = _UpConvBlock(16, 1)
        self.up_block_2 = _UpConvBlock(32, 1)
        self.up_block_3 = _UpConvBlock(48, 2)
        self.block_cat = _DoubleFusion(3)
        self.apply(_weight_init)

    def forward(
        self,
        input_tensor: torch.Tensor,
        *,
        single_test: bool = False,
    ) -> list[torch.Tensor]:
        del single_test
        block_1 = self.block_1(input_tensor)
        block_1_side = self.side_1(block_1)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])

        outputs = [
            self.up_block_1(block_1),
            self.up_block_2(block_2),
            self.up_block_3(block_3),
        ]
        block_cat = self.block_cat(torch.cat(outputs, dim=1))
        outputs.append(block_cat)
        return outputs
