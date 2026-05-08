from typing import Any
from torch import nn
from torch.nn import functional as F
import torch


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        _, _, kernel_H, kernel_W = self.weight.size()
        self.mask = nn.Buffer(torch.ones_like(self.weight.data))
        # mask right
        self.mask[:, :, kernel_H // 2, kernel_W // 2 + int(mask_type == "B"):] = 0
        # mask down
        self.mask[:, :, kernel_H // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class Conv2dBlock(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.conv1 = MaskedConv2d("B", in_channels=channel, out_channels=channel//2, kernel_size=1, stride=1, padding=0)
        self.conv2 = MaskedConv2d("B", in_channels=channel//2, out_channels=channel//2, kernel_size=3, stride=1, padding=1)
        self.conv3 = MaskedConv2d("B", in_channels=channel//2, out_channels=channel, kernel_size=1, stride=1, padding=0)

    def forward(self, input_image: torch.Tensor) ->torch.Tensor:
        output = self.conv1(F.relu(input_image))
        output = self.conv2(F.relu(output))
        output = self.conv3(F.relu(output))
        output = output + input_image

        return output


class PixelCNN(nn.Module):
    def __init__(self, in_channel: int, filters_num: int, out_channel: int, residual_num: int) -> None:
        super().__init__()

        # Mask A
        self.conv1 = MaskedConv2d("A", in_channel, filters_num, 7, 1, 3)

        # Mask B
        # residual block
        self.res_block = nn.ModuleList()
        for _ in range(residual_num):
            layer = Conv2dBlock(filters_num)
            self.res_block.append(layer)

        # output
        self.outputconv = MaskedConv2d("B", filters_num, out_channel, 1, 1, 0)


    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        output = self.conv1(input_image)
        for layers in self.res_block:
            output = layers(output)

        output = self.outputconv(output)
        return output
