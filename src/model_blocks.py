"""
This file contains different blocks of transfer learning models such as resnet blocks, spatial attention blocks etc
"""

import torch
import typing


def get_activation(activ_name: str = "relu"):
    """Consider moving this to activations file"""
    act_dict = {
        "relu": torch.nn.ReLU(inplace=True),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "identity": torch.nn.Identity(),
    }
    if activ_name in act_dict:
        return act_dict[activ_name]
    else:
        raise NotImplementedError


class Conv2dBNActiv(torch.nn.Module):
    """Conv2d -> (BN ->) -> Activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        use_bn: bool = True,
        activ: str = "relu",
    ):
        """"""
        super(Conv2dBNActiv, self).__init__()
        layers = []
        layers.append(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            )
        )
        if use_bn:
            layers.append(torch.nn.BatchNorm2d(out_channels))

        layers.append(get_activation(activ))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Forward"""
        return self.layers(x)


class SSEBlock(torch.nn.Module):
    """channel `S`queeze and `s`patial `E`xcitation Block."""

    def __init__(self, in_channels: int):
        """Initialize."""
        super(SSEBlock, self).__init__()
        self.channel_squeeze = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """Forward."""
        # # x: (bs, ch, h, w) => h: (bs, 1, h, w)
        h = self.sigmoid(self.channel_squeeze(x))
        # # x, h => return: (bs, ch, h, w)
        return x * h


class SpatialAttentionBlock(torch.nn.Module):
    """Spatial Attention for (C, H, W) feature maps"""

    def __init__(
        self,
        in_channels: int,
        out_channels_list: typing.List[int],
    ):
        """Initialize"""
        super(SpatialAttentionBlock, self).__init__()
        self.n_layers = len(out_channels_list)
        channels_list = [in_channels] + out_channels_list
        assert self.n_layers > 0
        assert channels_list[-1] == 1

        for i in range(self.n_layers - 1):
            in_chs, out_chs = channels_list[i : i + 2]
            layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="relu")
            setattr(self, f"conv{i + 1}", layer)

        in_chs, out_chs = channels_list[-2:]
        layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="sigmoid")
        setattr(self, f"conv{self.n_layers}", layer)

    def forward(self, x):
        """Forward"""
        h = x
        for i in range(self.n_layers):
            h = getattr(self, f"conv{i + 1}")(h)

        h = h * x
        return h