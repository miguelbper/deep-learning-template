"""
This type stub file was generated by pyright.
"""

import torch.nn as nn
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor
from .._api import WeightsEnum, register_model
from .._utils import handle_legacy_interface

__all__ = ["VideoResNet", "R3D_18_Weights", "MC3_18_Weights", "R2Plus1D_18_Weights", "r3d_18", "mc3_18", "r2plus1d_18"]
class Conv3DSimple(nn.Conv3d):
    def __init__(self, in_planes: int, out_planes: int, midplanes: Optional[int] = ..., stride: int = ..., padding: int = ...) -> None:
        ...

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        ...



class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = ..., padding: int = ...) -> None:
        ...

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        ...



class Conv3DNoTemporal(nn.Conv3d):
    def __init__(self, in_planes: int, out_planes: int, midplanes: Optional[int] = ..., stride: int = ..., padding: int = ...) -> None:
        ...

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        ...



class BasicBlock(nn.Module):
    expansion = ...
    def __init__(self, inplanes: int, planes: int, conv_builder: Callable[..., nn.Module], stride: int = ..., downsample: Optional[nn.Module] = ...) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...



class Bottleneck(nn.Module):
    expansion = ...
    def __init__(self, inplanes: int, planes: int, conv_builder: Callable[..., nn.Module], stride: int = ..., downsample: Optional[nn.Module] = ...) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...



class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""
    def __init__(self) -> None:
        ...



class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution"""
    def __init__(self) -> None:
        ...



class VideoResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]], layers: List[int], stem: Callable[..., nn.Module], num_classes: int = ..., zero_init_residual: bool = ...) -> None:
        """Generic resnet video generator.

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): resnet building block
            conv_makers (List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]]): generator
                function for each layer
            layers (List[int]): number of blocks per layer
            stem (Callable[..., nn.Module]): module specifying the ResNet stem.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...



_COMMON_META = ...
class R3D_18_Weights(WeightsEnum):
    KINETICS400_V1 = ...
    DEFAULT = ...


class MC3_18_Weights(WeightsEnum):
    KINETICS400_V1 = ...
    DEFAULT = ...


class R2Plus1D_18_Weights(WeightsEnum):
    KINETICS400_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", R3D_18_Weights.KINETICS400_V1))
def r3d_18(*, weights: Optional[R3D_18_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VideoResNet:
    """Construct 18 layer Resnet3D model.

    .. betastatus:: video module

    Reference: `A Closer Look at Spatiotemporal Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__.

    Args:
        weights (:class:`~torchvision.models.video.R3D_18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.R3D_18_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.resnet.VideoResNet`` base class.
            Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.R3D_18_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", MC3_18_Weights.KINETICS400_V1))
def mc3_18(*, weights: Optional[MC3_18_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VideoResNet:
    """Construct 18 layer Mixed Convolution network as in

    .. betastatus:: video module

    Reference: `A Closer Look at Spatiotemporal Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__.

    Args:
        weights (:class:`~torchvision.models.video.MC3_18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MC3_18_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.resnet.VideoResNet`` base class.
            Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MC3_18_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", R2Plus1D_18_Weights.KINETICS400_V1))
def r2plus1d_18(*, weights: Optional[R2Plus1D_18_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VideoResNet:
    """Construct 18 layer deep R(2+1)D network as in

    .. betastatus:: video module

    Reference: `A Closer Look at Spatiotemporal Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__.

    Args:
        weights (:class:`~torchvision.models.video.R2Plus1D_18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.R2Plus1D_18_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.resnet.VideoResNet`` base class.
            Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.R2Plus1D_18_Weights
        :members:
    """
    ...

model_urls = ...
