"""
This type stub file was generated by pyright.
"""

import torch
from typing import Callable, List, Optional, Tuple, Union
from torch import Tensor

interpolate = ...
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed

    Args:
        num_features (int): Number of features ``C`` from an expected input of size ``(N, C, H, W)``
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
    """
    def __init__(self, num_features: int, eps: float = ...) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...

    def __repr__(self) -> str:
        ...



class ConvNormActivation(torch.nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]] = ..., stride: Union[int, Tuple[int, ...]] = ..., padding: Optional[Union[int, Tuple[int, ...], str]] = ..., groups: int = ..., norm_layer: Optional[Callable[..., torch.nn.Module]] = ..., activation_layer: Optional[Callable[..., torch.nn.Module]] = ..., dilation: Union[int, Tuple[int, ...]] = ..., inplace: Optional[bool] = ..., bias: Optional[bool] = ..., conv_layer: Callable[..., torch.nn.Module] = ...) -> None:
        ...



class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]] = ..., stride: Union[int, Tuple[int, int]] = ..., padding: Optional[Union[int, Tuple[int, int], str]] = ..., groups: int = ..., norm_layer: Optional[Callable[..., torch.nn.Module]] = ..., activation_layer: Optional[Callable[..., torch.nn.Module]] = ..., dilation: Union[int, Tuple[int, int]] = ..., inplace: Optional[bool] = ..., bias: Optional[bool] = ...) -> None:
        ...



class Conv3dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution3d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input video.
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm3d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]] = ..., stride: Union[int, Tuple[int, int, int]] = ..., padding: Optional[Union[int, Tuple[int, int, int], str]] = ..., groups: int = ..., norm_layer: Optional[Callable[..., torch.nn.Module]] = ..., activation_layer: Optional[Callable[..., torch.nn.Module]] = ..., dilation: Union[int, Tuple[int, int, int]] = ..., inplace: Optional[bool] = ..., bias: Optional[bool] = ...) -> None:
        ...



class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """
    def __init__(self, input_channels: int, squeeze_channels: int, activation: Callable[..., torch.nn.Module] = ..., scale_activation: Callable[..., torch.nn.Module] = ...) -> None:
        ...

    def forward(self, input: Tensor) -> Tensor:
        ...



class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """
    def __init__(self, in_channels: int, hidden_channels: List[int], norm_layer: Optional[Callable[..., torch.nn.Module]] = ..., activation_layer: Optional[Callable[..., torch.nn.Module]] = ..., inplace: Optional[bool] = ..., bias: bool = ..., dropout: float = ...) -> None:
        ...



class Permute(torch.nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """
    def __init__(self, dims: List[int]) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...
