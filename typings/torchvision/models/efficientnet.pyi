"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Union
from torch import Tensor, nn
from ._api import WeightsEnum, register_model
from ._utils import handle_legacy_interface

__all__ = ["EfficientNet", "EfficientNet_B0_Weights", "EfficientNet_B1_Weights", "EfficientNet_B2_Weights", "EfficientNet_B3_Weights", "EfficientNet_B4_Weights", "EfficientNet_B5_Weights", "EfficientNet_B6_Weights", "EfficientNet_B7_Weights", "EfficientNet_V2_S_Weights", "EfficientNet_V2_M_Weights", "EfficientNet_V2_L_Weights", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7", "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"]
@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]
    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = ...) -> int:
        ...



class MBConvConfig(_MBConvConfig):
    def __init__(self, expand_ratio: float, kernel: int, stride: int, input_channels: int, out_channels: int, num_layers: int, width_mult: float = ..., depth_mult: float = ..., block: Optional[Callable[..., nn.Module]] = ...) -> None:
        ...

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float): # -> int:
        ...



class FusedMBConvConfig(_MBConvConfig):
    def __init__(self, expand_ratio: float, kernel: int, stride: int, input_channels: int, out_channels: int, num_layers: int, block: Optional[Callable[..., nn.Module]] = ...) -> None:
        ...



class MBConv(nn.Module):
    def __init__(self, cnf: MBConvConfig, stochastic_depth_prob: float, norm_layer: Callable[..., nn.Module], se_layer: Callable[..., nn.Module] = ...) -> None:
        ...

    def forward(self, input: Tensor) -> Tensor:
        ...



class FusedMBConv(nn.Module):
    def __init__(self, cnf: FusedMBConvConfig, stochastic_depth_prob: float, norm_layer: Callable[..., nn.Module]) -> None:
        ...

    def forward(self, input: Tensor) -> Tensor:
        ...



class EfficientNet(nn.Module):
    def __init__(self, inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]], dropout: float, stochastic_depth_prob: float = ..., num_classes: int = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., last_channel: Optional[int] = ...) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...



_COMMON_META: Dict[str, Any] = ...
_COMMON_META_V1 = ...
_COMMON_META_V2 = ...
class EfficientNet_B0_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class EfficientNet_B1_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class EfficientNet_B2_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class EfficientNet_B3_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class EfficientNet_B4_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class EfficientNet_B5_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class EfficientNet_B6_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class EfficientNet_B7_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class EfficientNet_V2_S_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class EfficientNet_V2_M_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class EfficientNet_V2_L_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B0_Weights.IMAGENET1K_V1))
def efficientnet_b0(*, weights: Optional[EfficientNet_B0_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet:
    """EfficientNet B0 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B0_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B0_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B0_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B1_Weights.IMAGENET1K_V1))
def efficientnet_b1(*, weights: Optional[EfficientNet_B1_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet:
    """EfficientNet B1 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B1_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B1_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B1_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B2_Weights.IMAGENET1K_V1))
def efficientnet_b2(*, weights: Optional[EfficientNet_B2_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet:
    """EfficientNet B2 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B2_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B3_Weights.IMAGENET1K_V1))
def efficientnet_b3(*, weights: Optional[EfficientNet_B3_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet:
    """EfficientNet B3 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B3_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B3_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B3_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B4_Weights.IMAGENET1K_V1))
def efficientnet_b4(*, weights: Optional[EfficientNet_B4_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet:
    """EfficientNet B4 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B4_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B4_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B4_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B5_Weights.IMAGENET1K_V1))
def efficientnet_b5(*, weights: Optional[EfficientNet_B5_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet:
    """EfficientNet B5 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B5_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B5_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B5_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B6_Weights.IMAGENET1K_V1))
def efficientnet_b6(*, weights: Optional[EfficientNet_B6_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet:
    """EfficientNet B6 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B6_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B6_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B6_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_B7_Weights.IMAGENET1K_V1))
def efficientnet_b7(*, weights: Optional[EfficientNet_B7_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet:
    """EfficientNet B7 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B7_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B7_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B7_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_V2_S_Weights.IMAGENET1K_V1))
def efficientnet_v2_s(*, weights: Optional[EfficientNet_V2_S_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet:
    """
    Constructs an EfficientNetV2-S architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_S_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_V2_M_Weights.IMAGENET1K_V1))
def efficientnet_v2_m(*, weights: Optional[EfficientNet_V2_M_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet:
    """
    Constructs an EfficientNetV2-M architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_M_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_M_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_M_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", EfficientNet_V2_L_Weights.IMAGENET1K_V1))
def efficientnet_v2_l(*, weights: Optional[EfficientNet_V2_L_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_L_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_L_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_L_Weights
        :members:
    """
    ...
