"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, Dict, List, Optional
from torch import Tensor, nn
from ..ops.misc import Conv2dNormActivation
from ._api import WeightsEnum, register_model
from ._utils import handle_legacy_interface

__all__ = ["RegNet", "RegNet_Y_400MF_Weights", "RegNet_Y_800MF_Weights", "RegNet_Y_1_6GF_Weights", "RegNet_Y_3_2GF_Weights", "RegNet_Y_8GF_Weights", "RegNet_Y_16GF_Weights", "RegNet_Y_32GF_Weights", "RegNet_Y_128GF_Weights", "RegNet_X_400MF_Weights", "RegNet_X_800MF_Weights", "RegNet_X_1_6GF_Weights", "RegNet_X_3_2GF_Weights", "RegNet_X_8GF_Weights", "RegNet_X_16GF_Weights", "RegNet_X_32GF_Weights", "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf", "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf", "regnet_y_128gf", "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_3_2gf", "regnet_x_8gf", "regnet_x_16gf", "regnet_x_32gf"]
class SimpleStemIN(Conv2dNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""
    def __init__(self, width_in: int, width_out: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module]) -> None:
        ...



class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""
    def __init__(self, width_in: int, width_out: int, stride: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int, bottleneck_multiplier: float, se_ratio: Optional[float]) -> None:
        ...



class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""
    def __init__(self, width_in: int, width_out: int, stride: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int = ..., bottleneck_multiplier: float = ..., se_ratio: Optional[float] = ...) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...



class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""
    def __init__(self, width_in: int, width_out: int, stride: int, depth: int, block_constructor: Callable[..., nn.Module], norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int, bottleneck_multiplier: float, se_ratio: Optional[float] = ..., stage_index: int = ...) -> None:
        ...



class BlockParams:
    def __init__(self, depths: List[int], widths: List[int], group_widths: List[int], bottleneck_multipliers: List[float], strides: List[int], se_ratio: Optional[float] = ...) -> None:
        ...

    @classmethod
    def from_init_params(cls, depth: int, w_0: int, w_a: float, w_m: float, group_width: int, bottleneck_multiplier: float = ..., se_ratio: Optional[float] = ..., **kwargs: Any) -> BlockParams:
        """
        Programmatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """
        ...



class RegNet(nn.Module):
    def __init__(self, block_params: BlockParams, num_classes: int = ..., stem_width: int = ..., stem_type: Optional[Callable[..., nn.Module]] = ..., block_type: Optional[Callable[..., nn.Module]] = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., activation: Optional[Callable[..., nn.Module]] = ...) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...



_COMMON_META: Dict[str, Any] = ...
_COMMON_SWAG_META = ...
class RegNet_Y_400MF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class RegNet_Y_800MF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class RegNet_Y_1_6GF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class RegNet_Y_3_2GF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class RegNet_Y_8GF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class RegNet_Y_16GF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    IMAGENET1K_SWAG_E2E_V1 = ...
    IMAGENET1K_SWAG_LINEAR_V1 = ...
    DEFAULT = ...


class RegNet_Y_32GF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    IMAGENET1K_SWAG_E2E_V1 = ...
    IMAGENET1K_SWAG_LINEAR_V1 = ...
    DEFAULT = ...


class RegNet_Y_128GF_Weights(WeightsEnum):
    IMAGENET1K_SWAG_E2E_V1 = ...
    IMAGENET1K_SWAG_LINEAR_V1 = ...
    DEFAULT = ...


class RegNet_X_400MF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class RegNet_X_800MF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class RegNet_X_1_6GF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class RegNet_X_3_2GF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class RegNet_X_8GF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class RegNet_X_16GF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


class RegNet_X_32GF_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_V2 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_Y_400MF_Weights.IMAGENET1K_V1))
def regnet_y_400mf(*, weights: Optional[RegNet_Y_400MF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_400MF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_400MF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_400MF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_400MF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_Y_800MF_Weights.IMAGENET1K_V1))
def regnet_y_800mf(*, weights: Optional[RegNet_Y_800MF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_800MF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_800MF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_800MF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_800MF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_Y_1_6GF_Weights.IMAGENET1K_V1))
def regnet_y_1_6gf(*, weights: Optional[RegNet_Y_1_6GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_1.6GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_1_6GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_1_6GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_1_6GF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_Y_3_2GF_Weights.IMAGENET1K_V1))
def regnet_y_3_2gf(*, weights: Optional[RegNet_Y_3_2GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_3.2GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_3_2GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_3_2GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_3_2GF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_Y_8GF_Weights.IMAGENET1K_V1))
def regnet_y_8gf(*, weights: Optional[RegNet_Y_8GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_8GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_8GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_8GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_8GF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_Y_16GF_Weights.IMAGENET1K_V1))
def regnet_y_16gf(*, weights: Optional[RegNet_Y_16GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_16GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_16GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_16GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_16GF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_Y_32GF_Weights.IMAGENET1K_V1))
def regnet_y_32gf(*, weights: Optional[RegNet_Y_32GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_32GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_32GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_32GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_32GF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", None))
def regnet_y_128gf(*, weights: Optional[RegNet_Y_128GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_128GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_128GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_128GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_128GF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_X_400MF_Weights.IMAGENET1K_V1))
def regnet_x_400mf(*, weights: Optional[RegNet_X_400MF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_400MF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_400MF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_400MF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_400MF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_X_800MF_Weights.IMAGENET1K_V1))
def regnet_x_800mf(*, weights: Optional[RegNet_X_800MF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_800MF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_800MF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_800MF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_800MF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_X_1_6GF_Weights.IMAGENET1K_V1))
def regnet_x_1_6gf(*, weights: Optional[RegNet_X_1_6GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_1.6GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_1_6GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_1_6GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_1_6GF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_X_3_2GF_Weights.IMAGENET1K_V1))
def regnet_x_3_2gf(*, weights: Optional[RegNet_X_3_2GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_3.2GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_3_2GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_3_2GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_3_2GF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_X_8GF_Weights.IMAGENET1K_V1))
def regnet_x_8gf(*, weights: Optional[RegNet_X_8GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_8GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_8GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_8GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_8GF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_X_16GF_Weights.IMAGENET1K_V1))
def regnet_x_16gf(*, weights: Optional[RegNet_X_16GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_16GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_16GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_16GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_16GF_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", RegNet_X_32GF_Weights.IMAGENET1K_V1))
def regnet_x_32gf(*, weights: Optional[RegNet_X_32GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_32GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_X_32GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_X_32GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_X_32GF_Weights
        :members:
    """
    ...
