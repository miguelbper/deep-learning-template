"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional, Union
from torch import Tensor
from torchvision.models.mobilenetv2 import InvertedResidual, MobileNetV2, MobileNet_V2_Weights
from .._api import WeightsEnum, register_model
from .._utils import handle_legacy_interface

__all__ = ["QuantizableMobileNetV2", "MobileNet_V2_QuantizedWeights", "mobilenet_v2"]
class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...

    def fuse_model(self, is_qat: Optional[bool] = ...) -> None:
        ...



class QuantizableMobileNetV2(MobileNetV2):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...

    def fuse_model(self, is_qat: Optional[bool] = ...) -> None:
        ...



class MobileNet_V2_QuantizedWeights(WeightsEnum):
    IMAGENET1K_QNNPACK_V1 = ...
    DEFAULT = ...


@register_model(name="quantized_mobilenet_v2")
@handle_legacy_interface(weights=("pretrained", lambda kwargs: MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1 if kwargs.get("quantize", False) else MobileNet_V2_Weights.IMAGENET1K_V1))
def mobilenet_v2(*, weights: Optional[Union[MobileNet_V2_QuantizedWeights, MobileNet_V2_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableMobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks
    <https://arxiv.org/abs/1801.04381>`_.

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.MobileNet_V2_QuantizedWeights` or :class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.MobileNet_V2_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        quantize (bool, optional): If True, returns a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.QuantizableMobileNetV2``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/mobilenetv2.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.quantization.MobileNet_V2_QuantizedWeights
        :members:
    .. autoclass:: torchvision.models.MobileNet_V2_Weights
        :members:
        :noindex:
    """
    ...
