"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import Transform
from ._transform import _RandomApplyTransform

class Grayscale(Transform):
    """Convert images or videos to grayscale.

    If the input is a :class:`torch.Tensor`, it is expected
    to have [..., 3 or 1, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image
    """
    _v1_transform_cls = _transforms.Grayscale
    def __init__(self, num_output_channels: int = ...) -> None:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class RandomGrayscale(_RandomApplyTransform):
    """Randomly convert image or videos to grayscale with a probability of p (default 0.1).

    If the input is a :class:`torch.Tensor`, it is expected to have [..., 3 or 1, H, W] shape,
    where ... means an arbitrary number of leading dimensions

    The output has the same number of channels as the input.

    Args:
        p (float): probability that image should be converted to grayscale.
    """
    _v1_transform_cls = _transforms.RandomGrayscale
    def __init__(self, p: float = ...) -> None:
        ...

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class RGB(Transform):
    """Convert images or videos to RGB (if they are already not RGB).

    If the input is a :class:`torch.Tensor`, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions
    """
    def __init__(self) -> None:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class ColorJitter(Transform):
    """Randomly change the brightness, contrast, saturation and hue of an image or video.

    If the input is a :class:`torch.Tensor`, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """
    _v1_transform_cls = _transforms.ColorJitter
    def __init__(self, brightness: Optional[Union[float, Sequence[float]]] = ..., contrast: Optional[Union[float, Sequence[float]]] = ..., saturation: Optional[Union[float, Sequence[float]]] = ..., hue: Optional[Union[float, Sequence[float]]] = ...) -> None:
        ...

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class RandomChannelPermutation(Transform):
    """Randomly permute the channels of an image or video"""
    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class RandomPhotometricDistort(Transform):
    """Randomly distorts the image or video as used in `SSD: Single Shot
    MultiBox Detector <https://arxiv.org/abs/1512.02325>`_.

    This transform relies on :class:`~torchvision.transforms.v2.ColorJitter`
    under the hood to adjust the contrast, saturation, hue, brightness, and also
    randomly permutes channels.

    Args:
        brightness (tuple of float (min, max), optional): How much to jitter brightness.
            brightness_factor is chosen uniformly from [min, max]. Should be non negative numbers.
        contrast (tuple of float (min, max), optional): How much to jitter contrast.
            contrast_factor is chosen uniformly from [min, max]. Should be non-negative numbers.
        saturation (tuple of float (min, max), optional): How much to jitter saturation.
            saturation_factor is chosen uniformly from [min, max]. Should be non negative numbers.
        hue (tuple of float (min, max), optional): How much to jitter hue.
            hue_factor is chosen uniformly from [min, max].  Should have -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
        p (float, optional) probability each distortion operation (contrast, saturation, ...) to be applied.
            Default is 0.5.
    """
    def __init__(self, brightness: Tuple[float, float] = ..., contrast: Tuple[float, float] = ..., saturation: Tuple[float, float] = ..., hue: Tuple[float, float] = ..., p: float = ...) -> None:
        ...

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class RandomEqualize(_RandomApplyTransform):
    """Equalize the histogram of the given image or video with a given probability.

    If the input is a :class:`torch.Tensor`, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".

    Args:
        p (float): probability of the image being equalized. Default value is 0.5
    """
    _v1_transform_cls = _transforms.RandomEqualize
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class RandomInvert(_RandomApplyTransform):
    """Inverts the colors of the given image or video with a given probability.

    If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        p (float): probability of the image being color inverted. Default value is 0.5
    """
    _v1_transform_cls = _transforms.RandomInvert
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class RandomPosterize(_RandomApplyTransform):
    """Posterize the image or video with a given probability by reducing the
    number of bits for each color channel.

    If the input is a :class:`torch.Tensor`, it should be of type torch.uint8,
    and it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        bits (int): number of bits to keep for each channel (0-8)
        p (float): probability of the image being posterized. Default value is 0.5
    """
    _v1_transform_cls = _transforms.RandomPosterize
    def __init__(self, bits: int, p: float = ...) -> None:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class RandomSolarize(_RandomApplyTransform):
    """Solarize the image or video with a given probability by inverting all pixel
    values above a threshold.

    If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        threshold (float): all pixels equal or above this value are inverted.
        p (float): probability of the image being solarized. Default value is 0.5
    """
    _v1_transform_cls = _transforms.RandomSolarize
    def __init__(self, threshold: float, p: float = ...) -> None:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class RandomAutocontrast(_RandomApplyTransform):
    """Autocontrast the pixels of the given image or video with a given probability.

    If the input is a :class:`torch.Tensor`, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        p (float): probability of the image being autocontrasted. Default value is 0.5
    """
    _v1_transform_cls = _transforms.RandomAutocontrast
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class RandomAdjustSharpness(_RandomApplyTransform):
    """Adjust the sharpness of the image or video with a given probability.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness_factor (float):  How much to adjust the sharpness. Can be
            any non-negative number. 0 gives a blurred image, 1 gives the
            original image while 2 increases the sharpness by a factor of 2.
        p (float): probability of the image being sharpened. Default value is 0.5
    """
    _v1_transform_cls = _transforms.RandomAdjustSharpness
    def __init__(self, sharpness_factor: float, p: float = ...) -> None:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...
