"""
This type stub file was generated by pyright.
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import Transform

class Identity(Transform):
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class Lambda(Transform):
    """Apply a user-defined function as a transform.

    This transform does not support torchscript.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """
    _transformed_types = ...
    def __init__(self, lambd: Callable[[Any], Any], *types: Type) -> None:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...

    def extra_repr(self) -> str:
        ...



class LinearTransformation(Transform):
    """Transform a tensor image or video with a square transformation matrix and a mean_vector computed offline.

    This transform does not support PIL Image.
    Given transformation_matrix and mean_vector, will flatten the torch.*Tensor and
    subtract mean_vector from it which is then followed by computing the dot
    product with the transformation matrix and then reshaping the tensor to its
    original shape.

    Applications:
        whitening transformation: Suppose X is a column vector zero-centered data.
        Then compute the data covariance matrix [D x D] with torch.mm(X.t(), X),
        perform SVD on this matrix and pass it as transformation_matrix.

    Args:
        transformation_matrix (Tensor): tensor [D x D], D = C x H x W
        mean_vector (Tensor): tensor [D], D = C x H x W
    """
    _v1_transform_cls = _transforms.LinearTransformation
    _transformed_types = ...
    def __init__(self, transformation_matrix: torch.Tensor, mean_vector: torch.Tensor) -> None:
        ...

    def check_inputs(self, sample: Any) -> Any:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class Normalize(Transform):
    """Normalize a tensor image or video with mean and standard deviation.

    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """
    _v1_transform_cls = _transforms.Normalize
    def __init__(self, mean: Sequence[float], std: Sequence[float], inplace: bool = ...) -> None:
        ...

    def check_inputs(self, sample: Any) -> Any:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class GaussianBlur(Transform):
    """Blurs image with randomly chosen Gaussian blur kernel.

    The convolution will be using reflection padding corresponding to the kernel size, to maintain the input shape.

    If the input is a Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
    """
    _v1_transform_cls = _transforms.GaussianBlur
    def __init__(self, kernel_size: Union[int, Sequence[int]], sigma: Union[int, float, Sequence[float]] = ...) -> None:
        ...

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class GaussianNoise(Transform):
    """Add gaussian noise to images or videos.

    The input tensor is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    Each image or frame in a batch will be transformed independently i.e. the
    noise added to each image will be different.

    The input tensor is also expected to be of float dtype in ``[0, 1]``.
    This transform does not support PIL images.

    Args:
        mean (float): Mean of the sampled normal distribution. Default is 0.
        sigma (float): Standard deviation of the sampled normal distribution. Default is 0.1.
        clip (bool, optional): Whether to clip the values in ``[0, 1]`` after adding noise. Default is True.
    """
    def __init__(self, mean: float = ..., sigma: float = ..., clip=...) -> None:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class ToDtype(Transform):
    """Converts the input to a specific dtype, optionally scaling the values for images or videos.

    .. note::
        ``ToDtype(dtype, scale=True)`` is the recommended replacement for ``ConvertImageDtype(dtype)``.

    Args:
        dtype (``torch.dtype`` or dict of ``TVTensor`` -> ``torch.dtype``): The dtype to convert to.
            If a ``torch.dtype`` is passed, e.g. ``torch.float32``, only images and videos will be converted
            to that dtype: this is for compatibility with :class:`~torchvision.transforms.v2.ConvertImageDtype`.
            A dict can be passed to specify per-tv_tensor conversions, e.g.
            ``dtype={tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others":None}``. The "others"
            key can be used as a catch-all for any other tv_tensor type, and ``None`` means no conversion.
        scale (bool, optional): Whether to scale the values for images or videos. See :ref:`range_and_dtype`.
            Default: ``False``.
    """
    _transformed_types = ...
    def __init__(self, dtype: Union[torch.dtype, Dict[Union[Type, str], Optional[torch.dtype]]], scale: bool = ...) -> None:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class ConvertImageDtype(Transform):
    """[DEPRECATED] Use ``v2.ToDtype(dtype, scale=True)`` instead.

    Convert input image to the given ``dtype`` and scale the values accordingly.

    .. warning::
        Consider using ``ToDtype(dtype, scale=True)`` instead. See :class:`~torchvision.transforms.v2.ToDtype`.

    This function does not support PIL Image.

    Args:
        dtype (torch.dtype): Desired data type of the output

    .. note::

        When converting from a smaller to a larger integer ``dtype`` the maximum values are **not** mapped exactly.
        If converted back and forth, this mismatch has no effect.

    Raises:
        RuntimeError: When trying to cast :class:`torch.float32` to :class:`torch.int32` or :class:`torch.int64` as
            well as for trying to cast :class:`torch.float64` to :class:`torch.int64`. These conversions might lead to
            overflow errors since the floating point ``dtype`` cannot store consecutive integers over the whole range
            of the integer ``dtype``.
    """
    _v1_transform_cls = _transforms.ConvertImageDtype
    def __init__(self, dtype: torch.dtype = ...) -> None:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...



class SanitizeBoundingBoxes(Transform):
    """Remove degenerate/invalid bounding boxes and their corresponding labels and masks.

    This transform removes bounding boxes and their associated labels/masks that:

    - are below a given ``min_size`` or ``min_area``: by default this also removes degenerate boxes that have e.g. X2 <= X1.
    - have any coordinate outside of their corresponding image. You may want to
      call :class:`~torchvision.transforms.v2.ClampBoundingBoxes` first to avoid undesired removals.

    It can also sanitize other tensors like the "iscrowd" or "area" properties from COCO
    (see ``labels_getter`` parameter).

    It is recommended to call it at the end of a pipeline, before passing the
    input to the models. It is critical to call this transform if
    :class:`~torchvision.transforms.v2.RandomIoUCrop` was called.
    If you want to be extra careful, you may call it after all transforms that
    may modify bounding boxes but once at the end should be enough in most
    cases.

    Args:
        min_size (float, optional): The size below which bounding boxes are removed. Default is 1.
        min_area (float, optional): The area below which bounding boxes are removed. Default is 1.
        labels_getter (callable or str or None, optional): indicates how to identify the labels in the input
            (or anything else that needs to be sanitized along with the bounding boxes).
            By default, this will try to find a "labels" key in the input (case-insensitive), if
            the input is a dict or it is a tuple whose second element is a dict.
            This heuristic should work well with a lot of datasets, including the built-in torchvision datasets.

            It can also be a callable that takes the same input as the transform, and returns either:

            - A single tensor (the labels)
            - A tuple/list of tensors, each of which will be subject to the same sanitization as the bounding boxes.
              This is useful to sanitize multiple tensors like the labels, and the "iscrowd" or "area" properties
              from COCO.

            If ``labels_getter`` is None then only bounding boxes are sanitized.
    """
    def __init__(self, min_size: float = ..., min_area: float = ..., labels_getter: Union[Callable[[Any], Any], str, None] = ...) -> None:
        ...

    def forward(self, *inputs: Any) -> Any:
        ...

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ...
