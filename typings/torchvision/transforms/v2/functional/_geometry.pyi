"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Optional, Tuple, Union
from torchvision import tv_tensors
from torchvision.transforms.functional import InterpolationMode
from ._utils import _FillTypeJIT, _register_five_ten_crop_kernel_internal, _register_kernel_internal

def horizontal_flip(inpt: torch.Tensor) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomHorizontalFlip` for details."""
    ...

@_register_kernel_internal(horizontal_flip, torch.Tensor)
@_register_kernel_internal(horizontal_flip, tv_tensors.Image)
def horizontal_flip_image(image: torch.Tensor) -> torch.Tensor:
    ...

@_register_kernel_internal(horizontal_flip, tv_tensors.Mask)
def horizontal_flip_mask(mask: torch.Tensor) -> torch.Tensor:
    ...

def horizontal_flip_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int]) -> torch.Tensor:
    ...

@_register_kernel_internal(horizontal_flip, tv_tensors.Video)
def horizontal_flip_video(video: torch.Tensor) -> torch.Tensor:
    ...

def vertical_flip(inpt: torch.Tensor) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomVerticalFlip` for details."""
    ...

@_register_kernel_internal(vertical_flip, torch.Tensor)
@_register_kernel_internal(vertical_flip, tv_tensors.Image)
def vertical_flip_image(image: torch.Tensor) -> torch.Tensor:
    ...

@_register_kernel_internal(vertical_flip, tv_tensors.Mask)
def vertical_flip_mask(mask: torch.Tensor) -> torch.Tensor:
    ...

def vertical_flip_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int]) -> torch.Tensor:
    ...

@_register_kernel_internal(vertical_flip, tv_tensors.Video)
def vertical_flip_video(video: torch.Tensor) -> torch.Tensor:
    ...

hflip = ...
vflip = ...
def resize(inpt: torch.Tensor, size: Optional[List[int]], interpolation: Union[InterpolationMode, int] = ..., max_size: Optional[int] = ..., antialias: Optional[bool] = ...) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.Resize` for details."""
    ...

@_register_kernel_internal(resize, torch.Tensor)
@_register_kernel_internal(resize, tv_tensors.Image)
def resize_image(image: torch.Tensor, size: Optional[List[int]], interpolation: Union[InterpolationMode, int] = ..., max_size: Optional[int] = ..., antialias: Optional[bool] = ...) -> torch.Tensor:
    ...

def resize_mask(mask: torch.Tensor, size: Optional[List[int]], max_size: Optional[int] = ...) -> torch.Tensor:
    ...

def resize_bounding_boxes(bounding_boxes: torch.Tensor, canvas_size: Tuple[int, int], size: Optional[List[int]], max_size: Optional[int] = ...) -> Tuple[torch.Tensor, Tuple[int, int]]:
    ...

@_register_kernel_internal(resize, tv_tensors.Video)
def resize_video(video: torch.Tensor, size: Optional[List[int]], interpolation: Union[InterpolationMode, int] = ..., max_size: Optional[int] = ..., antialias: Optional[bool] = ...) -> torch.Tensor:
    ...

def affine(inpt: torch.Tensor, angle: Union[int, float], translate: List[float], scale: float, shear: List[float], interpolation: Union[InterpolationMode, int] = ..., fill: _FillTypeJIT = ..., center: Optional[List[float]] = ...) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomAffine` for details."""
    ...

@_register_kernel_internal(affine, torch.Tensor)
@_register_kernel_internal(affine, tv_tensors.Image)
def affine_image(image: torch.Tensor, angle: Union[int, float], translate: List[float], scale: float, shear: List[float], interpolation: Union[InterpolationMode, int] = ..., fill: _FillTypeJIT = ..., center: Optional[List[float]] = ...) -> torch.Tensor:
    ...

def affine_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int], angle: Union[int, float], translate: List[float], scale: float, shear: List[float], center: Optional[List[float]] = ...) -> torch.Tensor:
    ...

def affine_mask(mask: torch.Tensor, angle: Union[int, float], translate: List[float], scale: float, shear: List[float], fill: _FillTypeJIT = ..., center: Optional[List[float]] = ...) -> torch.Tensor:
    ...

@_register_kernel_internal(affine, tv_tensors.Video)
def affine_video(video: torch.Tensor, angle: Union[int, float], translate: List[float], scale: float, shear: List[float], interpolation: Union[InterpolationMode, int] = ..., fill: _FillTypeJIT = ..., center: Optional[List[float]] = ...) -> torch.Tensor:
    ...

def rotate(inpt: torch.Tensor, angle: float, interpolation: Union[InterpolationMode, int] = ..., expand: bool = ..., center: Optional[List[float]] = ..., fill: _FillTypeJIT = ...) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomRotation` for details."""
    ...

@_register_kernel_internal(rotate, torch.Tensor)
@_register_kernel_internal(rotate, tv_tensors.Image)
def rotate_image(image: torch.Tensor, angle: float, interpolation: Union[InterpolationMode, int] = ..., expand: bool = ..., center: Optional[List[float]] = ..., fill: _FillTypeJIT = ...) -> torch.Tensor:
    ...

def rotate_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int], angle: float, expand: bool = ..., center: Optional[List[float]] = ...) -> Tuple[torch.Tensor, Tuple[int, int]]:
    ...

def rotate_mask(mask: torch.Tensor, angle: float, expand: bool = ..., center: Optional[List[float]] = ..., fill: _FillTypeJIT = ...) -> torch.Tensor:
    ...

@_register_kernel_internal(rotate, tv_tensors.Video)
def rotate_video(video: torch.Tensor, angle: float, interpolation: Union[InterpolationMode, int] = ..., expand: bool = ..., center: Optional[List[float]] = ..., fill: _FillTypeJIT = ...) -> torch.Tensor:
    ...

def pad(inpt: torch.Tensor, padding: List[int], fill: Optional[Union[int, float, List[float]]] = ..., padding_mode: str = ...) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.Pad` for details."""
    ...

@_register_kernel_internal(pad, torch.Tensor)
@_register_kernel_internal(pad, tv_tensors.Image)
def pad_image(image: torch.Tensor, padding: List[int], fill: Optional[Union[int, float, List[float]]] = ..., padding_mode: str = ...) -> torch.Tensor:
    ...

_pad_image_pil = ...
@_register_kernel_internal(pad, tv_tensors.Mask)
def pad_mask(mask: torch.Tensor, padding: List[int], fill: Optional[Union[int, float, List[float]]] = ..., padding_mode: str = ...) -> torch.Tensor:
    ...

def pad_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int], padding: List[int], padding_mode: str = ...) -> Tuple[torch.Tensor, Tuple[int, int]]:
    ...

@_register_kernel_internal(pad, tv_tensors.Video)
def pad_video(video: torch.Tensor, padding: List[int], fill: Optional[Union[int, float, List[float]]] = ..., padding_mode: str = ...) -> torch.Tensor:
    ...

def crop(inpt: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomCrop` for details."""
    ...

@_register_kernel_internal(crop, torch.Tensor)
@_register_kernel_internal(crop, tv_tensors.Image)
def crop_image(image: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    ...

_crop_image_pil = ...
def crop_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, top: int, left: int, height: int, width: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    ...

@_register_kernel_internal(crop, tv_tensors.Mask)
def crop_mask(mask: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    ...

@_register_kernel_internal(crop, tv_tensors.Video)
def crop_video(video: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    ...

def perspective(inpt: torch.Tensor, startpoints: Optional[List[List[int]]], endpoints: Optional[List[List[int]]], interpolation: Union[InterpolationMode, int] = ..., fill: _FillTypeJIT = ..., coefficients: Optional[List[float]] = ...) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomPerspective` for details."""
    ...

@_register_kernel_internal(perspective, torch.Tensor)
@_register_kernel_internal(perspective, tv_tensors.Image)
def perspective_image(image: torch.Tensor, startpoints: Optional[List[List[int]]], endpoints: Optional[List[List[int]]], interpolation: Union[InterpolationMode, int] = ..., fill: _FillTypeJIT = ..., coefficients: Optional[List[float]] = ...) -> torch.Tensor:
    ...

def perspective_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int], startpoints: Optional[List[List[int]]], endpoints: Optional[List[List[int]]], coefficients: Optional[List[float]] = ...) -> torch.Tensor:
    ...

def perspective_mask(mask: torch.Tensor, startpoints: Optional[List[List[int]]], endpoints: Optional[List[List[int]]], fill: _FillTypeJIT = ..., coefficients: Optional[List[float]] = ...) -> torch.Tensor:
    ...

@_register_kernel_internal(perspective, tv_tensors.Video)
def perspective_video(video: torch.Tensor, startpoints: Optional[List[List[int]]], endpoints: Optional[List[List[int]]], interpolation: Union[InterpolationMode, int] = ..., fill: _FillTypeJIT = ..., coefficients: Optional[List[float]] = ...) -> torch.Tensor:
    ...

def elastic(inpt: torch.Tensor, displacement: torch.Tensor, interpolation: Union[InterpolationMode, int] = ..., fill: _FillTypeJIT = ...) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.ElasticTransform` for details."""
    ...

elastic_transform = ...
@_register_kernel_internal(elastic, torch.Tensor)
@_register_kernel_internal(elastic, tv_tensors.Image)
def elastic_image(image: torch.Tensor, displacement: torch.Tensor, interpolation: Union[InterpolationMode, int] = ..., fill: _FillTypeJIT = ...) -> torch.Tensor:
    ...

def elastic_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int], displacement: torch.Tensor) -> torch.Tensor:
    ...

def elastic_mask(mask: torch.Tensor, displacement: torch.Tensor, fill: _FillTypeJIT = ...) -> torch.Tensor:
    ...

@_register_kernel_internal(elastic, tv_tensors.Video)
def elastic_video(video: torch.Tensor, displacement: torch.Tensor, interpolation: Union[InterpolationMode, int] = ..., fill: _FillTypeJIT = ...) -> torch.Tensor:
    ...

def center_crop(inpt: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomCrop` for details."""
    ...

@_register_kernel_internal(center_crop, torch.Tensor)
@_register_kernel_internal(center_crop, tv_tensors.Image)
def center_crop_image(image: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    ...

def center_crop_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int], output_size: List[int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    ...

@_register_kernel_internal(center_crop, tv_tensors.Mask)
def center_crop_mask(mask: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    ...

@_register_kernel_internal(center_crop, tv_tensors.Video)
def center_crop_video(video: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    ...

def resized_crop(inpt: torch.Tensor, top: int, left: int, height: int, width: int, size: List[int], interpolation: Union[InterpolationMode, int] = ..., antialias: Optional[bool] = ...) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomResizedCrop` for details."""
    ...

@_register_kernel_internal(resized_crop, torch.Tensor)
@_register_kernel_internal(resized_crop, tv_tensors.Image)
def resized_crop_image(image: torch.Tensor, top: int, left: int, height: int, width: int, size: List[int], interpolation: Union[InterpolationMode, int] = ..., antialias: Optional[bool] = ...) -> torch.Tensor:
    ...

def resized_crop_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, top: int, left: int, height: int, width: int, size: List[int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    ...

def resized_crop_mask(mask: torch.Tensor, top: int, left: int, height: int, width: int, size: List[int]) -> torch.Tensor:
    ...

@_register_kernel_internal(resized_crop, tv_tensors.Video)
def resized_crop_video(video: torch.Tensor, top: int, left: int, height: int, width: int, size: List[int], interpolation: Union[InterpolationMode, int] = ..., antialias: Optional[bool] = ...) -> torch.Tensor:
    ...

def five_crop(inpt: torch.Tensor, size: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """See :class:`~torchvision.transforms.v2.FiveCrop` for details."""
    ...

@_register_five_ten_crop_kernel_internal(five_crop, torch.Tensor)
@_register_five_ten_crop_kernel_internal(five_crop, tv_tensors.Image)
def five_crop_image(image: torch.Tensor, size: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...

@_register_five_ten_crop_kernel_internal(five_crop, tv_tensors.Video)
def five_crop_video(video: torch.Tensor, size: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...

def ten_crop(inpt: torch.Tensor, size: List[int], vertical_flip: bool = ...) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    """See :class:`~torchvision.transforms.v2.TenCrop` for details."""
    ...

@_register_five_ten_crop_kernel_internal(ten_crop, torch.Tensor)
@_register_five_ten_crop_kernel_internal(ten_crop, tv_tensors.Image)
def ten_crop_image(image: torch.Tensor, size: List[int], vertical_flip: bool = ...) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    ...

@_register_five_ten_crop_kernel_internal(ten_crop, tv_tensors.Video)
def ten_crop_video(video: torch.Tensor, size: List[int], vertical_flip: bool = ...) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    ...
