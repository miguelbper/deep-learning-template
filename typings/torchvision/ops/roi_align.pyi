"""
This type stub file was generated by pyright.
"""

import torch
import torch.fx
from typing import List, Union
from torch import Tensor, nn
from torch.jit.annotations import BroadcastingList2

def lazy_compile(**compile_kwargs): # -> Callable[..., _Wrapped[Callable[..., Any], Any, Callable[..., Any], Any]]:
    """Lazily wrap a function with torch.compile on the first call

    This avoids eagerly importing dynamo.
    """
    ...

def maybe_cast(tensor):
    ...

@torch.fx.wrap
def roi_align(input: Tensor, boxes: Union[Tensor, List[Tensor]], output_size: BroadcastingList2[int], spatial_scale: float = ..., sampling_ratio: int = ..., aligned: bool = ...) -> Tensor:
    """
    Performs Region of Interest (RoI) Align operator with average pooling, as described in Mask R-CNN.

    Args:
        input (Tensor[N, C, H, W]): The input tensor, i.e. a batch with ``N`` elements. Each element
            contains ``C`` feature maps of dimensions ``H x W``.
            If the tensor is quantized, we expect a batch size of ``N == 1``.
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from.
            The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            If a single Tensor is passed, then the first column should
            contain the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``.
            If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i
            in the batch.
        output_size (int or Tuple[int, int]): the size of the output (in bins or pixels) after the pooling
            is performed, as (height, width).
        spatial_scale (float): a scaling factor that maps the box coordinates to
            the input coordinates. For example, if your boxes are defined on the scale
            of a 224x224 image and your input is a 112x112 feature map (resulting from a 0.5x scaling of
            the original image), you'll want to set this to 0.5. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly ``sampling_ratio x sampling_ratio`` sampling points per bin are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ``ceil(roi_width / output_width)``, and likewise for height). Default: -1
        aligned (bool): If False, use the legacy implementation.
            If True, pixel shift the box coordinates it by -0.5 for a better alignment with the two
            neighboring pixel indices. This version is used in Detectron2

    Returns:
        Tensor[K, C, output_size[0], output_size[1]]: The pooled RoIs.
    """
    ...

class RoIAlign(nn.Module):
    """
    See :func:`roi_align`.
    """
    def __init__(self, output_size: BroadcastingList2[int], spatial_scale: float, sampling_ratio: int, aligned: bool = ...) -> None:
        ...

    def forward(self, input: Tensor, rois: Union[Tensor, List[Tensor]]) -> Tensor:
        ...

    def __repr__(self) -> str:
        ...
