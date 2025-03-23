"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Tuple
from torch import Tensor

class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Args:
        tensors (tensor): Tensor containing images.
        image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
    """
    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
        ...

    def to(self, device: torch.device) -> ImageList:
        ...
