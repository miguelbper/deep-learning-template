"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, List, Tuple, Type, Union
from torchvision import tv_tensors

def get_bounding_boxes(flat_inputs: List[Any]) -> tv_tensors.BoundingBoxes:
    """Return the Bounding Boxes in the input.

    Assumes only one ``BoundingBoxes`` object is present.
    """
    ...

def query_chw(flat_inputs: List[Any]) -> Tuple[int, int, int]:
    """Return Channel, Height, and Width."""
    ...

def query_size(flat_inputs: List[Any]) -> Tuple[int, int]:
    """Return Height and Width."""
    ...

def check_type(obj: Any, types_or_checks: Tuple[Union[Type, Callable[[Any], bool]], ...]) -> bool:
    ...

def has_any(flat_inputs: List[Any], *types_or_checks: Union[Type, Callable[[Any], bool]]) -> bool:
    ...

def has_all(flat_inputs: List[Any], *types_or_checks: Union[Type, Callable[[Any], bool]]) -> bool:
    ...
