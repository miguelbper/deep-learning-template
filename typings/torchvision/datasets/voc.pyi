"""
This type stub file was generated by pyright.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from xml.etree.ElementTree import Element as ET_Element
from .vision import VisionDataset

DATASET_YEAR_DICT = ...
class _VOCBase(VisionDataset):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str
    def __init__(self, root: Union[str, Path], year: str = ..., image_set: str = ..., download: bool = ..., transform: Optional[Callable] = ..., target_transform: Optional[Callable] = ..., transforms: Optional[Callable] = ...) -> None:
        ...

    def __len__(self) -> int:
        ...



class VOCSegmentation(_VOCBase):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    _SPLITS_DIR = ...
    _TARGET_DIR = ...
    _TARGET_FILE_EXT = ...
    @property
    def masks(self) -> List[str]:
        ...

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        ...



class VOCDetection(_VOCBase):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    _SPLITS_DIR = ...
    _TARGET_DIR = ...
    _TARGET_FILE_EXT = ...
    @property
    def annotations(self) -> List[str]:
        ...

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        ...

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        ...
