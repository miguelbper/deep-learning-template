"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple
from .._api import WeightsEnum, register_model
from .._utils import handle_legacy_interface

__all__ = ["MViT", "MViT_V1_B_Weights", "mvit_v1_b", "MViT_V2_S_Weights", "mvit_v2_s"]
@dataclass
class MSBlockConfig:
    num_heads: int
    input_channels: int
    output_channels: int
    kernel_q: List[int]
    kernel_kv: List[int]
    stride_q: List[int]
    stride_kv: List[int]
    ...


class Pool(nn.Module):
    def __init__(self, pool: nn.Module, norm: Optional[nn.Module], activation: Optional[nn.Module] = ..., norm_before_pool: bool = ...) -> None:
        ...

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        ...



class MultiscaleAttention(nn.Module):
    def __init__(self, input_size: List[int], embed_dim: int, output_dim: int, num_heads: int, kernel_q: List[int], kernel_kv: List[int], stride_q: List[int], stride_kv: List[int], residual_pool: bool, residual_with_cls_embed: bool, rel_pos_embed: bool, dropout: float = ..., norm_layer: Callable[..., nn.Module] = ...) -> None:
        ...

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        ...



class MultiscaleBlock(nn.Module):
    def __init__(self, input_size: List[int], cnf: MSBlockConfig, residual_pool: bool, residual_with_cls_embed: bool, rel_pos_embed: bool, proj_after_attn: bool, dropout: float = ..., stochastic_depth_prob: float = ..., norm_layer: Callable[..., nn.Module] = ...) -> None:
        ...

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        ...



class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, spatial_size: Tuple[int, int], temporal_size: int, rel_pos_embed: bool) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...



class MViT(nn.Module):
    def __init__(self, spatial_size: Tuple[int, int], temporal_size: int, block_setting: Sequence[MSBlockConfig], residual_pool: bool, residual_with_cls_embed: bool, rel_pos_embed: bool, proj_after_attn: bool, dropout: float = ..., attention_dropout: float = ..., stochastic_depth_prob: float = ..., num_classes: int = ..., block: Optional[Callable[..., nn.Module]] = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., patch_embed_kernel: Tuple[int, int, int] = ..., patch_embed_stride: Tuple[int, int, int] = ..., patch_embed_padding: Tuple[int, int, int] = ...) -> None:
        """
        MViT main class.

        Args:
            spatial_size (tuple of ints): The spacial size of the input as ``(H, W)``.
            temporal_size (int): The temporal size ``T`` of the input.
            block_setting (sequence of MSBlockConfig): The Network structure.
            residual_pool (bool): If True, use MViTv2 pooling residual connection.
            residual_with_cls_embed (bool): If True, the addition on the residual connection will include
                the class embedding.
            rel_pos_embed (bool): If True, use MViTv2's relative positional embeddings.
            proj_after_attn (bool): If True, apply the projection after the attention.
            dropout (float): Dropout rate. Default: 0.0.
            attention_dropout (float): Attention dropout rate. Default: 0.0.
            stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
            num_classes (int): The number of classes.
            block (callable, optional): Module specifying the layer which consists of the attention and mlp.
            norm_layer (callable, optional): Module specifying the normalization layer to use.
            patch_embed_kernel (tuple of ints): The kernel of the convolution that patchifies the input.
            patch_embed_stride (tuple of ints): The stride of the convolution that patchifies the input.
            patch_embed_padding (tuple of ints): The padding of the convolution that patchifies the input.
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...



class MViT_V1_B_Weights(WeightsEnum):
    KINETICS400_V1 = ...
    DEFAULT = ...


class MViT_V2_S_Weights(WeightsEnum):
    KINETICS400_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", MViT_V1_B_Weights.KINETICS400_V1))
def mvit_v1_b(*, weights: Optional[MViT_V1_B_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MViT:
    """
    Constructs a base MViTV1 architecture from
    `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__.

    .. betastatus:: video module

    Args:
        weights (:class:`~torchvision.models.video.MViT_V1_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MViT_V1_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.MViT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MViT_V1_B_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", MViT_V2_S_Weights.KINETICS400_V1))
def mvit_v2_s(*, weights: Optional[MViT_V2_S_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MViT:
    """Constructs a small MViTV2 architecture from
    `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__ and
    `MViTv2: Improved Multiscale Vision Transformers for Classification
    and Detection <https://arxiv.org/abs/2112.01526>`__.

    .. betastatus:: video module

    Args:
        weights (:class:`~torchvision.models.video.MViT_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MViT_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.MViT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MViT_V2_S_Weights
            :members:
    """
    ...
