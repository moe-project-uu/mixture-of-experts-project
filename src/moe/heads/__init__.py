# src/moe/heads/__init__.py
from .base import BaseHead
from .dense import DenseHead
from .factory import build_head
__all__ = ["BaseHead", "DenseHead", "build_head"] # add MoE heads here as well
