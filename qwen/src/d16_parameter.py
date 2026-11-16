"""Trainable D16 wedge coordinates and their persistent sector masks."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from d16_complex import D16ComplexTensor


class D16Parameter(nn.Module):
    """Register the coordinates and mask of a canonical D16 value."""

    def __init__(self, value: D16ComplexTensor, *, real_only: bool = False) -> None:
        super().__init__()
        self.x = nn.Parameter(value.x.detach())
        self.y = nn.Parameter(value.y.detach())
        self.register_buffer("rotation", value.rotation.detach().to(torch.int32))
        self.real_only = real_only

    @classmethod
    def zeros(cls, shape: tuple[int, ...], *, real_only: bool = False) -> "D16Parameter":
        return cls(D16ComplexTensor.zeros(shape), real_only=real_only)

    @classmethod
    @torch.no_grad()
    def random_matrix(cls, shape: tuple[int, int], std: float, *, normal: bool) -> "D16Parameter":
        """Sample an isotropic disk or Gaussian directly via D16ComplexTensor.random_polar."""
        parameter = cls.zeros(shape)
        out_features, in_features = shape
        tile_rows = max(1, (1 << 20) // in_features)
        for start in range(0, out_features, tile_rows):
            selection = slice(start, min(start + tile_rows, out_features))
            tile = parameter.x[selection]
            sample_d16 = D16ComplexTensor.random_polar(
                tile.shape,
                std=std,
                normal=normal,
                dtype=tile.dtype,
                device=tile.device,
            )
            parameter.copy_(sample_d16, index=selection)
        return parameter

    def value(self) -> D16ComplexTensor:
        """View the stored value with autograd on its two coordinates."""
        return D16ComplexTensor(self.x, self.y, self.rotation)

    def has_grad(self) -> bool:
        """Return True if either coordinate has accumulated a gradient."""
        return (self.x.grad is not None) or (not self.real_only and self.y.grad is not None)

    @torch.no_grad()
    def grad_d16(self, index=..., *, dtype: Optional[torch.dtype] = None) -> D16ComplexTensor:
        """Return the parameter gradient as a canonical D16ComplexTensor."""
        weight_slice = self.value()[index]
        target_dtype = weight_slice.dtype if dtype is None else dtype
        weight = weight_slice.to(dtype=target_dtype)
        gx = (
            torch.zeros_like(weight.x)
            if self.x.grad is None
            else self.x.grad[index].to(dtype=target_dtype)
        )
        gy = (
            torch.zeros_like(weight.y)
            if (self.real_only or self.y.grad is None)
            else self.y.grad[index].to(dtype=target_dtype)
        )
        return D16ComplexTensor.from_local_tangent(gx, gy, weight)

    @torch.no_grad()
    def copy_(self, value: D16ComplexTensor, index=...) -> "D16Parameter":
        """Keep parameter identities while storing the result of D16 arithmetic."""
        shape = self.x[index].shape
        if any(part.shape != shape for part in (value.x, value.y, value.rotation)):
            raise ValueError("D16 value does not match the parameter slice shape")
        self.x[index] = value.x
        self.y[index] = value.y
        self.rotation[index] = value.rotation
        return self


__all__ = ["D16Parameter"]