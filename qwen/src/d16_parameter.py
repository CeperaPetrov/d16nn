"""Trainable D16 wedge coordinates and their persistent sector masks."""
from __future__ import annotations

import math

import torch
from torch import nn

from d16_complex import D16ComplexTensor, MASK_ZERO, d16_reduced_u16


class D16Parameter(nn.Module):
    """Register the coordinates and mask of a canonical D16 value."""

    def __init__(self, value: D16ComplexTensor, *, real_only: bool = False) -> None:
        super().__init__()
        self.x = nn.Parameter(value.x.detach())
        self.y = nn.Parameter(value.y.detach())
        self.register_buffer("rotation", value.rotation.detach().to(torch.int32))
        self.real_only = real_only

    @classmethod
    def zeros(cls, shape: tuple[int, ...], *, real_only: bool = False) -> D16Parameter:
        x = torch.zeros(shape)
        mask = torch.full(shape, MASK_ZERO, dtype=torch.int32, device=x.device)
        return cls(D16ComplexTensor(x, torch.zeros_like(x), mask), real_only=real_only)

    @classmethod
    @torch.no_grad()
    def random_matrix(cls, shape: tuple[int, int], std: float, *, normal: bool) -> D16Parameter:
        """Sample an isotropic disk or Gaussian directly in D16 polar coordinates."""
        parameter = cls.zeros(shape)
        out_features, in_features = shape
        tile_rows = max(1, (1 << 20) // in_features)
        for start in range(0, out_features, tile_rows):
            selection = slice(start, min(start + tile_rows, out_features))
            tile = parameter.x[selection]
            sample = torch.rand_like(tile)
            if normal:
                radius = (-2 * sample.clamp_min(torch.finfo(sample.dtype).tiny).log()).sqrt() * std
            else:
                radius = sample.sqrt() * (2 * std)
            angle = torch.rand_like(tile) * (math.pi / 8)
            sector = torch.randint(16, tile.shape, device=tile.device)
            x, y, mask = d16_reduced_u16(radius * angle.cos(), radius * angle.sin(), sector, radius == 0)
            parameter.copy_(D16ComplexTensor(x, y, mask), index=selection)
        return parameter

    def value(self) -> D16ComplexTensor:
        """View the stored value with autograd on its two coordinates."""
        return D16ComplexTensor(self.x, self.y, self.rotation)

    @torch.no_grad()
    def copy_(self, value: D16ComplexTensor, index=...) -> D16Parameter:
        """Keep parameter identities while storing the result of D16 arithmetic."""
        shape = self.x[index].shape
        if any(part.shape != shape for part in (value.x, value.y, value.rotation)):
            raise ValueError("D16 value does not match the parameter slice shape")
        self.x[index] = value.x
        self.y[index] = value.y
        self.rotation[index] = value.rotation
        return self
