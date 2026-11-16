"""Complex AdamW updates for weights stored as canonical D16 tensors."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import torch

from d16_complex import D16ComplexTensor, MASK_ZERO, reverse_bits_u16
from d16_parameter import D16Parameter
from d16_qwen_model2 import _chart_parameters, _complex_add, _from_real, _real_add, _scale_wedge


@dataclass
class _Moments:
    step: int
    first: D16ComplexTensor
    # A nonnegative real-axis value has y=0 and its mask follows from its radius.
    second_radius: torch.Tensor


def _accumulation_dtype(value: torch.Tensor) -> torch.dtype:
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def _gradient_in_weight_chart(grad_x, grad_y, weight: D16ComplexTensor) -> D16ComplexTensor:
    """Encode the signed local tangent and compose its mask with the weight chart."""
    local = D16ComplexTensor.from_xy(grad_x, grad_y)
    turns, reflected = _chart_parameters(weight.rotation)
    mask = torch.where(reflected, reverse_bits_u16(local.rotation), local.rotation)
    shift = turns * 2
    rotation = ((mask << shift) | (mask >> ((-shift) & 15))) & MASK_ZERO
    return D16ComplexTensor(local.x, local.y, rotation)


def _round_coordinates(value: D16ComplexTensor, dtype: torch.dtype) -> D16ComplexTensor:
    if value.dtype == dtype:
        return value
    x, y = value.x.to(dtype), value.y.to(dtype)
    undefined = value.is_nan() | ~torch.isfinite(x) | ~torch.isfinite(y)
    return D16ComplexTensor.make(
        x.masked_fill(undefined, 0), y.masked_fill(undefined, 0), value.rotation.masked_fill(undefined, 0),
    )


def _new_moments(parameter: D16Parameter) -> _Moments:
    x, y, rotation = parameter.x, parameter.y, parameter.rotation
    dtype = _accumulation_dtype(x)
    first = D16ComplexTensor(
        torch.zeros_like(x, dtype=dtype), torch.zeros_like(y, dtype=dtype),
        torch.full_like(rotation, MASK_ZERO),
    )
    return _Moments(0, first, torch.zeros_like(x, dtype=dtype))


class D16AdamW:
    """AdamW with a complex first moment and a shared radial second moment.

    Each pair of coordinate gradients describes a tangent in the weight's chart.
    Its squared norm is the second-moment sample, so both complex components use
    the same adaptive gain. The update and decay produce a canonical D16 value.
    """

    def __init__(
        self,
        parameters: Iterable[D16Parameter],
        lr: float = 1e-3,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        tile_size: int = 1 << 20,
    ) -> None:
        self.parameters = tuple(dict.fromkeys(parameters))
        if not self.parameters:
            raise ValueError("D16AdamW requires at least one D16 parameter")
        if any(not isinstance(p, D16Parameter) for p in self.parameters):
            raise TypeError("D16AdamW only accepts D16Parameter modules")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        self._set_options(lr, betas, eps, weight_decay)
        self.tile_size = tile_size
        self._moments: list[_Moments | None] = [None] * len(self.parameters)

    def _set_options(self, lr, betas, eps, weight_decay) -> None:
        if not all(0 <= value < math.inf for value in (lr, eps, weight_decay)):
            raise ValueError("lr, eps and weight_decay must be nonnegative and finite")
        if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("Expected two betas in [0, 1)")
        self.lr, self.betas, self.eps, self.weight_decay = lr, tuple(betas), eps, weight_decay

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            parameter.zero_grad(set_to_none=set_to_none)

    def _tiles(self, parameter: D16Parameter):
        if parameter.x.ndim == 0:
            yield Ellipsis
            return
        row_width = max(1, math.prod(parameter.x.shape[1:]))
        rows = max(1, self.tile_size // row_width)
        for start in range(0, parameter.x.shape[0], rows):
            yield slice(start, start + rows)

    @torch.no_grad()
    def step(self, max_grad_norm: float | None = None) -> torch.Tensor:
        """Apply one update and return the unclipped global tangent norm."""
        if max_grad_norm is not None and not 0 <= max_grad_norm < math.inf:
            raise ValueError("max_grad_norm must be nonnegative and finite")
        first_parameter = self.parameters[0].x
        grads = [(p.x.grad, None if p.real_only else p.y.grad) for p in self.parameters]
        norms = [
            torch.linalg.vector_norm(g, dtype=_accumulation_dtype(g)).to(first_parameter.device)
            for pair in grads for g in pair if g is not None
        ]
        total_norm = (torch.linalg.vector_norm(torch.stack(norms)) if norms else
                      first_parameter.new_zeros((), dtype=_accumulation_dtype(first_parameter)))
        if not bool(torch.isfinite(total_norm)):
            raise RuntimeError("D16 tangent norm is nonfinite")
        clip = None if max_grad_norm is None else (max_grad_norm / (total_norm + 1e-6)).clamp(max=1)

        for i, (parameter, (grad_x, grad_y)) in enumerate(zip(self.parameters, grads)):
            if grad_x is None and grad_y is None:
                continue
            moments = self._moments[i]
            if moments is None:
                moments = self._moments[i] = _new_moments(parameter)
            self._update(parameter, grad_x, grad_y, moments, clip)
        return total_norm

    def _update(self, parameter, grad_x, grad_y, moments, clip) -> None:
        beta1, beta2 = self.betas
        moments.step += 1
        step_size = self.lr / (1 - beta1 ** moments.step)
        correction2 = 1 - beta2 ** moments.step
        decay = 1 - self.lr * self.weight_decay
        dtype = moments.first.dtype
        weights = parameter.value()
        coeff = moments.second_radius.new_tensor([beta1, beta2, 1 - beta1, 1 - beta2, abs(decay)])
        clip = None if clip is None else clip.to(device=weights.device, dtype=dtype)

        for index in self._tiles(parameter):
            stored_weight = weights[index]
            weight = stored_weight.to(dtype=dtype)
            gx = torch.zeros_like(weight.x) if grad_x is None else grad_x[index].to(dtype)
            gy = torch.zeros_like(weight.y) if grad_y is None else grad_y[index].to(dtype)
            gradient = _gradient_in_weight_chart(gx, gy, weight)
            if clip is not None:
                gradient = _scale_wedge(gradient, clip)

            previous = moments.first[index]
            first = _complex_add(_scale_wedge(previous, coeff[0]), _scale_wedge(gradient, coeff[2]))
            magnitude_squared = _from_real(gradient.norm_squared().masked_fill(gradient.is_nan(), torch.nan))
            second = _real_add(
                _scale_wedge(_from_real(moments.second_radius[index]), coeff[1]),
                _scale_wedge(magnitude_squared, coeff[3]),
            )
            second_radius = second.x.masked_fill(second.is_nan(), torch.nan)
            delta = _scale_wedge(first, step_size / ((second_radius / correction2).sqrt() + self.eps))
            decayed = _scale_wedge(weight, coeff[4])
            if decay < 0:
                decayed = decayed.negate()
            updated = _complex_add(decayed, delta.negate())
            parameter.copy_(_round_coordinates(updated, stored_weight.dtype), index=index)

            previous.x.copy_(first.x)
            previous.y.copy_(first.y)
            previous.rotation.copy_(first.rotation)
            moments.second_radius[index].copy_(second_radius)

    def state_dict(self) -> dict:
        """Serialize optimizer moments using tensors and primitive metadata."""
        return {
            "format_version": 1,
            "lr": self.lr, "betas": self.betas, "eps": self.eps, "weight_decay": self.weight_decay,
            "moments": [
                None if item is None else {
                    "step": item.step, "first_x": item.first.x, "first_y": item.first.y,
                    "first_rotation": item.first.rotation, "second_radius": item.second_radius,
                }
                for item in self._moments
            ],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict.get("format_version") != 1 or len(state_dict["moments"]) != len(self.parameters):
            raise ValueError("D16AdamW state does not match these parameters")
        restored = []
        for parameter, entry in zip(self.parameters, state_dict["moments"]):
            if entry is None:
                restored.append(None)
                continue
            names = ("first_x", "first_y", "first_rotation", "second_radius")
            if entry["step"] < 0 or any(entry[name].shape != parameter.x.shape for name in names):
                raise ValueError("D16AdamW moment state has an invalid step or shape")
            if entry["first_rotation"].dtype != parameter.rotation.dtype:
                raise ValueError("D16AdamW sector mask dtype does not match the parameter")
            device, dtype = parameter.x.device, _accumulation_dtype(parameter.x)
            first_x, first_y, second = (
                entry[name].to(device=device, dtype=dtype, copy=True)
                for name in ("first_x", "first_y", "second_radius")
            )
            rotation = entry["first_rotation"].to(device=device, copy=True)
            restored.append(_Moments(entry["step"], D16ComplexTensor(first_x, first_y, rotation), second))
        self._set_options(*(state_dict[name] for name in ("lr", "betas", "eps", "weight_decay")))
        self._moments = restored
