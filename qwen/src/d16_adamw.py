"""Complex AdamW updates for weights stored as canonical D16 tensors."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Literal

import torch

from d16_complex import D16ComplexTensor
from d16_parameter import D16Parameter

IndiffThreshold = float | Literal["auto"]


@dataclass
class _Moments:
    step: int
    first: D16ComplexTensor
    second: D16ComplexTensor


def _accumulation_dtype(value: torch.Tensor) -> torch.dtype:
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def _new_moments(parameter: D16Parameter) -> _Moments:
    dtype = _accumulation_dtype(parameter.x)
    first = D16ComplexTensor.zeros_like(parameter.value(), dtype=dtype)
    second = D16ComplexTensor.zeros_like(parameter.value(), dtype=dtype)
    return _Moments(0, first, second)


class D16AdamW:
    """AdamW operating purely through D16ComplexTensor arithmetic (+, -, *).

    First moments are complex D16ComplexTensor objects accumulated via D16 addition.
    Second moments are real-axis D16ComplexTensor objects accumulated via real_add.
    Weight updates and decay are executed via D16 multiplication and subtraction,
    automatically updating sector masks inside D16ComplexTensor.

    Parameters
    ----------
    indiff_eps : float or "auto", default 0.0
        L1 manifold threshold for snapping small complex weights into State 16:
        Indifferent (0xFFFF) after each update step:
        - float >= 0.0 : manual fixed threshold (0.0 disables pruning).
        - "auto"       : automatic per-tensor adaptive threshold equal to
                         `indiff_rel * sqrt(mean(|w|^2))`.
    indiff_rel : float, default 0.25
        Relative RMS multiplier used when `indiff_eps="auto"`.
    """

    def __init__(
        self,
        parameters: Iterable[D16Parameter],
        lr: float = 1e-3,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        indiff_eps: IndiffThreshold = 0.0,
        indiff_rel: float = 0.25,
        tile_size: int = 1 << 20,
    ) -> None:
        self.parameters = tuple(dict.fromkeys(parameters))
        if not self.parameters:
            raise ValueError("D16AdamW requires at least one D16 parameter")
        if any(not isinstance(p, D16Parameter) for p in self.parameters):
            raise TypeError("D16AdamW only accepts D16Parameter modules")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        self._set_options(lr, betas, eps, weight_decay, indiff_eps, indiff_rel)
        self.tile_size = tile_size
        self._moments: list[_Moments | None] = [None] * len(self.parameters)

    def _set_options(
        self,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        indiff_eps: IndiffThreshold = 0.0,
        indiff_rel: float = 0.25,
    ) -> None:
        if not all(0 <= value < math.inf for value in (lr, eps, weight_decay, indiff_rel)):
            raise ValueError("lr, eps, weight_decay and indiff_rel must be nonnegative and finite")
        if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("Expected two betas in [0, 1)")
        if indiff_eps != "auto":
            if not isinstance(indiff_eps, (int, float)) or not (0.0 <= float(indiff_eps) < math.inf):
                raise ValueError("indiff_eps must be a nonnegative finite float or 'auto'")
            indiff_eps = float(indiff_eps)
        self.lr = lr
        self.betas = tuple(betas)
        self.eps = eps
        self.weight_decay = weight_decay
        self.indiff_eps: IndiffThreshold = indiff_eps
        self.indiff_rel = float(indiff_rel)

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

    def _should_apply_indiff_threshold(self, parameter: D16Parameter) -> bool:
        """Apply 0xFFFF manifold threshold to 2D+ complex weight matrices (skipping 1D gains and biases)."""
        if parameter.real_only or parameter.x.ndim < 2:
            return False
        if self.indiff_eps == "auto":
            return self.indiff_rel > 0.0
        return float(self.indiff_eps) > 0.0

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
        total_norm = (
            torch.linalg.vector_norm(torch.stack(norms))
            if norms
            else first_parameter.new_zeros((), dtype=_accumulation_dtype(first_parameter))
        )
        if not bool(torch.isfinite(total_norm)):
            raise RuntimeError("D16 tangent norm is nonfinite")
        clip = None if max_grad_norm is None else (max_grad_norm / (total_norm + 1e-6)).clamp(max=1)

        for i, parameter in enumerate(self.parameters):
            if not parameter.has_grad():
                continue
            moments = self._moments[i]
            if moments is None:
                moments = self._moments[i] = _new_moments(parameter)
            self._update(parameter, moments, clip)
        return total_norm

    def _update(self, parameter: D16Parameter, moments: _Moments, clip: torch.Tensor | None) -> None:
        beta1, beta2 = self.betas
        moments.step += 1
        step_size = self.lr / (1.0 - beta1 ** moments.step)
        correction2 = 1.0 - beta2 ** moments.step
        decay = 1.0 - self.lr * self.weight_decay
        dtype = moments.first.dtype
        weights = parameter.value()
        clip_val = None if clip is None else clip.to(device=weights.device, dtype=dtype)

        for index in self._tiles(parameter):
            stored_weight = weights[index]
            weight = stored_weight.to(dtype=dtype)
            gradient = parameter.grad_d16(index=index, dtype=dtype)
            if clip_val is not None:
                gradient = gradient * clip_val

            prev_first = moments.first[index]
            prev_second = moments.second[index]

            # Pure D16ComplexTensor arithmetic:
            first = prev_first * beta1 + gradient * (1.0 - beta1)
            second = (prev_second * beta2).real_add(gradient.norm_squared_d16() * (1.0 - beta2))

            second_radius = second.real_signed()
            denom = (second_radius / correction2).sqrt() + self.eps
            delta = first * (step_size / denom)
            updated = weight * decay - delta

            if self._should_apply_indiff_threshold(parameter):
                updated = updated.apply_Manifold_Threshold_L1(
                    self.indiff_eps, rel=self.indiff_rel, dim=None
                )

            parameter.copy_(updated.round_to_dtype(stored_weight.dtype), index=index)
            moments.first.copy_slice_(index, first)
            moments.second.copy_slice_(index, second)

    def state_dict(self) -> dict:
        """Serialize optimizer moments using tensors and primitive metadata."""
        return {
            "format_version": 1,
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "indiff_eps": self.indiff_eps,
            "indiff_rel": self.indiff_rel,
            "moments": [
                None
                if item is None
                else {
                    "step": item.step,
                    "first_x": item.first.x,
                    "first_y": item.first.y,
                    "first_rotation": item.first.rotation,
                    "second_radius": item.second.x,
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
            first_x, first_y, second_r = (
                entry[name].to(device=device, dtype=dtype, copy=True)
                for name in ("first_x", "first_y", "second_radius")
            )
            rotation = entry["first_rotation"].to(device=device, copy=True)
            first = D16ComplexTensor(first_x, first_y, rotation)
            second = D16ComplexTensor.from_real(second_r)
            restored.append(_Moments(entry["step"], first, second))
        self._set_options(
            state_dict["lr"],
            state_dict["betas"],
            state_dict["eps"],
            state_dict["weight_decay"],
            state_dict.get("indiff_eps", 0.0),
            state_dict.get("indiff_rel", 0.25),
        )
        self._moments = restored


__all__ = ["D16AdamW", "IndiffThreshold"]
