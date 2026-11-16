"""Complex Qwen decoder with D16 wedge coordinates throughout the forward pass.

A hidden width of 896 represents 448 complex channels. The model accepts
real-axis D16 token IDs of shape [batch, sequence]. Use encode_token_ids to
convert tokenizer output. A learned complex embedding maps each token to the
hidden width. Hidden states, KV cache, logits and generated IDs stay in D16.
Token selection and cross-entropy use the real-axis component in D16.

Complex parameters store two trainable wedge coordinates and a persistent
sector mask. Embedding lookup and linear kernels consume these triples.
PyTorch supplies kernels for the wedge coordinates and masks. Select
weight_storage="cartesian" to load checkpoints from the earlier weight layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import repeat
import math
from typing import Callable, Literal, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from d16_complex import (
    D16ComplexTensor,
    INV_SQRT2,
    MASK_NAN,
    MASK_ZERO,
    above_bisector_d16,
    d16_reduced_u16,
    even_anchor,
    reverse_bits_u16,
    rotated_sum_d16,
)
from d16_parameter import D16Parameter
from d16_qwen_model import (
    D16QwenConfig as _QwenConfig,
    D16QwenForCausalLM as _QwenControls,
    _apply_sector_steering,
)


# The real line is a subalgebra of D16. Its two nonzero rays have these
# canonical boundary masks. Maxima and probabilities are defined on this line.
_REAL_POSITIVE = 0x8001
_REAL_NEGATIVE = 0x0180

WeightStorage = Literal["d16", "cartesian"]


def _stores_d16(weight_storage: WeightStorage) -> bool:
    if weight_storage not in ("d16", "cartesian"):
        raise ValueError(f"Unknown weight storage: {weight_storage!r}")
    return weight_storage == "d16"


def _real_magnitude(
    magnitude: torch.Tensor,
    negative: torch.Tensor,
    undefined: torch.Tensor,
) -> D16ComplexTensor:
    undefined = undefined | ~torch.isfinite(magnitude)
    zero = magnitude == 0
    negative_mask = torch.full((), _REAL_NEGATIVE, device=magnitude.device, dtype=torch.int32)
    positive_mask = torch.full_like(negative_mask, _REAL_POSITIVE)
    rotation = torch.where(negative, negative_mask, positive_mask)
    rotation = torch.where(zero, MASK_ZERO, rotation)
    rotation = torch.where(undefined, MASK_NAN, rotation)

    # The zero mask uses the positive tangent, even when approached from the
    # negative ray. Keep that tangent for exact cancellations in training.
    x = torch.where(negative & zero, -magnitude, magnitude)
    x = torch.where(undefined, 0.0, x)
    return D16ComplexTensor(x, x.new_zeros(()).expand_as(x), rotation)


def _from_real(value: torch.Tensor) -> D16ComplexTensor:
    """Encode a real boundary value and retain its derivative at zero."""
    negative = value < 0
    magnitude = torch.where(negative, -value, value)
    return _real_magnitude(magnitude, negative, ~torch.isfinite(value))


def encode_token_ids(token_ids: torch.Tensor) -> D16ComplexTensor:
    """Encode a batch of integer IDs on the real axis without rounding vocabulary indices."""
    if token_ids.ndim != 2 or token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Token IDs must be an integer tensor of shape [batch, sequence]")
    return _from_real(token_ids.to(torch.float64))


def _token_addresses(
    tokens: D16ComplexTensor, keys: D16ComplexTensor, *, allow_ignore: bool = False
) -> torch.Tensor:
    if not isinstance(tokens, D16ComplexTensor):
        raise TypeError("Expected D16 token IDs. Use encode_token_ids to encode integer IDs.")
    if tokens.x.ndim != 2 or tokens.x.shape != tokens.y.shape or tokens.x.shape != tokens.rotation.shape:
        raise ValueError("D16 token IDs must have shape [batch, sequence]")
    # The nonnegative real-axis vocabulary is ordered by its wedge radius x.
    # The binary search yields table addresses. Verify the whole D16 key.
    addresses = torch.searchsorted(keys.x, tokens.x.contiguous()).clamp_max(keys.shape[0] - 1)
    found = keys[addresses]
    valid = (tokens.x == found.x) & (tokens.y == found.y) & (tokens.rotation == found.rotation)
    if allow_ignore:
        ignored = (tokens.rotation == _REAL_NEGATIVE) & (tokens.x == 100) & (tokens.y == 0)
        valid = valid | ignored
        addresses = addresses.masked_fill(ignored, -100)
    if not bool(valid.all()):
        raise ValueError("D16 token IDs must be integers on the real axis in the vocabulary range")
    return addresses


def _real_add(a: D16ComplexTensor, b: D16ComplexTensor) -> D16ComplexTensor:
    a_negative = a.rotation == _REAL_NEGATIVE
    b_negative = b.rotation == _REAL_NEGATIVE
    same_ray = a_negative == b_negative
    difference = a.x - b.x
    flip = difference < 0
    magnitude = torch.where(same_ray, a.x + b.x, torch.where(flip, -difference, difference))
    negative = a_negative ^ (~same_ray & flip)
    return _real_magnitude(magnitude, negative, a.is_nan() | b.is_nan())


def _real_negate(value: D16ComplexTensor) -> D16ComplexTensor:
    return _real_magnitude(value.x, value.rotation != _REAL_NEGATIVE, value.is_nan())


def _real_multiply(a: D16ComplexTensor, b: D16ComplexTensor) -> D16ComplexTensor:
    negative = (a.rotation == _REAL_NEGATIVE) ^ (b.rotation == _REAL_NEGATIVE)
    return _real_magnitude(a.x * b.x, negative, a.is_nan() | b.is_nan())


def _real_matmul(a: D16ComplexTensor, b: D16ComplexTensor) -> D16ComplexTensor:
    """Contract wedge magnitudes, grouping products by their resulting ray."""
    dtype = torch.promote_types(a.dtype, b.dtype)
    accum_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

    # A vector is treated as a one-row left operand or a one-column right
    # operand, with these dimensions removed after contraction.
    a_vector = a.x.ndim == 1
    b_vector = b.x.ndim == 1
    ax = a.x.unsqueeze(0) if a_vector else a.x
    am = a.rotation.unsqueeze(0) if a_vector else a.rotation
    bx = b.x.unsqueeze(1) if b_vector else b.x
    bm = b.rotation.unsqueeze(1) if b_vector else b.rotation

    a_negative = am == _REAL_NEGATIVE
    b_negative = bm == _REAL_NEGATIVE
    ax, bx = ax.to(accum_dtype), bx.to(accum_dtype)
    a_pos = torch.where(a_negative, 0.0, ax)
    a_neg = torch.where(a_negative, ax, 0.0)
    b_pos = torch.where(b_negative, 0.0, bx)
    b_neg = torch.where(b_negative, bx, 0.0)

    with torch.autocast(device_type=a.device.type, enabled=False):
        positive = a_pos @ b_pos + a_neg @ b_neg
        negative = a_pos @ b_neg + a_neg @ b_pos
        difference = positive - negative

    # Undefined times zero is still undefined in D16 arithmetic.
    undefined = (am == MASK_NAN).any(dim=-1, keepdim=True) | (bm == MASK_NAN).any(dim=-2, keepdim=True)
    if a_vector:
        difference = difference.squeeze(-2)
        undefined = undefined.squeeze(-2)
    if b_vector:
        difference = difference.squeeze(-1)
        undefined = undefined.squeeze(-1)

    on_negative_ray = difference < 0
    magnitude = torch.where(on_negative_ray, -difference, difference).to(dtype)
    return _real_magnitude(magnitude, on_negative_ray, undefined)


def _shifted_exp(value: D16ComplexTensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    negative = value.rotation == _REAL_NEGATIVE
    dtype = value.dtype
    x = value.x.float() if dtype in (torch.float16, torch.bfloat16) else value.x
    positive_max = torch.where(active & ~negative, x, -float("inf")).amax(dim=-1, keepdim=True)
    negative_max = -torch.where(active & negative, x, float("inf")).amin(dim=-1, keepdim=True)
    maximum = torch.maximum(positive_max, negative_max)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    centered = torch.where(negative, -x - maximum, x - maximum)
    return maximum, torch.where(active, centered, -float("inf")).exp()


def _real_softmax(value: D16ComplexTensor, allowed: Optional[torch.Tensor] = None) -> D16ComplexTensor:
    """Normalize real-axis D16 scores; fully masked rows have zero weight."""
    if value.shape[-1] == 0:
        return value
    undefined = value.is_nan()
    active = ~undefined if allowed is None else allowed & ~undefined
    if allowed is not None:
        undefined = undefined & allowed
    _, terms = _shifted_exp(value, active)
    probabilities = (terms / terms.sum(dim=-1, keepdim=True).clamp_min(1)).to(value.dtype)
    return _real_magnitude(probabilities, torch.zeros_like(undefined), undefined.any(dim=-1, keepdim=True))


def _real_argmax(value: D16ComplexTensor) -> torch.Tensor:
    """Return the address of the largest real-axis value."""
    negative = value.rotation == _REAL_NEGATIVE
    nonnegative = ~negative & ~value.is_nan()
    positive_max, positive_address = value.x.masked_fill(~nonnegative, -torch.inf).max(-1, keepdim=True)
    negative_min, negative_address = value.x.masked_fill(~negative, torch.inf).min(-1, keepdim=True)
    first = torch.where(positive_max.isfinite(), positive_address, negative_address)
    if bool((~positive_max.isfinite() & ~negative_min.isfinite()).any()):
        raise ValueError("Cannot select a token from undefined logits")
    return first


def _real_cross_entropy(scores: D16ComplexTensor, targets: torch.Tensor) -> D16ComplexTensor:
    """Mean negative log-likelihood from real-axis D16 scores and target addresses."""
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.to(torch.float32)
    valid = targets != -100
    addresses = targets.masked_fill(~valid, 0).unsqueeze(-1)
    selected = _map_d16(scores, lambda t: t.gather(-1, addresses).squeeze(-1))
    undefined = scores.is_nan()
    maximum, terms = _shifted_exp(scores, ~undefined)
    distance = _real_add(_from_real(maximum.squeeze(-1)), _real_negate(selected))
    losses = _real_add(distance, _from_real(terms.sum(dim=-1).log()))
    # Sum the wedge magnitudes for each ray separately.
    negative = losses.rotation == _REAL_NEGATIVE
    total = losses.x.masked_fill(~valid | negative, 0).sum() - losses.x.masked_fill(~valid | ~negative, 0).sum()
    mean = (total / valid.sum()).masked_fill((undefined.any(dim=-1) & valid).any(), torch.nan)
    return _from_real(mean)


def _real_sigmoid(value: D16ComplexTensor) -> D16ComplexTensor:
    """Evaluate sigmoid using the ray and a nonpositive exponent."""
    negative = value.rotation == _REAL_NEGATIVE
    exp = (-value.x).exp()
    one = torch.ones_like(exp)
    probability = torch.where(negative, exp, one) / (one + exp)
    return _real_magnitude(probability, torch.zeros_like(negative), value.is_nan())


@lru_cache(maxsize=None)
def _chart_table(device: torch.device) -> torch.Tensor:
    """Pack chart turns into bits 0–2 and reflection into bit 3 for every u16 mask."""
    if device.type != "cpu":
        return _chart_table(torch.device("cpu")).to(device)

    charts = []
    for mask in range(1 << 16):
        anchor = 0 if mask in (MASK_ZERO, MASK_NAN) else even_anchor(mask)
        charts.append((((anchor + 1) // 2) & 7) | ((anchor & 1) << 3))
    return torch.tensor(charts, dtype=torch.uint8, device=device)


def _chart_parameters(rotation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the even anchor at a sector boundary and the identity chart at zero."""
    chart = _chart_table(rotation.device)[rotation.to(torch.long)]
    return (chart & 7).long(), (chart & 8) != 0


def _rotation_basis(turns: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    def sine(angle: torch.Tensor) -> torch.Tensor:
        axis_magnitude = ((angle & 3) == 2).to(dtype=dtype)
        magnitude = torch.where((angle & 1) != 0, INV_SQRT2, axis_magnitude)
        return torch.where(angle < 4, magnitude, -magnitude)

    return sine((turns + 2) & 7), sine(turns)


def _from_chart(
    x: torch.Tensor, y: torch.Tensor, rotation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    turns, reflected = _chart_parameters(rotation)
    cos, sin = _rotation_basis(turns, x.dtype)
    y = torch.where(reflected, -y, y)
    return cos * x - sin * y, sin * x + cos * y


def _into_chart(
    re: torch.Tensor, im: torch.Tensor, rotation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    turns, reflected = _chart_parameters(rotation)
    cos, sin = _rotation_basis(turns, re.dtype)
    y = cos * im - sin * re
    return cos * re + sin * im, torch.where(reflected, -y, y)


class _PackComplex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, re, re_mask, im, im_mask):
        # Orient the two real-axis magnitudes into a first-quadrant chart.
        re_negative = re_mask == _REAL_NEGATIVE
        im_negative = im_mask == _REAL_NEGATIVE
        upper = torch.where(re_negative, 1, 0)
        lower = torch.where(re_negative, 2, 3)
        quadrant = torch.where(im_negative, lower, upper)
        even_quadrant = (quadrant & 1) == 0
        u = torch.where(even_quadrant, re, im)
        v = torch.where(even_quadrant, im, re)
        swapped = v >= u
        wx = torch.where(swapped, v, u)
        wy = torch.where(swapped, u, v)

        diagonal = above_bisector_d16(wx, wy)
        x = torch.where(diagonal, rotated_sum_d16(wx, wy), wx)
        y = torch.where(diagonal, (wx - wy) * INV_SQRT2, wy)
        sector = 4 * quadrant + torch.where(diagonal, torch.where(swapped, 2, 1), torch.where(swapped, 3, 0))
        x, y, rotation = d16_reduced_u16(x, y, sector, (wx == 0) & (wy == 0))
        undefined = (re_mask == MASK_NAN) | (im_mask == MASK_NAN)
        rotation = rotation.masked_fill(undefined, MASK_NAN)
        ctx.save_for_backward(rotation, re_negative, im_negative)
        ctx.mark_non_differentiable(rotation)
        return x.masked_fill(undefined, 0), y.masked_fill(undefined, 0), rotation

    @staticmethod
    def backward(ctx, grad_x, grad_y, _grad_rotation):
        rotation, re_negative, im_negative = ctx.saved_tensors
        # Canonicalization snaps boundary coordinates. Its derivative uses the
        # selected chart without snapping the tangent at the boundary.
        grad_re, grad_im = _from_chart(grad_x, grad_y, rotation)
        undefined = rotation == MASK_NAN
        grad_re = torch.where(re_negative, -grad_re, grad_re).masked_fill(undefined, 0)
        grad_im = torch.where(im_negative, -grad_im, grad_im).masked_fill(undefined, 0)
        return grad_re.sum_to_size(re_negative.shape), None, grad_im.sum_to_size(im_negative.shape), None


def _pack_complex(re: D16ComplexTensor, im: D16ComplexTensor) -> D16ComplexTensor:
    return D16ComplexTensor(*_PackComplex.apply(re.x, re.rotation, im.x, im.rotation))


def _unpack_complex(z: D16ComplexTensor) -> tuple[D16ComplexTensor, D16ComplexTensor]:
    re, im = _from_chart(z.x, z.y, z.rotation)
    undefined = z.is_nan()
    return _from_real(re.masked_fill(undefined, torch.nan)), _from_real(im.masked_fill(undefined, torch.nan))


class _MultiplyComplex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ax, ay, ar, bx, by, br):
        z = D16ComplexTensor(ax, ay, ar) * D16ComplexTensor(bx, by, br)
        ctx.save_for_backward(ax, ay, ar, bx, by, br, z.rotation)
        ctx.mark_non_differentiable(z.rotation)
        return z.x, z.y, z.rotation

    @staticmethod
    def backward(ctx, grad_x, grad_y, _grad_rotation):
        ax, ay, ar, bx, by, br, rotation = ctx.saved_tensors
        grad_re, grad_im = _from_chart(grad_x, grad_y, rotation)
        defined = rotation != MASK_NAN

        # The adjoint of multiplication by the other operand uses its
        # conjugate. Move that adjoint into each input's selected chart.
        def operand_gradient(x, y, mask, other_x, other_y, other_mask):
            re, im = _from_chart(other_x, other_y, other_mask)
            local_x, local_y = _into_chart(
                grad_re * re + grad_im * im, grad_im * re - grad_re * im, mask
            )
            return (
                local_x.masked_fill(~defined, 0).sum_to_size(x.shape),
                local_y.masked_fill(~defined, 0).sum_to_size(y.shape),
            )

        grad_ax = grad_ay = grad_bx = grad_by = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_ax, grad_ay = operand_gradient(ax, ay, ar, bx, by, br)
        if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            grad_bx, grad_by = operand_gradient(bx, by, br, ax, ay, ar)
        return grad_ax, grad_ay, None, grad_bx, grad_by, None


def _complex_multiply(a: D16ComplexTensor, b: D16ComplexTensor) -> D16ComplexTensor:
    return D16ComplexTensor(*_MultiplyComplex.apply(a.x, a.y, a.rotation, b.x, b.y, b.rotation))


class _AddComplex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ax, ay, ar, bx, by, br):
        a_turns, a_reflected = _chart_parameters(ar)
        b_turns, b_reflected = _chart_parameters(br)
        cos, sin = _rotation_basis((b_turns - a_turns) & 7, bx.dtype)
        by_oriented = torch.where(b_reflected, -by, by)
        bx_in_a = cos * bx - sin * by_oriented
        by_in_a = sin * bx + cos * by_oriented
        by_in_a = torch.where(a_reflected, -by_in_a, by_in_a)
        local = D16ComplexTensor.from_xy(ax + bx_in_a, ay + by_in_a)

        # Compose the local result's mask with the left operand's chart.
        relative_mask = torch.where(a_reflected, reverse_bits_u16(local.rotation), local.rotation)
        shift = a_turns * 2
        rotation = ((relative_mask << shift) | (relative_mask >> ((-shift) & 15))) & MASK_ZERO
        rotation = torch.where(ar == MASK_ZERO, br, torch.where(br == MASK_ZERO, ar, rotation))
        undefined = (ar == MASK_NAN) | (br == MASK_NAN)
        rotation = rotation.masked_fill(undefined, MASK_NAN)
        x = torch.where(ar == MASK_ZERO, bx, torch.where(br == MASK_ZERO, ax, local.x))
        y = torch.where(ar == MASK_ZERO, by, torch.where(br == MASK_ZERO, ay, local.y))

        ctx.save_for_backward(ar, br, rotation)
        ctx.mark_non_differentiable(rotation)
        return x.masked_fill(undefined, 0), y.masked_fill(undefined, 0), rotation

    @staticmethod
    def backward(ctx, grad_x, grad_y, _grad_rotation):
        ar, br, rotation = ctx.saved_tensors
        grad_re, grad_im = _from_chart(grad_x, grad_y, rotation)
        undefined = rotation == MASK_NAN

        def operand_gradient(mask):
            local_x, local_y = _into_chart(grad_re, grad_im, mask)
            return (
                local_x.masked_fill(undefined, 0).sum_to_size(mask.shape),
                local_y.masked_fill(undefined, 0).sum_to_size(mask.shape),
            )

        grad_ax = grad_ay = grad_bx = grad_by = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_ax, grad_ay = operand_gradient(ar)
        if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            grad_bx, grad_by = operand_gradient(br)
        return grad_ax, grad_ay, None, grad_bx, grad_by, None


def _complex_add(a: D16ComplexTensor, b: D16ComplexTensor) -> D16ComplexTensor:
    return D16ComplexTensor(*_AddComplex.apply(a.x, a.y, a.rotation, b.x, b.y, b.rotation))


class D16QwenConfig(_QwenConfig):
    """Hidden size counts real coordinates, two per complex channel."""

    @property
    def complex_hidden_size(self) -> int:
        return self.hidden_size // 2

    @property
    def head_dim(self) -> int:
        return self.complex_hidden_size // self.num_attention_heads


D16KeyValue = tuple[D16ComplexTensor, D16ComplexTensor]


def _map_d16(z: D16ComplexTensor, op: Callable[[torch.Tensor], torch.Tensor]) -> D16ComplexTensor:
    return D16ComplexTensor(op(z.x), op(z.y), op(z.rotation))


def _cat_d16(a: D16ComplexTensor, b: D16ComplexTensor, dim: int) -> D16ComplexTensor:
    return D16ComplexTensor(
        torch.cat((a.x, b.x), dim),
        torch.cat((a.y, b.y), dim),
        torch.cat((a.rotation, b.rotation), dim),
    )


def _scale_wedge(z: D16ComplexTensor, scale: torch.Tensor) -> D16ComplexTensor:
    """A positive radial gain preserves the sector of every nonzero value."""
    x = (z.x * scale).to(z.dtype)
    y = (z.y * scale).to(z.dtype)
    undefined = z.is_nan() | ~torch.isfinite(x) | ~torch.isfinite(y)
    mask = torch.where((x == 0) & (y == 0), MASK_ZERO, z.rotation)
    return D16ComplexTensor(
        x.masked_fill(undefined, 0), y.masked_fill(undefined, 0), mask.masked_fill(undefined, MASK_NAN)
    )


def _encode_coordinates(real: torch.Tensor, imag: torch.Tensor) -> D16ComplexTensor:
    return _pack_complex(_from_real(real), _from_real(imag))


def _split_d16(value: D16ComplexTensor, tile_size: int):
    return (D16ComplexTensor(*tile) for tile in zip(
        value.x.split(tile_size), value.y.split(tile_size), value.rotation.split(tile_size),
    ))


class D16Linear(nn.Module):
    """Complex-linear projection of D16 activations by learned D16 weights."""

    # Limit transient wedge storage for large weight matrices, including the LM head.
    _MAX_WEIGHT_TILE = 1 << 20

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        weight_storage: WeightStorage = "d16",
        _shared_weight: Optional[D16Parameter] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gradient_checkpointing = False
        self._native_weights = _stores_d16(weight_storage)
        shape = (out_features, in_features)
        bound = 1 / math.sqrt(2 * in_features)
        if self._native_weights:
            self.weight = (
                _shared_weight if _shared_weight is not None
                else D16Parameter.random_matrix(shape, std=bound / math.sqrt(3), normal=False)
            )
            self.bias = D16Parameter.zeros((out_features,)) if bias else None
        else:
            self.weight_real = nn.Parameter(torch.empty(shape))
            self.weight_imag = nn.Parameter(torch.empty(shape))
            self.bias_real = nn.Parameter(torch.zeros(out_features)) if bias else None
            self.bias_imag = nn.Parameter(torch.zeros(out_features)) if bias else None
            nn.init.uniform_(self.weight_real, -bound, bound)
            nn.init.uniform_(self.weight_imag, -bound, bound)

    def _project_tile(
        self,
        re: D16ComplexTensor,
        im: D16ComplexTensor,
        weight: D16ComplexTensor,
        bias: Optional[D16ComplexTensor],
        dtype: torch.dtype,
        eps: float,
    ) -> D16ComplexTensor:
        wr, wi = _unpack_complex(_map_d16(weight, lambda t: t.transpose(0, 1)).to(re.dtype))
        # The Hamilton product is evaluated with four real-axis D16 contractions.
        out_re = _real_add(_real_matmul(re, wr), _real_negate(_real_matmul(im, wi)))
        out_im = _real_add(_real_matmul(re, wi), _real_matmul(im, wr))
        z = _pack_complex(out_re, out_im)
        if bias is not None:
            z = _complex_add(z, bias.to(re.dtype))
        if eps > 0:
            zr, zi = _unpack_complex(z)
            z = z.to_indifferent_where((zr.x + zi.x) <= eps)
        return z.to(dtype)

    def _project_cartesian_tile(
        self,
        re: D16ComplexTensor,
        im: D16ComplexTensor,
        weight_real: torch.Tensor,
        weight_imag: torch.Tensor,
        bias_real: Optional[torch.Tensor],
        bias_imag: Optional[torch.Tensor],
        dtype: torch.dtype,
        eps: float,
    ) -> D16ComplexTensor:
        weight = _encode_coordinates(weight_real.to(re.dtype), weight_imag.to(re.dtype))
        bias = None if bias_real is None else _encode_coordinates(bias_real.to(re.dtype), bias_imag.to(re.dtype))
        return self._project_tile(re, im, weight, bias, dtype, eps)

    def forward(self, hidden_states: D16ComplexTensor, *, eps: float = 0.0) -> D16ComplexTensor:
        re, im = _unpack_complex(hidden_states)
        weight_dtype = self.weight.x.dtype if self._native_weights else self.weight_real.dtype
        dtype = torch.promote_types(hidden_states.dtype, weight_dtype)
        calc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        re, im = re.to(calc_dtype), im.to(calc_dtype)

        tiles = []
        tile_size = max(1, self._MAX_WEIGHT_TILE // self.in_features)
        # One split per parameter combines the tile gradients in a single backward step.
        if self._native_weights:
            project = self._project_tile
            bias_tiles = _split_d16(self.bias.value(), tile_size) if self.bias is not None else repeat(None)
            parameter_tiles = zip(_split_d16(self.weight.value(), tile_size), bias_tiles)
        else:
            project = self._project_cartesian_tile
            bias_real = self.bias_real.split(tile_size) if self.bias_real is not None else repeat(None)
            bias_imag = self.bias_imag.split(tile_size) if self.bias_imag is not None else repeat(None)
            parameter_tiles = zip(self.weight_real.split(tile_size), self.weight_imag.split(tile_size), bias_real, bias_imag)
        recompute = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for parameter_tile in parameter_tiles:
            args = (re, im, *parameter_tile, dtype, eps)
            if recompute:
                z = checkpoint(project, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                z = project(*args)
            tiles.append(z)

        if len(tiles) == 1:
            return tiles[0]
        return D16ComplexTensor(
            torch.cat([tile.x for tile in tiles], dim=-1),
            torch.cat([tile.y for tile in tiles], dim=-1),
            torch.cat([tile.rotation for tile in tiles], dim=-1),
        )


class D16Embedding(nn.Module):
    """Look up learned complex vectors by exact D16 token keys."""

    def __init__(self, vocab_size: int, hidden_size: int, *, weight_storage: WeightStorage = "d16") -> None:
        super().__init__()
        self.register_buffer("_key_numbers", torch.arange(vocab_size), persistent=False)
        self._native_weights = _stores_d16(weight_storage)
        shape = (vocab_size, hidden_size)
        if self._native_weights:
            self.weight = D16Parameter.random_matrix(shape, std=0.02, normal=True)
        else:
            self.weight_real = nn.Parameter(torch.empty(shape))
            self.weight_imag = nn.Parameter(torch.empty(shape))
            nn.init.normal_(self.weight_real, std=0.02)
            nn.init.normal_(self.weight_imag, std=0.02)

    def token_keys(self) -> D16ComplexTensor:
        return encode_token_ids(self._key_numbers.unsqueeze(0))[0]

    def forward(self, input_ids: D16ComplexTensor) -> D16ComplexTensor:
        addresses = _token_addresses(input_ids, self.token_keys())
        if self._native_weights:
            return self.weight.value()[addresses]
        re = self.weight_real[addresses]
        im = self.weight_imag[addresses]
        return _encode_coordinates(re, im)


class D16RMSNorm(nn.Module):
    """RMS normalization by wedge radius with a learnable positive gain."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, *, weight_storage: WeightStorage = "d16") -> None:
        super().__init__()
        self._native_weights = _stores_d16(weight_storage)
        if self._native_weights:
            self.gain = D16Parameter.zeros((hidden_size,), real_only=True)
        else:
            self.log_gain = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: D16ComplexTensor) -> D16ComplexTensor:
        dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32
        x, y = hidden_states.x.to(dtype), hidden_states.y.to(dtype)
        mean_square = (x.square() + y.square()).mean(dim=-1, keepdim=True)
        undefined = hidden_states.is_nan().any(dim=-1, keepdim=True)
        if self._native_weights:
            # Exponentiate the real-axis D16 log gain using its magnitude and ray.
            log_gain, _ = _unpack_complex(self.gain.value().to(dtype))
            gain = torch.where(log_gain.rotation == _REAL_NEGATIVE, -log_gain.x, log_gain.x).exp()
            gain = gain.masked_fill(log_gain.is_nan(), torch.nan)
        else:
            gain = self.log_gain.to(dtype).exp()
        scale = (mean_square + self.eps).rsqrt() * gain
        return _scale_wedge(hidden_states, scale.masked_fill(undefined, torch.nan))


class D16RotaryEmbedding(nn.Module):
    """One D16 rotation per complex head channel."""

    def __init__(self, head_dim: int, base: float = 1000000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        # Positions remain integer indices and inverse frequencies are analytic.
        self.register_buffer("frequency_indices", torch.arange(head_dim), persistent=False)

    def forward(
        self, hidden_states: D16ComplexTensor, position_ids: torch.Tensor
    ) -> D16ComplexTensor:
        dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32
        inverse_frequencies = self.base ** (-self.frequency_indices.to(dtype) / self.head_dim)
        phase = position_ids.to(dtype).unsqueeze(-1) * inverse_frequencies
        return _encode_coordinates(phase.cos(), phase.sin()).to(hidden_states.dtype)


class D16QwenAttention(nn.Module):
    """Grouped-query attention with a real-axis Hermitian score and complex values."""

    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = config.num_attention_heads // self.num_kv_heads
        self.hidden_size = config.complex_hidden_size
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = D16Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.k_proj = D16Linear(self.hidden_size, kv_size, bias=config.attention_bias)
        self.v_proj = D16Linear(self.hidden_size, kv_size, bias=config.attention_bias)
        self.o_proj = D16Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: D16ComplexTensor,
        position_embeddings: D16ComplexTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[D16KeyValue] = None,
        use_cache: bool = False,
    ) -> tuple[D16ComplexTensor, Optional[D16KeyValue]]:
        batch, length, _ = hidden_states.shape
        head_dim, kv_heads = self.head_dim, self.num_kv_heads
        query = _map_d16(
            self.q_proj(hidden_states),
            lambda t: t.reshape(batch, length, kv_heads, self.num_groups, head_dim).permute(0, 2, 3, 1, 4),
        )
        key = _map_d16(
            self.k_proj(hidden_states), lambda t: t.reshape(batch, length, kv_heads, head_dim).transpose(1, 2)
        )
        value = _map_d16(
            self.v_proj(hidden_states), lambda t: t.reshape(batch, length, kv_heads, head_dim).transpose(1, 2)
        )

        key_phases = _map_d16(position_embeddings, lambda t: t[:, None, :, :])
        query_phases = _map_d16(key_phases, lambda t: t.unsqueeze(2))
        query = _complex_multiply(query, query_phases)
        key = _complex_multiply(key, key_phases)
        if past_key_value is not None:
            key = _cat_d16(past_key_value[0], key, dim=-2)
            value = _cat_d16(past_key_value[1], value, dim=-2)
        present = (key, value) if use_cache else None

        dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        q_re, q_im = _unpack_complex(query.to(dtype))
        k_re, k_im = _unpack_complex(
            _map_d16(key.to(dtype), lambda t: t.transpose(-1, -2).unsqueeze(2))
        )
        scores = _real_add(_real_matmul(q_re, k_re), _real_matmul(q_im, k_im))
        scores = _scale_wedge(scores, scores.x.new_tensor(head_dim ** -0.5))
        allowed = None if attention_mask is None else attention_mask.unsqueeze(2)
        weights = _real_softmax(scores, allowed)

        v_re, v_im = _unpack_complex(_map_d16(value.to(dtype), lambda t: t.unsqueeze(2)))
        output = _pack_complex(_real_matmul(weights, v_re), _real_matmul(weights, v_im))
        output = _map_d16(
            output, lambda t: t.permute(0, 3, 1, 2, 4).reshape(batch, length, self.hidden_size)
        )
        return self.o_proj(output.to(hidden_states.dtype)), present


class D16QwenMLP(nn.Module):
    """Apply 0.5 * z² * sigmoid(Im(z)) and project the full complex result."""

    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        self.up_proj = D16Linear(config.complex_hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = D16Linear(config.intermediate_size, config.complex_hidden_size, bias=config.mlp_bias)
        self.indiff_eps = config.indiff_eps
        self.collect_stats = config.collect_d16_stats
        self.steer_mode = "identity"
        self.steer_shift = 0
        self.steer_channels: Optional[torch.Tensor] = None
        self.register_buffer("state_counts", torch.zeros(18, dtype=torch.long), persistent=False)
        self.total_elements = 0
        self.indifferent_elements = 0

    def reset_stats(self) -> None:
        self.state_counts.zero_()
        self.total_elements = 0
        self.indifferent_elements = 0

    def forward(self, hidden_states: D16ComplexTensor) -> D16ComplexTensor:
        z = self.up_proj(hidden_states, eps=self.indiff_eps)
        if self.steer_mode != "identity":
            channels = self.steer_channels
            if channels is not None:
                channels = channels.to(z.device)
            z = _apply_sector_steering(z, self.steer_mode, self.steer_shift, channels)

        if self.collect_stats:
            with torch.no_grad():
                counts = torch.bincount(z.state().reshape(-1), minlength=18)
                self.state_counts += counts
                self.total_elements += z.x.numel()
                self.indifferent_elements += int(counts[16])

        dtype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
        z = z.to(dtype)
        _, imag = _unpack_complex(z)
        gate = _real_sigmoid(imag)
        squared = _complex_multiply(z, z)
        return self.down_proj(_scale_wedge(squared, 0.5 * gate.x).to(hidden_states.dtype))


class D16QwenDecoderLayer(nn.Module):
    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        self.input_layernorm = D16RMSNorm(config.complex_hidden_size, config.rms_norm_eps)
        self.self_attn = D16QwenAttention(config)
        self.post_attention_layernorm = D16RMSNorm(config.complex_hidden_size, config.rms_norm_eps)
        self.mlp = D16QwenMLP(config)

    def forward(
        self,
        hidden_states: D16ComplexTensor,
        position_embeddings: D16ComplexTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[D16KeyValue] = None,
        use_cache: bool = False,
    ) -> tuple[D16ComplexTensor, Optional[D16KeyValue]]:
        attended, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = _complex_add(hidden_states, attended)
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        return _complex_add(hidden_states, mlp_output), present


class D16QwenModel(nn.Module):
    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        if config.num_attention_heads <= 0 or config.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if config.hidden_size <= 0 or config.hidden_size % (2 * config.num_attention_heads):
            raise ValueError("hidden_size must be positive and divisible by twice num_attention_heads")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if min(config.num_hidden_layers, config.intermediate_size, config.vocab_size) <= 0:
            raise ValueError("num_hidden_layers, intermediate_size and vocab_size must be positive")

        self.config = config
        self.embed_tokens = D16Embedding(config.vocab_size, config.complex_hidden_size)
        self.layers = nn.ModuleList(D16QwenDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = D16RMSNorm(config.complex_hidden_size, config.rms_norm_eps)
        self.rotary_emb = D16RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(
        self,
        input_ids: D16ComplexTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[D16KeyValue]] = None,
        use_cache: bool = False,
    ) -> tuple[D16ComplexTensor, Optional[list[D16KeyValue]]]:
        hidden_states = self.embed_tokens(input_ids)
        batch, length, _ = hidden_states.shape
        device = hidden_states.device
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("Cache must contain one key/value pair per decoder layer")
        past_length = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        total_length = past_length + length
        positions = torch.arange(past_length, total_length, device=device)
        if position_ids is None:
            position_ids = positions.unsqueeze(0)
        phases = self.rotary_emb(hidden_states, position_ids)
        allowed = torch.arange(total_length, device=device) <= positions.unsqueeze(-1)
        allowed = allowed[None, None, :, :]
        if attention_mask is not None:
            if attention_mask.shape != (batch, total_length):
                raise ValueError("attention_mask must cover all current and cached token positions")
            allowed = allowed & attention_mask[:, None, None, :].bool()

        present = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            past = past_key_values[index] if past_key_values is not None else None
            hidden_states, layer_present = layer(
                hidden_states, phases, attention_mask=allowed, past_key_value=past, use_cache=use_cache
            )
            if present is not None and layer_present is not None:
                present.append(layer_present)
        return self.norm(hidden_states), present


@dataclass
class D16CausalLMOutput:
    """Model results in D16. Backpropagate a defined loss through loss.x."""

    logits: D16ComplexTensor
    hidden_states: D16ComplexTensor
    past_key_values: Optional[list[D16KeyValue]] = None
    loss: Optional[D16ComplexTensor] = None


class D16QwenForCausalLM(_QwenControls):
    """D16 language model with a complex readout at each vocabulary position."""

    def __init__(self, config: D16QwenConfig) -> None:
        # The base class supplies the telemetry and sector-steering controls.
        nn.Module.__init__(self)
        self.config = config
        self.model = D16QwenModel(config)
        self.lm_head = D16Linear(config.complex_hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight_real = self.model.embed_tokens.weight_real
            self.lm_head.weight_imag = self.model.embed_tokens.weight_imag

    def gradient_checkpointing_enable(self) -> None:
        """Recompute D16 projection tiles during backward to bound GPU memory."""
        for module in self.modules():
            if isinstance(module, D16Linear):
                module.gradient_checkpointing = True

    def forward(
        self,
        input_ids: D16ComplexTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[D16KeyValue]] = None,
        labels: Optional[D16ComplexTensor] = None,
        use_cache: bool = False,
    ) -> D16CausalLMOutput:
        hidden_states, present = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        output = D16CausalLMOutput(self.lm_head(hidden_states), hidden_states, present)
        if labels is not None:
            targets = _token_addresses(labels, self.model.embed_tokens.token_keys(), allow_ignore=True)
            if targets.shape != input_ids.shape:
                raise ValueError("Labels must have the same shape as the input token IDs")
            scores, _ = _unpack_complex(output.logits[..., :-1, :])
            output.loss = _real_cross_entropy(scores, targets[..., 1:])
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: D16ComplexTensor,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> D16ComplexTensor:
        """Append generated D16 token IDs to the D16 input sequence."""
        keys = self.model.embed_tokens.token_keys()
        _token_addresses(input_ids, keys)
        if input_ids.shape[-1] == 0:
            raise ValueError("Generation requires at least one prompt token")
        was_training = self.training
        self.eval()
        generated = input_ids
        past = None
        try:
            for _ in range(max_new_tokens):
                inputs = generated[:, -1:] if past is not None else generated
                hidden_states, past = self.model(inputs, past_key_values=past, use_cache=use_cache)
                logits = self.lm_head(hidden_states[:, -1])
                scores, _ = _unpack_complex(logits)
                if do_sample:
                    if scores.dtype in (torch.float16, torch.bfloat16):
                        scores = scores.to(torch.float32)
                    inverse_temperature = _from_real(scores.x.new_tensor(1.0 / max(temperature, 1e-5)))
                    probabilities = _real_softmax(_real_multiply(scores, inverse_temperature))
                    addresses = torch.multinomial(probabilities.x, num_samples=1)
                else:
                    addresses = _real_argmax(scores)
                generated = _cat_d16(generated, keys[addresses], dim=-1)
                if eos_token_id is not None and bool((addresses == eos_token_id).all()):
                    break
            return generated
        finally:
            self.train(was_training)


__all__ = [
    "D16ComplexTensor",
    "encode_token_ids",
    "D16QwenConfig",
    "D16CausalLMOutput",
    "D16Linear",
    "D16Embedding",
    "D16RMSNorm",
    "D16RotaryEmbedding",
    "D16QwenAttention",
    "D16QwenMLP",
    "D16QwenDecoderLayer",
    "D16QwenModel",
    "D16QwenForCausalLM",
]
