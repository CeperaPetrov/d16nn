"""D16 complex arithmetic: nonnegative coordinates in the 0..22.5 degree wedge.

Coordinates (x, y) live exclusively in the closed 22.5-degree fundamental wedge:
    0 <= y <= (sqrt(2) - 1) * x.

The orientation is a 16-bit sector mask (0x0000..0xFFFF) representing elements
and boundaries of the dihedral group Z_8 |>< Z_2 of order 16:
    - Interior of sector s in 0..15 : one bit (1 << s)
    - Boundary (multiples of 22.5°) : two adjacent bits
    - Zero (State::Indifferent)     : 0xFFFF (all 16 orbit positions active)
    - NaN  (State::Undefined)       : 0x0000

All 16 dihedral operations of Z_8 |>< Z_2 act purely on the 16-bit mask
without modifying (x, y).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Real
from typing import Callable, Optional, Sequence, Tuple, Union

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# =====================================================================
# 1. GEOMETRIC CONSTANTS & STATE DEFINITIONS
# =====================================================================

SQRT2: float = 1.41421356237309504880
INV_SQRT2: float = 0.70710678118654752440
TAN_PI_8: float = 0.41421356237309504880  # sqrt(2) - 1
EPS_BOUND: float = 1e-12

MASK_ZERO: int = 0xFFFF  # State 16: Indifferent (Zero)
MASK_NAN: int = 0x0000   # State 17: Undefined (NaN)

# Canonical boundary masks of the real-line subalgebra in D16
REAL_POSITIVE: int = 0x8001
REAL_NEGATIVE: int = 0x0180

STATE_INDIFFERENT: int = 16
STATE_UNDEFINED: int = 17

STATE_LABELS: Tuple[str, ...] = (
    "Sector_0",
    "Sector_1",
    "Sector_2",
    "Sector_3",
    "Sector_4",
    "Sector_5",
    "Sector_6",
    "Sector_7",
    "Sector_8",
    "Sector_9",
    "Sector_10",
    "Sector_11",
    "Sector_12",
    "Sector_13",
    "Sector_14",
    "Sector_15",
    "Indifferent",
    "Undefined",
)


# =====================================================================
# 2. SCALAR 16-BIT MASK & LOCAL CHART HELPERS
# =====================================================================

def rol(mask: int, steps: int = 1) -> int:
    """16-bit circular shift left."""
    steps &= 15
    mask &= 0xFFFF
    return ((mask << steps) | (mask >> ((16 - steps) & 15))) & 0xFFFF


def ror(mask: int, steps: int = 1) -> int:
    """16-bit circular shift right."""
    return rol(mask, -steps)


def reverse_bits(mask: int) -> int:
    """Reverse all 16 bits of a uint16 mask (reflection across 0°/180° axis)."""
    m = mask & 0xFFFF
    m = ((m & 0xFF00) >> 8) | ((m & 0x00FF) << 8)
    m = ((m & 0xF0F0) >> 4) | ((m & 0x0F0F) << 4)
    m = ((m & 0xCCCC) >> 2) | ((m & 0x3333) << 2)
    m = ((m & 0xAAAA) >> 1) | ((m & 0x5555) << 1)
    return m & 0xFFFF


def bit_index(mask: int) -> int:
    """Index of the highest set bit in a 16-bit mask (0 if mask == 0)."""
    m = mask & 0xFFFF
    return (m.bit_length() - 1) if m != 0 else 0


def is_one_hot(mask: int) -> bool:
    """True iff exactly one bit is set in the 16-bit mask."""
    m = mask & 0xFFFF
    return m != 0 and (m & (m - 1)) == 0


def _mask_is_boundary(mask: int) -> bool:
    """True iff mask consists of two circularly adjacent bits in 0..15."""
    m = mask & 0xFFFF
    return m.bit_count() == 2 and any(
        m == ((1 << k) | (1 << ((k + 1) & 15))) for k in range(16)
    )


def even_anchor(mask: int) -> int:
    """Pick one representative sector bit. Boundaries have a unique even bit."""
    m = mask & 0xFFFF
    if is_one_hot(m):
        return bit_index(m)
    for k in range(0, 16, 2):
        if m & (1 << k):
            return k
    return -1


def _finite_value(value: float) -> bool:
    return math.isfinite(value)


def on_bisector(x: float, y: float) -> bool:
    """True iff (x, y) lies on the 22.5-degree boundary y = (sqrt(2) - 1) * x."""
    scale = x if x > y else y
    if not (scale > 0.0) or not _finite_value(scale):
        return False
    nx, ny = x / scale, y / scale
    return abs((nx - ny) - SQRT2 * ny) <= EPS_BOUND * (nx + ny)


def above_bisector(x: float, y: float) -> bool:
    """True iff (x, y) lies strictly above the 22.5-degree bisector."""
    scale = x if x > y else y
    if not (scale > 0.0) or not _finite_value(scale):
        return False
    nx, ny = x / scale, y / scale
    return (nx - ny) < (SQRT2 * ny - EPS_BOUND * nx)


def rotated_sum(a: float, b: float) -> float:
    """Compute (a + b) / sqrt(2) safely without intermediate overflow."""
    scale = a if a > b else b
    if scale == 0.0:
        return 0.0
    return scale * (((a / scale) + (b / scale)) * INV_SQRT2)


def _group_params(k: int) -> Tuple[int, int]:
    """Decompose sector k in 0..15 into (m, p) where T_k = R_{45}^m o C^p."""
    return ((k + 1) // 2) & 7, k & 1


def _group_from_params(m: int, p: int) -> int:
    """Recover sector k in 0..15 from group parameters (m in 0..7, p in 0..1)."""
    return ((2 * (m & 7)) - (p & 1)) & 15


def _group_compose(k1: int, k2: int) -> int:
    """Composition T_{k1} o T_{k2} in Z_8 |>< Z_2."""
    m1, p1 = _group_params(k1)
    m2, p2 = _group_params(k2)
    return _group_from_params(m1 + (-m2 if p1 else m2), p1 ^ p2)


def _group_inverse(k: int) -> int:
    """Inverse transformation T_k^{-1} in Z_8 |>< Z_2."""
    m, p = _group_params(k)
    return _group_from_params(m if p else -m, p)


def _apply_relative(k: int, x: float, y: float) -> Tuple[float, float]:
    """Apply D16 symmetry T_k in a local reference frame without trig functions."""
    m, p = _group_params(k)
    if p:
        y = -y
    if m & 1:
        x, y = (x - y) * INV_SQRT2, (x + y) * INV_SQRT2
    q = (m >> 1) & 3
    if q == 0:
        return x, y
    if q == 1:
        return -y, x
    if q == 2:
        return -x, -y
    return y, -x


def _fold_local_displacement(u: float, v: float) -> Tuple[float, float, int, bool]:
    """Fold signed local displacements (u, v) into the 0..22.5° wedge and sector k."""
    if not (_finite_value(u) and _finite_value(v)):
        return 0.0, 0.0, 0, False
    if u == 0.0 and v == 0.0:
        return 0.0, 0.0, 0, True

    if u > 0.0 and v >= 0.0:
        q, qu, qv = 0, u, v
    elif u <= 0.0 and v > 0.0:
        q, qu, qv = 1, v, -u
    elif u < 0.0 and v <= 0.0:
        q, qu, qv = 2, -u, -v
    else:
        q, qu, qv = 3, -v, u

    odd_oct = qv >= qu
    wx = qv if odd_oct else qu
    wy = qu if odd_oct else qv

    near_diag = above_bisector(wx, wy)
    if not near_diag:
        x, y = wx, wy
        s16 = 4 * q + (3 if odd_oct else 0)
    else:
        x = rotated_sum(wx, wy)
        y = (wx - wy) * INV_SQRT2
        s16 = 4 * q + (2 if odd_oct else 1)

    return x, y, s16, False


# =====================================================================
# 3. SCALAR D16Complex OBJECT
# =====================================================================

@dataclass(frozen=True, slots=True)
class D16Complex:
    """Scalar D16 complex number with coordinates in the [0°, 22.5°] wedge."""

    x: float = 0.0
    y: float = 0.0
    rotation: int = MASK_ZERO

    SQRT2 = SQRT2
    INV_SQRT2 = INV_SQRT2
    TAN_PI_8 = TAN_PI_8
    EPS_BOUND = EPS_BOUND

    def __post_init__(self) -> None:
        raw_x, raw_y, m = self.x, self.y, self.rotation
        if not isinstance(m, int) or not (0 <= m <= 0xFFFF):
            raise ValueError("rotation must be a 16-bit integer mask in 0x0000..0xFFFF")
        if not isinstance(raw_x, Real) or not isinstance(raw_y, Real):
            raise TypeError("x and y must be real numbers")

        x = float(raw_x)
        y = float(raw_y)
        if not (_finite_value(x) and _finite_value(y)):
            raise ValueError("x and y must be finite and nonnegative; use D16Complex.nan() for NaN")
        if x < 0.0 or y < 0.0:
            raise ValueError("x and y must be finite and nonnegative")

        if m == MASK_NAN:
            if x != 0.0 or y != 0.0:
                raise ValueError("NaN mask requires (0,0)")
            object.__setattr__(self, "x", 0.0)
            object.__setattr__(self, "y", 0.0)
            return

        if x == 0.0 and y == 0.0:
            object.__setattr__(self, "x", 0.0)
            object.__setattr__(self, "y", 0.0)
            object.__setattr__(self, "rotation", MASK_ZERO)
            return

        if m == MASK_ZERO:
            raise ValueError("0xFFFF is reserved for zero")

        if is_one_hot(m):
            k = bit_index(m)
            if y > x:
                x, y = y, x
                k = (k + (13 if (k & 1) else 3)) & 15
            if above_bisector(x, y):
                nx = rotated_sum(x, y)
                ny = (x - y) * INV_SQRT2
                x, y = nx, ny
                k ^= 1
            if y <= EPS_BOUND * x:
                y = 0.0
                neighbor = ((k + 1) & 15) if (k & 1) else ((k + 15) & 15)
                m = (1 << k) | (1 << neighbor)
            elif on_bisector(x, y):
                y = TAN_PI_8 * x
                neighbor = ((k + 15) & 15) if (k & 1) else ((k + 1) & 15)
                m = (1 << k) | (1 << neighbor)
            else:
                m = 1 << k

            object.__setattr__(self, "x", x)
            object.__setattr__(self, "y", y)
            object.__setattr__(self, "rotation", m)
            return

        on_y0 = y <= EPS_BOUND * x
        on_bis = (not on_y0) and on_bisector(x, y)
        if not _mask_is_boundary(m) or y > x or (not on_y0 and not on_bis):
            raise ValueError("invalid boundary representation")

        k = even_anchor(m)
        if k < 0:
            raise ValueError("boundary mask incompatible with coordinates")
        neighbor = ((k + 15) & 15) if on_y0 else ((k + 1) & 15)
        expected = (1 << k) | (1 << neighbor)
        if m != expected:
            raise ValueError("boundary mask incompatible with coordinates")

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", 0.0 if on_y0 else (TAN_PI_8 * x))
        object.__setattr__(self, "rotation", m)

    @staticmethod
    def rol(m: int, n: int = 1) -> int:
        return rol(m, n)

    @staticmethod
    def ror(m: int, n: int = 1) -> int:
        return ror(m, n)

    @staticmethod
    def rotate_left(m: int, n: int = 1) -> int:
        return rol(m, n)

    @staticmethod
    def rotate_right(m: int, n: int = 1) -> int:
        return ror(m, n)

    @staticmethod
    def reverse_bits(m: int) -> int:
        return reverse_bits(m)

    @staticmethod
    def bit_index(m: int) -> int:
        return bit_index(m)

    @staticmethod
    def is_one_hot(m: int) -> bool:
        return is_one_hot(m)

    @staticmethod
    def even_anchor(m: int) -> int:
        return even_anchor(m)

    @staticmethod
    def on_bisector(x: float, y: float) -> bool:
        return on_bisector(x, y)

    @staticmethod
    def above_bisector(x: float, y: float) -> bool:
        return above_bisector(x, y)

    @staticmethod
    def rotated_sum(a: float, b: float) -> float:
        return rotated_sum(a, b)

    @classmethod
    def zero(cls) -> D16Complex:
        return cls(0.0, 0.0, MASK_ZERO)

    @classmethod
    def nan(cls) -> D16Complex:
        return cls(0.0, 0.0, MASK_NAN)

    @classmethod
    def make(cls, x: float, y: float, mask: int) -> D16Complex:
        return cls(x, y, mask)

    @classmethod
    def reduced(cls, x: float, y: float, counter: int) -> D16Complex:
        if not (_finite_value(x) and _finite_value(y)):
            return cls.nan()
        if x == 0.0 and y == 0.0:
            return cls.zero()
        return cls(x, y, counter)

    @classmethod
    def from_xy(cls, logic: float, emotion: float, eps: float = 0.0) -> D16Complex:
        l_val = float(logic)
        e_val = float(emotion)
        if not (_finite_value(l_val) and _finite_value(e_val)):
            return cls.nan()
        if (abs(l_val) + abs(e_val)) <= eps:
            return cls.zero()
        x, y, s16, is_z = _fold_local_displacement(l_val, e_val)
        if is_z:
            return cls.zero()
        return cls.reduced(x, y, 1 << s16)

    def is_zero(self) -> bool:
        return self.rotation == MASK_ZERO

    def is_nan(self) -> bool:
        return self.rotation == MASK_NAN

    @property
    def is_boundary(self) -> bool:
        return self.rotation.bit_count() == 2

    @property
    def norm_squared(self) -> float:
        return self.x * self.x + self.y * self.y

    def state(self) -> int:
        if self.rotation == MASK_ZERO:
            return STATE_INDIFFERENT
        if self.rotation == MASK_NAN:
            return STATE_UNDEFINED
        t = self.rotation & rol(self.rotation, 1)
        return bit_index(t if t != 0 else self.rotation)

    @property
    def state_label(self) -> str:
        return STATE_LABELS[self.state()]

    def to_xy(self) -> Tuple[float, float]:
        if self.is_zero():
            return (0.0, 0.0)
        if self.is_nan():
            return (math.nan, math.nan)
        s = self.state()
        q = s >> 2
        r = s & 3

        is_diag_sector = (r == 1 or r == 2)
        wx = rotated_sum(self.x, self.y) if is_diag_sector else self.x
        wy = ((self.x - self.y) * INV_SQRT2) if is_diag_sector else self.y

        u = wy if r >= 2 else wx
        v = wx if r >= 2 else wy

        if q == 0:
            return (u, v)
        if q == 1:
            return (-v, u)
        if q == 2:
            return (-u, -v)
        return (v, -u)

    def apply_group_op(self, op_id: int) -> D16Complex:
        if self.is_zero() or self.is_nan():
            return self
        op = op_id & 15
        base_m = self.rotation if op < 8 else reverse_bits(self.rotation)
        shift = 2 * (op & 7)
        new_m = rol(base_m, shift)
        obj = object.__new__(D16Complex)
        object.__setattr__(obj, "x", self.x)
        object.__setattr__(obj, "y", self.y)
        object.__setattr__(obj, "rotation", new_m)
        return obj

    def conjugate(self) -> D16Complex:
        return self.apply_group_op(8)

    def negate(self) -> D16Complex:
        return self.apply_group_op(4)

    def multiply(self, other: D16Complex) -> D16Complex:
        if not isinstance(other, D16Complex):
            return NotImplemented
        if self.is_nan() or other.is_nan():
            return self.nan()
        if self.is_zero() or other.is_zero():
            return self.zero()

        ka = even_anchor(self.rotation)
        kb = even_anchor(other.rotation)
        ma, pa = _group_params(ka)
        mb, pb = _group_params(kb)

        ac = self.x * other.x
        bd = self.y * other.y
        ad = self.x * other.y
        bc = self.y * other.x

        counter = rol(0x0001, 2 * ((ma + mb) & 7))

        if pa == pb:
            u = ac - bd
            v = ad + bc
            cross_bisector = above_bisector(u, v)
            if pa == 0:
                if cross_bisector:
                    counter = rol(counter, 1)
            else:
                counter = ror(counter, 1)
                if cross_bisector:
                    counter = ror(counter, 1)

            if cross_bisector:
                out_x = rotated_sum(u, v)
                out_y = (u - v) * INV_SQRT2
            else:
                out_x = u
                out_y = v
        else:
            negative_deviation = (bc < ad) if (pa == 0) else (ad < bc)
            if negative_deviation:
                counter = ror(counter, 1)
            out_x = ac + bd
            out_y = (bc - ad) if (bc >= ad) else (ad - bc)

        if not (_finite_value(out_x) and _finite_value(out_y)):
            return self.nan()
        return self.reduced(out_x, out_y, counter)

    def square(self) -> D16Complex:
        return self.multiply(self)

    def add(self, other: D16Complex) -> D16Complex:
        if not isinstance(other, D16Complex):
            return NotImplemented
        if self.is_nan() or other.is_nan():
            return self.nan()
        if self.is_zero():
            return other
        if other.is_zero():
            return self

        ka = even_anchor(self.rotation)
        kb = even_anchor(other.rotation)
        relative_k = _group_compose(_group_inverse(ka), kb)
        c_rel, d_rel = _apply_relative(relative_k, other.x, other.y)
        u, v = self.x + c_rel, self.y + d_rel
        if not (_finite_value(u) and _finite_value(v)):
            return self.nan()
        x, y, delta_k, is_z = _fold_local_displacement(u, v)
        if is_z:
            return self.zero()
        final_k = _group_compose(ka, delta_k)
        return self.reduced(x, y, 1 << final_k)

    def __mul__(self, other: D16Complex) -> D16Complex:
        return self.multiply(other)

    def __add__(self, other: D16Complex) -> D16Complex:
        return self.add(other)

    def __sub__(self, other: D16Complex) -> D16Complex:
        if not isinstance(other, D16Complex):
            return NotImplemented
        return self.add(other.negate())

    def __neg__(self) -> D16Complex:
        return self.negate()

    def __abs__(self) -> float:
        if self.is_nan():
            return math.nan
        if self.is_zero():
            return 0.0
        return math.hypot(self.x, self.y)

    def __repr__(self) -> str:
        return f"D16Complex(x={self.x!r}, y={self.y!r}, rotation=0x{self.rotation:04X})"


# =====================================================================
# 4. VECTORIZED PYTORCH TENSOR ARITHMETIC & AUTOGRAD (D16ComplexTensor)
# =====================================================================

if _HAS_TORCH:

    def rol_u16(m: torch.Tensor, n: int = 1) -> torch.Tensor:
        """Vectorized 16-bit circular shift left."""
        n &= 15
        return ((m << n) | (m >> ((16 - n) & 15))) & 0xFFFF

    def ror_u16(m: torch.Tensor, n: int = 1) -> torch.Tensor:
        """Vectorized 16-bit circular shift right."""
        return rol_u16(m, (16 - (n & 15)) & 15)

    def reverse_bits_u16(m: torch.Tensor) -> torch.Tensor:
        """Vectorized 16-bit bit reversal."""
        m = ((m & 0xFF00) >> 8) | ((m & 0x00FF) << 8)
        m = ((m & 0xF0F0) >> 4) | ((m & 0x0F0F) << 4)
        m = ((m & 0xCCCC) >> 2) | ((m & 0x3333) << 2)
        m = ((m & 0xAAAA) >> 1) | ((m & 0x5555) << 1)
        return m & 0xFFFF

    def bit_index_u16(m: torch.Tensor) -> torch.Tensor:
        """Vectorized index of highest set bit in uint16 mask."""
        k = torch.zeros_like(m, dtype=torch.long)
        for b in range(1, 16):
            k = torch.where((m >> b) != 0, b, k)
        return k

    def is_one_hot_u16(m: torch.Tensor) -> torch.Tensor:
        """Vectorized one-hot check for uint16 mask."""
        return (m != 0) & ((m & (m - 1)) == 0)

    def even_anchor_u16(m: torch.Tensor) -> torch.Tensor:
        """Vectorized representative bit extraction (unique even bit on boundaries)."""
        one_hot = is_one_hot_u16(m)
        even_k = torch.full_like(m, -1, dtype=torch.long)
        for k in range(14, -1, -2):
            even_k = torch.where((m & (1 << k)) != 0, k, even_k)
        return torch.where(one_hot, bit_index_u16(m), even_k)

    def state_u16(m: torch.Tensor) -> torch.Tensor:
        """Vectorized canonical state in 0..17 from 16-bit mask."""
        t = m & rol_u16(m, 1)
        target = torch.where(t != 0, t, m)
        s = bit_index_u16(target)
        s = torch.where(m == MASK_ZERO, torch.full_like(s, STATE_INDIFFERENT), s)
        s = torch.where(m == MASK_NAN, torch.full_like(s, STATE_UNDEFINED), s)
        return s

    def on_bisector_d16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scale = torch.maximum(x, y)
        valid = (scale > 0.0) & torch.isfinite(scale)
        safe_scale = torch.where(valid, scale, torch.ones_like(scale))
        nx = x / safe_scale
        ny = y / safe_scale
        return valid & (((nx - ny) - SQRT2 * ny).abs() <= EPS_BOUND * (nx + ny))

    def above_bisector_d16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scale = torch.maximum(x, y)
        valid = (scale > 0.0) & torch.isfinite(scale)
        safe_scale = torch.where(valid, scale, torch.ones_like(scale))
        nx = x / safe_scale
        ny = y / safe_scale
        return valid & ((nx - ny) < (SQRT2 * ny - EPS_BOUND * nx))

    def rotated_sum_d16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        scale = torch.maximum(a, b)
        nonzero = scale > 0.0
        safe_scale = torch.where(nonzero, scale, torch.ones_like(scale))
        res = safe_scale * (((a / safe_scale) + (b / safe_scale)) * INV_SQRT2)
        return torch.where(nonzero, res, torch.zeros_like(a))

    def d16_reduced_u16(
        x: torch.Tensor,
        y: torch.Tensor,
        k_counter: torch.Tensor,
        is_zero: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        non_finite = (~torch.isfinite(x)) | (~torch.isfinite(y))
        z_mask = (~non_finite) & (is_zero | ((x == 0.0) & (y == 0.0)))
        k = k_counter & 15
        mask = (1 << k) & 0xFFFF

        is_odd = (k & 1) != 0
        nb_y0 = torch.where(is_odd, (k + 1) & 15, (k + 15) & 15)
        nb_bis = torch.where(is_odd, (k + 15) & 15, (k + 1) & 15)

        active = (~non_finite) & (~z_mask)
        on_y0 = active & (y <= EPS_BOUND * x)
        on_bis = active & (~on_y0) & on_bisector_d16(x, y)

        mask = torch.where(on_y0, mask | (1 << nb_y0), mask)
        mask = torch.where(on_bis, mask | (1 << nb_bis), mask)
        mask = torch.where(z_mask, torch.full_like(mask, MASK_ZERO), mask)
        mask = torch.where(non_finite, torch.zeros_like(mask), mask)

        ox = torch.where(z_mask | non_finite, torch.zeros_like(x), x)
        oy = torch.where(
            z_mask | non_finite | on_y0,
            torch.zeros_like(y),
            torch.where(on_bis, ox * TAN_PI_8, y),
        )
        return ox, oy, mask

    def d16_make_u16(
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized port of D16Complex::make(x, y, mask)."""
        if bool(((~torch.isfinite(x)) | (~torch.isfinite(y)) | (x < 0.0) | (y < 0.0)).any()):
            raise ValueError("x and y must be finite and nonnegative")
        if bool(((mask == MASK_NAN) & ((x != 0.0) | (y != 0.0))).any()):
            raise ValueError("NaN mask requires (0,0)")
        is_zero = (x == 0.0) & (y == 0.0) & (mask != MASK_NAN)
        if bool(((~is_zero) & (mask == MASK_ZERO)).any()):
            raise ValueError("0xFFFF is reserved for zero")

        k = even_anchor_u16(mask)
        swap_xy = y > x
        nx = torch.where(swap_xy, y, x)
        ny = torch.where(swap_xy, x, y)
        delta_k = torch.where((k & 1) != 0, 13, 3)
        k = torch.where(swap_xy, (k + delta_k) & 15, k)

        fold_bis = above_bisector_d16(nx, ny)
        bx = rotated_sum_d16(nx, ny)
        by = (nx - ny) * INV_SQRT2
        fx = torch.where(fold_bis, bx, nx)
        fy = torch.where(fold_bis, by, ny)
        k = torch.where(fold_bis, k ^ 1, k)

        rx, ry, rmask = d16_reduced_u16(fx, fy, k, is_zero)
        nan_mask = mask == MASK_NAN
        rx = torch.where(nan_mask, torch.zeros_like(rx), rx)
        ry = torch.where(nan_mask, torch.zeros_like(ry), ry)
        rmask = torch.where(nan_mask, torch.zeros_like(rmask), rmask)
        return rx, ry, rmask

    def d16_from_xy_u16(
        logic: torch.Tensor,
        emotion: torch.Tensor,
        eps: Union[float, torch.Tensor] = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized encoding of (logic, emotion) into (x, y) in [0°, 22.5°] and u16 mask."""
        is_zero = (logic.abs() + emotion.abs()) <= eps

        q0 = (logic > 0.0) & (emotion >= 0.0)
        q1 = (logic <= 0.0) & (emotion > 0.0)
        q2 = (logic < 0.0) & (emotion <= 0.0)

        q = torch.where(q0, 0, torch.where(q1, 1, torch.where(q2, 2, 3)))
        u = torch.where(q0, logic, torch.where(q1, emotion, torch.where(q2, -logic, -emotion)))
        v = torch.where(q0, emotion, torch.where(q1, -logic, torch.where(q2, -emotion, logic)))

        odd_oct = v >= u
        wx = torch.where(odd_oct, v, u)
        wy = torch.where(odd_oct, u, v)

        near_diag = above_bisector_d16(wx, wy)
        x = torch.where(near_diag, rotated_sum_d16(wx, wy), wx)
        y = torch.where(near_diag, (wx - wy) * INV_SQRT2, wy)

        s_axis = 4 * q + torch.where(odd_oct, 3, 0)
        s_diag = 4 * q + torch.where(odd_oct, 2, 1)
        s16 = torch.where(near_diag, s_diag, s_axis)

        return d16_reduced_u16(x, y, s16, is_zero)

    def d16_to_xy_u16(
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized decoding from (x, y, mask_u16) back to Cartesian (Re, Im)."""
        s = state_u16(mask)
        is_zero = (mask == MASK_ZERO) | (s == STATE_INDIFFERENT)
        is_nan = (mask == MASK_NAN) | (s == STATE_UNDEFINED)

        q = s >> 2
        r = s & 3

        is_diag = (r == 1) | (r == 2)
        wx = torch.where(is_diag, rotated_sum_d16(x, y), x)
        wy = torch.where(is_diag, (x - y) * INV_SQRT2, y)

        u = torch.where(r >= 2, wy, wx)
        v = torch.where(r >= 2, wx, wy)

        re = torch.where(q == 0, u, torch.where(q == 1, -v, torch.where(q == 2, -u, v)))
        im = torch.where(q == 0, v, torch.where(q == 1, u, torch.where(q == 2, -v, -u)))

        re = torch.where(is_zero, torch.zeros_like(re), re)
        im = torch.where(is_zero, torch.zeros_like(im), im)
        re = torch.where(is_nan, torch.full_like(re, float("nan")), re)
        im = torch.where(is_nan, torch.full_like(im, float("nan")), im)
        return re, im

    def d16_multiply_u16(
        ax: torch.Tensor,
        ay: torch.Tensor,
        am: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
        bm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized local multiplication inside the [0°, 22.5°] fundamental wedge."""
        is_nan = (am == MASK_NAN) | (bm == MASK_NAN)
        is_zero = (~is_nan) & ((am == MASK_ZERO) | (bm == MASK_ZERO))
        ka = even_anchor_u16(am)
        kb = even_anchor_u16(bm)

        ma = ((ka + 1) // 2) & 7
        mb = ((kb + 1) // 2) & 7
        pa = ka & 1
        pb = kb & 1

        ac = ax * bx
        bd = ay * by
        ad = ax * by
        bc = ay * bx

        base_k = (2 * ((ma + mb) & 7)) & 15
        same_p = pa == pb

        u = ac - bd
        v = ad + bc
        cross_bis = above_bisector_d16(u, v)
        k_same_p0 = torch.where(cross_bis, (base_k + 1) & 15, base_k)
        k_same_p1 = torch.where(cross_bis, (base_k + 14) & 15, (base_k + 15) & 15)
        k_same = torch.where(pa == 0, k_same_p0, k_same_p1)
        ox_same = torch.where(cross_bis, rotated_sum_d16(u, v), u)
        oy_same = torch.where(cross_bis, (u - v) * INV_SQRT2, v)

        neg_dev = torch.where(pa == 0, bc < ad, ad < bc)
        k_diff = torch.where(neg_dev, (base_k + 15) & 15, base_k)
        ox_diff = ac + bd
        oy_diff = (bc - ad).abs()

        out_x = torch.where(same_p, ox_same, ox_diff)
        out_y = torch.where(same_p, oy_same, oy_diff)
        out_k = torch.where(same_p, k_same, k_diff)
        rx, ry, rm = d16_reduced_u16(out_x, out_y, out_k, is_zero)
        rx = torch.where(is_nan, torch.zeros_like(rx), rx)
        ry = torch.where(is_nan, torch.zeros_like(ry), ry)
        rm = torch.where(is_nan, torch.zeros_like(rm), rm)
        return rx, ry, rm

    def apply_group_op_u16(m: torch.Tensor, op_id: int) -> torch.Tensor:
        """Apply Z_8 |>< Z_2 element op_id in 0..15 purely to 16-bit mask tensor."""
        op = op_id & 15
        special = (m == MASK_ZERO) | (m == MASK_NAN)
        base_m = m if op < 8 else reverse_bits_u16(m)
        out = rol_u16(base_m, 2 * (op & 7))
        return torch.where(special, m, out)

    def apply_all_16_d16_ops_u16(m: torch.Tensor) -> torch.Tensor:
        """Return stack [..., 16] of masks under all 16 dihedral operations."""
        special = (m == MASK_ZERO) | (m == MASK_NAN)
        rev = reverse_bits_u16(m)
        raw = torch.stack([
            m,                     # 0:  (+L, +E)
            rol_u16(m, 8),         # 1:  (-L, -E)
            rev,                   # 2:  (+L, -E)
            rol_u16(rev, 8),       # 3:  (-L, +E)
            rol_u16(m, 4),         # 4:  (-E, +L)
            ror_u16(m, 4),         # 5:  (+E, -L)
            rol_u16(rev, 4),       # 6:  (+E, +L)
            rol_u16(rev, 12),      # 7:  (-E, -L)
            rol_u16(m, 2),         # 8:  ( L-E,  L+E)/sqrt(2)
            rol_u16(m, 6),         # 9:  (-L-E,  L-E)/sqrt(2)
            rol_u16(m, 10),        # 10: (-L+E, -L-E)/sqrt(2)
            rol_u16(m, 14),        # 11: ( L+E, -L+E)/sqrt(2)
            rol_u16(rev, 2),       # 12: ( L+E,  L-E)/sqrt(2)
            rol_u16(rev, 6),       # 13: (-L+E,  L+E)/sqrt(2)
            rol_u16(rev, 10),      # 14: (-L-E, -L+E)/sqrt(2)
            rol_u16(rev, 14),      # 15: ( L-E, -L-E)/sqrt(2)
        ], dim=-1)
        return torch.where(special.unsqueeze(-1), m.unsqueeze(-1), raw)

    # =====================================================================
    # 5. ENCAPSULATED LOCAL CHARTS, AUTOGRAD & REAL-AXIS SUBALGEBRA
    # =====================================================================

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

    def _chart_parameters(rotation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract 45-degree turn index (0..7) and reflection flag from a 16-bit mask."""
        chart = _chart_table(rotation.device)[rotation.to(torch.long)]
        return (chart & 7).long(), (chart & 8) != 0

    def _rotation_basis(turns: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        def sine(angle: torch.Tensor) -> torch.Tensor:
            axis_magnitude = ((angle & 3) == 2).to(dtype=dtype)
            magnitude = torch.where((angle & 1) != 0, INV_SQRT2, axis_magnitude)
            return torch.where(angle < 4, magnitude, -magnitude)

        return sine((turns + 2) & 7), sine(turns)

    def _from_chart(
        x: torch.Tensor, y: torch.Tensor, rotation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        turns, reflected = _chart_parameters(rotation)
        cos, sin = _rotation_basis(turns, x.dtype)
        y_oriented = torch.where(reflected, -y, y)
        return cos * x - sin * y_oriented, sin * x + cos * y_oriented

    def _into_chart(
        re: torch.Tensor, im: torch.Tensor, rotation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        turns, reflected = _chart_parameters(rotation)
        cos, sin = _rotation_basis(turns, re.dtype)
        y = cos * im - sin * re
        return cos * re + sin * im, torch.where(reflected, -y, y)

    def _compose_chart_mask(
        local_mask: torch.Tensor, turns: torch.Tensor, reflected: torch.Tensor
    ) -> torch.Tensor:
        """Compose a local mask with a chart's reflection and 45-degree turns."""
        relative_mask = torch.where(reflected, reverse_bits_u16(local_mask), local_mask)
        shift = turns * 2
        return ((relative_mask << shift) | (relative_mask >> ((-shift) & 15))) & MASK_ZERO

    class _PackComplex(torch.autograd.Function):
        @staticmethod
        def forward(ctx, re, re_mask, im, im_mask):
            re_negative = re_mask == REAL_NEGATIVE
            im_negative = im_mask == REAL_NEGATIVE
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
            sector = 4 * quadrant + torch.where(
                diagonal, torch.where(swapped, 2, 1), torch.where(swapped, 3, 0)
            )
            x, y, rotation = d16_reduced_u16(x, y, sector, (wx == 0) & (wy == 0))
            undefined = (re_mask == MASK_NAN) | (im_mask == MASK_NAN)
            rotation = rotation.masked_fill(undefined, MASK_NAN)
            ctx.save_for_backward(rotation, re_negative, im_negative)
            ctx.mark_non_differentiable(rotation)
            return x.masked_fill(undefined, 0), y.masked_fill(undefined, 0), rotation

        @staticmethod
        def backward(ctx, grad_x, grad_y, _grad_rotation):
            rotation, re_negative, im_negative = ctx.saved_tensors
            grad_re, grad_im = _from_chart(grad_x, grad_y, rotation)
            undefined = rotation == MASK_NAN
            grad_re = torch.where(re_negative, -grad_re, grad_re).masked_fill(undefined, 0)
            grad_im = torch.where(im_negative, -grad_im, grad_im).masked_fill(undefined, 0)
            return grad_re.sum_to_size(re_negative.shape), None, grad_im.sum_to_size(im_negative.shape), None

    class _MultiplyComplex(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ax, ay, ar, bx, by, br):
            zx, zy, zr = d16_multiply_u16(ax, ay, ar, bx, by, br)
            ctx.save_for_backward(ax, ay, ar, bx, by, br, zr)
            ctx.mark_non_differentiable(zr)
            return zx, zy, zr

        @staticmethod
        def backward(ctx, grad_x, grad_y, _grad_rotation):
            ax, ay, ar, bx, by, br, rotation = ctx.saved_tensors
            grad_re, grad_im = _from_chart(grad_x, grad_y, rotation)
            defined = rotation != MASK_NAN

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
            lx, ly, lrot = d16_from_xy_u16(ax + bx_in_a, ay + by_in_a)

            rotation = _compose_chart_mask(lrot, a_turns, a_reflected)
            rotation = torch.where(ar == MASK_ZERO, br, torch.where(br == MASK_ZERO, ar, rotation))
            undefined = (ar == MASK_NAN) | (br == MASK_NAN)
            rotation = rotation.masked_fill(undefined, MASK_NAN)
            x = torch.where(ar == MASK_ZERO, bx, torch.where(br == MASK_ZERO, ax, lx))
            y = torch.where(ar == MASK_ZERO, by, torch.where(br == MASK_ZERO, ay, ly))

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

    def _real_magnitude(
        magnitude: torch.Tensor,
        negative: torch.Tensor,
        undefined: torch.Tensor,
    ) -> "D16ComplexTensor":
        undefined = undefined | ~torch.isfinite(magnitude)
        zero = magnitude == 0
        negative_mask = torch.full((), REAL_NEGATIVE, device=magnitude.device, dtype=torch.int32)
        positive_mask = torch.full_like(negative_mask, REAL_POSITIVE)
        rotation = torch.where(negative, negative_mask, positive_mask)
        rotation = torch.where(zero, MASK_ZERO, rotation)
        rotation = torch.where(undefined, MASK_NAN, rotation)
        x = torch.where(negative & zero, -magnitude, magnitude)
        x = torch.where(undefined, 0.0, x)
        return D16ComplexTensor(x, x.new_zeros(()).expand_as(x), rotation)

    def _from_real(value: torch.Tensor) -> "D16ComplexTensor":
        negative = value < 0
        magnitude = torch.where(negative, -value, value)
        return _real_magnitude(magnitude, negative, ~torch.isfinite(value))

    def _real_add(a: "D16ComplexTensor", b: "D16ComplexTensor") -> "D16ComplexTensor":
        a_negative = a.rotation == REAL_NEGATIVE
        b_negative = b.rotation == REAL_NEGATIVE
        same_ray = a_negative == b_negative
        difference = a.x - b.x
        flip = difference < 0
        magnitude = torch.where(same_ray, a.x + b.x, torch.where(flip, -difference, difference))
        negative = a_negative ^ (~same_ray & flip)
        return _real_magnitude(magnitude, negative, a.is_nan() | b.is_nan())

    def _real_negate(value: "D16ComplexTensor") -> "D16ComplexTensor":
        return _real_magnitude(value.x, value.rotation != REAL_NEGATIVE, value.is_nan())

    def _real_multiply(a: "D16ComplexTensor", b: "D16ComplexTensor") -> "D16ComplexTensor":
        negative = (a.rotation == REAL_NEGATIVE) ^ (b.rotation == REAL_NEGATIVE)
        return _real_magnitude(a.x * b.x, negative, a.is_nan() | b.is_nan())

    def _real_matmul(a: "D16ComplexTensor", b: "D16ComplexTensor") -> "D16ComplexTensor":
        dtype = torch.promote_types(a.dtype, b.dtype)
        accum_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        a_vector = a.x.ndim == 1
        b_vector = b.x.ndim == 1
        ax = a.x.unsqueeze(0) if a_vector else a.x
        am = a.rotation.unsqueeze(0) if a_vector else a.rotation
        bx = b.x.unsqueeze(1) if b_vector else b.x
        bm = b.rotation.unsqueeze(1) if b_vector else b.rotation

        a_negative = am == REAL_NEGATIVE
        b_negative = bm == REAL_NEGATIVE
        ax, bx = ax.to(accum_dtype), bx.to(accum_dtype)
        a_pos = torch.where(a_negative, 0.0, ax)
        a_neg = torch.where(a_negative, ax, 0.0)
        b_pos = torch.where(b_negative, 0.0, bx)
        b_neg = torch.where(b_negative, bx, 0.0)

        with torch.autocast(device_type=a.device.type, enabled=False):
            positive = a_pos @ b_pos + a_neg @ b_neg
            negative = a_pos @ b_neg + a_neg @ b_pos
            difference = positive - negative

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

    def _shifted_exp(value: "D16ComplexTensor", active: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        negative = value.rotation == REAL_NEGATIVE
        dtype = value.dtype
        x = value.x.float() if dtype in (torch.float16, torch.bfloat16) else value.x
        positive_max = torch.where(active & ~negative, x, -float("inf")).amax(dim=-1, keepdim=True)
        negative_max = -torch.where(active & negative, x, float("inf")).amin(dim=-1, keepdim=True)
        maximum = torch.maximum(positive_max, negative_max)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
        centered = torch.where(negative, -x - maximum, x - maximum)
        return maximum, torch.where(active, centered, -float("inf")).exp()

    def _scale_wedge(z: "D16ComplexTensor", scale: Union[float, int, torch.Tensor]) -> "D16ComplexTensor":
        """Scale a D16ComplexTensor by a real scalar or tensor (handling negative scales via 180° shift)."""
        if not isinstance(scale, torch.Tensor):
            scale_t = z.x.new_tensor(scale)
        else:
            scale_t = scale.to(device=z.device)
        neg = scale_t < 0
        mag = torch.where(neg, -scale_t, scale_t)
        x = (z.x * mag).to(z.dtype)
        y = (z.y * mag).to(z.dtype)
        undefined = z.is_nan() | ~torch.isfinite(scale_t) | ~torch.isfinite(x) | ~torch.isfinite(y)
        rot = torch.where(neg, apply_group_op_u16(z.rotation, 4), z.rotation)
        mask = torch.where((x == 0) & (y == 0), MASK_ZERO, rot)
        return D16ComplexTensor(
            x.masked_fill(undefined, 0),
            y.masked_fill(undefined, 0),
            mask.masked_fill(undefined, MASK_NAN),
        )

    class D16ComplexTensor:
        """Vectorized PyTorch container of D16Complex objects with full autograd and operator encapsulation."""

        __slots__ = ("x", "y", "rotation")

        def __init__(self, x: torch.Tensor, y: torch.Tensor, rotation: torch.Tensor) -> None:
            self.x = x
            self.y = y
            self.rotation = rotation

        # --- Constructors & Encodings ---

        @classmethod
        def zeros(
            cls,
            shape: Union[int, Sequence[int], torch.Size],
            *,
            dtype: torch.dtype = torch.float32,
            device: Optional[Union[str, torch.device]] = None,
        ) -> "D16ComplexTensor":
            if isinstance(shape, int):
                shape = (shape,)
            x = torch.zeros(shape, dtype=dtype, device=device)
            mask = torch.full(shape, MASK_ZERO, dtype=torch.int32, device=x.device)
            return cls(x, torch.zeros_like(x), mask)

        @classmethod
        def zeros_like(
            cls,
            other: Union["D16ComplexTensor", torch.Tensor],
            *,
            dtype: Optional[torch.dtype] = None,
            device: Optional[Union[str, torch.device]] = None,
        ) -> "D16ComplexTensor":
            ref = other.x if isinstance(other, D16ComplexTensor) else other
            target_dtype = ref.dtype if dtype is None else dtype
            target_device = ref.device if device is None else device
            x = torch.zeros_like(ref, dtype=target_dtype, device=target_device)
            rot_dtype = other.rotation.dtype if isinstance(other, D16ComplexTensor) else torch.int32
            mask = torch.full_like(ref, MASK_ZERO, dtype=rot_dtype, device=target_device)
            return cls(x, torch.zeros_like(x), mask)

        @classmethod
        @torch.no_grad()
        def random_polar(
            cls,
            shape: Sequence[int],
            std: float = 1.0,
            *,
            normal: bool = False,
            dtype: torch.dtype = torch.float32,
            device: Optional[Union[str, torch.device]] = None,
        ) -> "D16ComplexTensor":
            """Sample an isotropic disk or Gaussian directly in D16 polar coordinates."""
            sample = torch.rand(shape, dtype=dtype, device=device)
            if normal:
                radius = (-2 * sample.clamp_min(torch.finfo(sample.dtype).tiny).log()).sqrt() * std
            else:
                radius = sample.sqrt() * (2 * std)
            angle = torch.rand_like(sample) * (math.pi / 8)
            sector = torch.randint(16, sample.shape, device=sample.device)
            x, y, mask = d16_reduced_u16(radius * angle.cos(), radius * angle.sin(), sector, radius == 0)
            return cls(x, y, mask.to(torch.int32))

        @classmethod
        def from_real(cls, value: Union[float, int, torch.Tensor]) -> "D16ComplexTensor":
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            return _from_real(value)

        @classmethod
        def pack(cls, re: "D16ComplexTensor", im: "D16ComplexTensor") -> "D16ComplexTensor":
            return cls(*_PackComplex.apply(re.x, re.rotation, im.x, im.rotation))

        @classmethod
        def encode_coordinates(cls, real: torch.Tensor, imag: torch.Tensor) -> "D16ComplexTensor":
            return cls.pack(_from_real(real), _from_real(imag))

        @classmethod
        def from_xy(
            cls,
            logic: torch.Tensor,
            emotion: torch.Tensor,
            eps: Union[float, torch.Tensor] = 0.0,
        ) -> "D16ComplexTensor":
            x, y, m = d16_from_xy_u16(logic, emotion, eps=eps)
            return cls(x, y, m)

        @classmethod
        def make(cls, x: torch.Tensor, y: torch.Tensor, rotation: torch.Tensor) -> "D16ComplexTensor":
            rx, ry, rm = d16_make_u16(x, y, rotation)
            return cls(rx, ry, rm)

        @classmethod
        def from_local_tangent(
            cls,
            grad_x: torch.Tensor,
            grad_y: torch.Tensor,
            chart_owner: "D16ComplexTensor",
        ) -> "D16ComplexTensor":
            """Construct a canonical D16ComplexTensor from a signed local tangent in chart_owner's chart."""
            lx, ly, lrot = d16_from_xy_u16(grad_x, grad_y)
            turns, reflected = _chart_parameters(chart_owner.rotation)
            rot = _compose_chart_mask(lrot, turns, reflected)
            is_z = (lrot == MASK_ZERO) | ((lx == 0) & (ly == 0))
            is_nan = (lrot == MASK_NAN) | chart_owner.is_nan()
            rot = torch.where(is_z, torch.full_like(rot, MASK_ZERO), rot)
            rot = torch.where(is_nan, torch.full_like(rot, MASK_NAN), rot)
            return cls(lx.masked_fill(is_nan, 0.0), ly.masked_fill(is_nan, 0.0), rot)

        # --- Basic Properties & Tensor Utilities ---

        @property
        def shape(self) -> torch.Size:
            return self.x.shape

        @property
        def ndim(self) -> int:
            return self.x.ndim

        @property
        def dtype(self) -> torch.dtype:
            return self.x.dtype

        @property
        def device(self) -> torch.device:
            return self.x.device

        def numel(self) -> int:
            return self.x.numel()

        def is_zero(self) -> torch.Tensor:
            return self.rotation == MASK_ZERO

        def is_nan(self) -> torch.Tensor:
            return self.rotation == MASK_NAN

        def is_negative_real(self) -> torch.Tensor:
            return self.rotation == REAL_NEGATIVE

        def state(self) -> torch.Tensor:
            return state_u16(self.rotation)

        def norm_squared(self) -> torch.Tensor:
            return self.x * self.x + self.y * self.y

        def norm_squared_d16(self) -> "D16ComplexTensor":
            ns = self.norm_squared().masked_fill(self.is_nan(), float("nan"))
            return _from_real(ns)

        def to_xy(self) -> Tuple[torch.Tensor, torch.Tensor]:
            return d16_to_xy_u16(self.x, self.y, self.rotation)

        def unpack(self) -> Tuple["D16ComplexTensor", "D16ComplexTensor"]:
            re, im = _from_chart(self.x, self.y, self.rotation)
            undefined = self.is_nan()
            return (
                _from_real(re.masked_fill(undefined, float("nan"))),
                _from_real(im.masked_fill(undefined, float("nan"))),
            )

        def real_signed(self) -> torch.Tensor:
            """Extract signed scalar tensor for a real-axis D16ComplexTensor."""
            val = torch.where(self.rotation == REAL_NEGATIVE, -self.x, self.x)
            return val.masked_fill(self.is_nan(), float("nan"))

        def to(self, *args, **kwargs) -> "D16ComplexTensor":
            nx = self.x.to(*args, **kwargs)
            ny = self.y.to(*args, **kwargs)
            nrot = self.rotation.to(device=nx.device)
            undefined = self.is_nan() | ~torch.isfinite(nx) | ~torch.isfinite(ny)
            if bool(undefined.any()):
                nx = nx.masked_fill(undefined, 0.0)
                ny = ny.masked_fill(undefined, 0.0)
                nrot = nrot.masked_fill(undefined, MASK_NAN)
            return D16ComplexTensor(nx, ny, nrot)

        def round_to_dtype(self, dtype: torch.dtype) -> "D16ComplexTensor":
            if self.dtype == dtype:
                return self
            nx, ny = self.x.to(dtype), self.y.to(dtype)
            undefined = self.is_nan() | ~torch.isfinite(nx) | ~torch.isfinite(ny)
            return D16ComplexTensor.make(
                nx.masked_fill(undefined, 0.0),
                ny.masked_fill(undefined, 0.0),
                self.rotation.masked_fill(undefined, MASK_NAN),
            )

        def clone(self) -> "D16ComplexTensor":
            return D16ComplexTensor(self.x.clone(), self.y.clone(), self.rotation.clone())

        def detach(self) -> "D16ComplexTensor":
            return D16ComplexTensor(self.x.detach(), self.y.detach(), self.rotation.detach())

        def __getitem__(self, index) -> "D16ComplexTensor":
            return D16ComplexTensor(self.x[index], self.y[index], self.rotation[index])

        def copy_slice_(self, index, value: "D16ComplexTensor") -> "D16ComplexTensor":
            self.x[index].copy_(value.x)
            self.y[index].copy_(value.y)
            self.rotation[index].copy_(value.rotation)
            return self

        def map_tensors(self, op: Callable[[torch.Tensor], torch.Tensor]) -> "D16ComplexTensor":
            return D16ComplexTensor(op(self.x), op(self.y), op(self.rotation))

        def split(self, split_size: int, dim: int = 0) -> Tuple["D16ComplexTensor", ...]:
            return tuple(
                D16ComplexTensor(tx, ty, tr)
                for tx, ty, tr in zip(
                    self.x.split(split_size, dim=dim),
                    self.y.split(split_size, dim=dim),
                    self.rotation.split(split_size, dim=dim),
                )
            )

        @staticmethod
        def cat(tensors: Sequence["D16ComplexTensor"], dim: int = 0) -> "D16ComplexTensor":
            return D16ComplexTensor(
                torch.cat([t.x for t in tensors], dim=dim),
                torch.cat([t.y for t in tensors], dim=dim),
                torch.cat([t.rotation for t in tensors], dim=dim),
            )

        def to_indifferent_where(self, cond: torch.Tensor) -> "D16ComplexTensor":
            z_x = torch.where(cond, torch.zeros_like(self.x), self.x)
            z_y = torch.where(cond, torch.zeros_like(self.y), self.y)
            z_m = torch.where(cond, torch.full_like(self.rotation, MASK_ZERO), self.rotation)
            return D16ComplexTensor(z_x, z_y, z_m)

        def apply_Manifold_Threshold_L1(self, eps: Union[float, torch.Tensor]) -> "D16ComplexTensor":
            re, im = self.unpack()
            return self.to_indifferent_where((re.x + im.x) <= eps)

        # --- Dihedral & Algebraic Operations (Autograd-Enabled) ---

        def apply_group_op(self, op_id: int) -> "D16ComplexTensor":
            return D16ComplexTensor(self.x, self.y, apply_group_op_u16(self.rotation, op_id))

        def conjugate(self) -> "D16ComplexTensor":
            return self.apply_group_op(8)

        def negate(self) -> "D16ComplexTensor":
            return self.apply_group_op(4)

        def scale(self, factor: Union[float, int, torch.Tensor]) -> "D16ComplexTensor":
            return _scale_wedge(self, factor)

        def add(self, other: "D16ComplexTensor") -> "D16ComplexTensor":
            if not isinstance(other, D16ComplexTensor):
                return NotImplemented
            return D16ComplexTensor(*_AddComplex.apply(self.x, self.y, self.rotation, other.x, other.y, other.rotation))

        def multiply(self, other: Union["D16ComplexTensor", float, int, torch.Tensor]) -> "D16ComplexTensor":
            if isinstance(other, D16ComplexTensor):
                return D16ComplexTensor(
                    *_MultiplyComplex.apply(self.x, self.y, self.rotation, other.x, other.y, other.rotation)
                )
            if isinstance(other, (int, float, torch.Tensor)):
                return _scale_wedge(self, other)
            return NotImplemented

        def square(self) -> "D16ComplexTensor":
            return self.multiply(self)

        def __add__(self, other: "D16ComplexTensor") -> "D16ComplexTensor":
            return self.add(other)

        def __sub__(self, other: "D16ComplexTensor") -> "D16ComplexTensor":
            if not isinstance(other, D16ComplexTensor):
                return NotImplemented
            return self.add(other.negate())

        def __mul__(self, other: Union["D16ComplexTensor", float, int, torch.Tensor]) -> "D16ComplexTensor":
            return self.multiply(other)

        def __rmul__(self, other: Union[float, int, torch.Tensor]) -> "D16ComplexTensor":
            if isinstance(other, (int, float, torch.Tensor)):
                return _scale_wedge(self, other)
            return NotImplemented

        def __neg__(self) -> "D16ComplexTensor":
            return self.negate()

        # --- Real-Axis Subalgebra Operations (Scores, Softmax, Loss, Linear Contractions) ---

        def real_add(self, other: "D16ComplexTensor") -> "D16ComplexTensor":
            return _real_add(self, other)

        def real_sub(self, other: "D16ComplexTensor") -> "D16ComplexTensor":
            return _real_add(self, _real_negate(other))

        def real_negate(self) -> "D16ComplexTensor":
            return _real_negate(self)

        def real_multiply(self, other: "D16ComplexTensor") -> "D16ComplexTensor":
            return _real_multiply(self, other)

        def real_matmul(self, other: "D16ComplexTensor") -> "D16ComplexTensor":
            return _real_matmul(self, other)

        def real_sigmoid(self) -> "D16ComplexTensor":
            negative = self.rotation == REAL_NEGATIVE
            exp = (-self.x).exp()
            one = torch.ones_like(exp)
            probability = torch.where(negative, exp, one) / (one + exp)
            return _real_magnitude(probability, torch.zeros_like(negative), self.is_nan())

        def real_softmax(self, allowed: Optional[torch.Tensor] = None) -> "D16ComplexTensor":
            if self.shape[-1] == 0:
                return self
            undefined = self.is_nan()
            active = ~undefined if allowed is None else allowed & ~undefined
            if allowed is not None:
                undefined = undefined & allowed
            _, terms = _shifted_exp(self, active)
            probabilities = (terms / terms.sum(dim=-1, keepdim=True).clamp_min(1)).to(self.dtype)
            return _real_magnitude(probabilities, torch.zeros_like(undefined), undefined.any(dim=-1, keepdim=True))

        def real_argmax(self) -> torch.Tensor:
            negative = self.rotation == REAL_NEGATIVE
            nonnegative = ~negative & ~self.is_nan()
            positive_max, positive_address = self.x.masked_fill(~nonnegative, -torch.inf).max(-1, keepdim=True)
            negative_min, negative_address = self.x.masked_fill(~negative, torch.inf).min(-1, keepdim=True)
            first = torch.where(positive_max.isfinite(), positive_address, negative_address)
            if bool((~positive_max.isfinite() & ~negative_min.isfinite()).any()):
                raise ValueError("Cannot select a token from undefined logits")
            return first

        def real_cross_entropy(self, targets: torch.Tensor) -> "D16ComplexTensor":
            scores = self.to(torch.float32) if self.dtype in (torch.float16, torch.bfloat16) else self
            valid = targets != -100
            addresses = targets.masked_fill(~valid, 0).unsqueeze(-1)
            selected = scores.map_tensors(lambda t: t.gather(-1, addresses).squeeze(-1))
            undefined = scores.is_nan()
            maximum, terms = _shifted_exp(scores, ~undefined)
            distance = _from_real(maximum.squeeze(-1)).real_sub(selected)
            losses = distance.real_add(_from_real(terms.sum(dim=-1).log()))
            negative = losses.rotation == REAL_NEGATIVE
            total = (
                losses.x.masked_fill(~valid | negative, 0).sum()
                - losses.x.masked_fill(~valid | ~negative, 0).sum()
            )
            mean = (total / valid.sum()).masked_fill((undefined.any(dim=-1) & valid).any(), float("nan"))
            return _from_real(mean)

        def __repr__(self) -> str:
            return f"D16ComplexTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"


__all__ = [
    "SQRT2",
    "INV_SQRT2",
    "TAN_PI_8",
    "EPS_BOUND",
    "MASK_ZERO",
    "MASK_NAN",
    "REAL_POSITIVE",
    "REAL_NEGATIVE",
    "STATE_INDIFFERENT",
    "STATE_UNDEFINED",
    "STATE_LABELS",
    "rol",
    "ror",
    "reverse_bits",
    "bit_index",
    "is_one_hot",
    "even_anchor",
    "on_bisector",
    "above_bisector",
    "rotated_sum",
    "D16Complex",
    "rol_u16",
    "ror_u16",
    "reverse_bits_u16",
    "bit_index_u16",
    "is_one_hot_u16",
    "even_anchor_u16",
    "state_u16",
    "on_bisector_d16",
    "above_bisector_d16",
    "rotated_sum_d16",
    "d16_reduced_u16",
    "d16_make_u16",
    "d16_from_xy_u16",
    "d16_to_xy_u16",
    "d16_multiply_u16",
    "apply_group_op_u16",
    "apply_all_16_d16_ops_u16",
    "D16ComplexTensor",
]