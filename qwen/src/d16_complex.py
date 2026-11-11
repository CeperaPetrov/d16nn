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
import math
from numbers import Real
from typing import Tuple, Union

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
# 3. SCALAR D16Complex OBJECT (PORT OF struct D16Complex & D8Complex)
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
            # 1. Swap x/y while preserving value: even k -> k+3, odd k -> k-3 (mod 16)
            if y > x:
                x, y = y, x
                k = (k + (13 if (k & 1) else 3)) & 15
            # 2. Fold across 22.5° bisector if above_bisector(x, y)
            if above_bisector(x, y):
                nx = rotated_sum(x, y)
                ny = (x - y) * INV_SQRT2
                x, y = nx, ny
                k ^= 1
            # 3. Snap to boundary mask if on y=0 or on 22.5° bisector
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

    # --- Static / Class Constructors & Helpers ---

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
        """Construct and fold (x, y, mask) into the canonical [0°, 22.5°] wedge."""
        return cls(x, y, mask)

    @classmethod
    def reduced(cls, x: float, y: float, counter: int) -> D16Complex:
        """Construct from already reduced wedge coordinates (x, y) and one-hot counter."""
        if not (_finite_value(x) and _finite_value(y)):
            return cls.nan()
        if x == 0.0 and y == 0.0:
            return cls.zero()
        return cls(x, y, counter)

    @classmethod
    def from_xy(cls, logic: float, emotion: float, eps: float = 0.0) -> D16Complex:
        """Encode Cartesian (logic, emotion) into the [0°, 22.5°] wedge and u16 mask."""
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

    # --- State & Properties ---

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
        """Return canonical state in 0..15, 16 for Indifferent (0xFFFF), 17 for Undefined (0x0000)."""
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
        """Unfold from the [0°, 22.5°] wedge and u16 mask back to Cartesian (Re, Im)."""
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

    # --- Pure Bitwise Dihedral Group Operations (Z_8 |>< Z_2) ---

    def apply_group_op(self, op_id: int) -> D16Complex:
        """Pure bitwise group action of Z_8 |>< Z_2 (op_id in 0..15): (x, y) NEVER changes."""
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
        """Complex conjugation (L, E) -> (L, -E) via pure bit reversal of rotation."""
        return self.apply_group_op(8)

    def negate(self) -> D16Complex:
        """Negation (L, E) -> (-L, -E) via 8-bit circular shift of rotation."""
        return self.apply_group_op(4)

    # --- Local Wedge Arithmetic (Multiply, Square, Add) ---

    def multiply(self, other: D16Complex) -> D16Complex:
        """Multiply two D16Complex numbers directly inside the [0°, 22.5°] wedge."""
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
        """Square z^2 = z * z inside the [0°, 22.5°] fundamental wedge."""
        return self.multiply(self)

    def add(self, other: D16Complex) -> D16Complex:
        """Add two D16Complex numbers in self's local reference chart without global Re/Im."""
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
# 4. VECTORIZED PYTORCH TENSOR ARITHMETIC (D16ComplexTensor & *_u16)
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

    class D16ComplexTensor:
        """Vectorized PyTorch container of D16Complex objects over arbitrary tensor shapes."""

        __slots__ = ("x", "y", "rotation")

        def __init__(self, x: torch.Tensor, y: torch.Tensor, rotation: torch.Tensor) -> None:
            self.x = x
            self.y = y
            self.rotation = rotation

        @classmethod
        def from_xy(
            cls,
            re: torch.Tensor,
            im: torch.Tensor,
            eps: Union[float, torch.Tensor] = 0.0,
        ) -> D16ComplexTensor:
            x, y, rot = d16_from_xy_u16(re, im, eps=eps)
            return cls(x, y, rot)

        @classmethod
        def make(cls, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> D16ComplexTensor:
            rx, ry, rmask = d16_make_u16(x, y, mask)
            return cls(rx, ry, rmask)

        @property
        def shape(self) -> torch.Size:
            return self.x.shape

        @property
        def device(self) -> torch.device:
            return self.x.device

        @property
        def dtype(self) -> torch.dtype:
            return self.x.dtype

        def to(self, *args, **kwargs) -> D16ComplexTensor:
            nx = self.x.to(*args, **kwargs)
            ny = self.y.to(*args, **kwargs)
            nrot = self.rotation.to(device=nx.device)
            return D16ComplexTensor(nx, ny, nrot)

        def detach(self) -> D16ComplexTensor:
            return D16ComplexTensor(self.x.detach(), self.y.detach(), self.rotation.detach())

        def clone(self) -> D16ComplexTensor:
            return D16ComplexTensor(self.x.clone(), self.y.clone(), self.rotation.clone())

        def __getitem__(self, item) -> D16ComplexTensor:
            return D16ComplexTensor(self.x[item], self.y[item], self.rotation[item])

        def to_xy(self) -> Tuple[torch.Tensor, torch.Tensor]:
            return d16_to_xy_u16(self.x, self.y, self.rotation)

        def real(self) -> torch.Tensor:
            re, _ = self.to_xy()
            return re

        def imag(self) -> torch.Tensor:
            _, im = self.to_xy()
            return im

        def state(self) -> torch.Tensor:
            return state_u16(self.rotation)

        def is_zero(self) -> torch.Tensor:
            return self.rotation == MASK_ZERO

        def is_nan(self) -> torch.Tensor:
            return self.rotation == MASK_NAN

        def norm_squared(self) -> torch.Tensor:
            return self.x.square() + self.y.square()

        def abs(self) -> torch.Tensor:
            return torch.hypot(self.x, self.y)

        def multiply(self, other: D16ComplexTensor) -> D16ComplexTensor:
            nx, ny, nrot = d16_multiply_u16(
                self.x, self.y, self.rotation,
                other.x, other.y, other.rotation,
            )
            return D16ComplexTensor(nx, ny, nrot)

        def square(self) -> D16ComplexTensor:
            return self.multiply(self)

        def apply_group_op(self, op_id: int) -> D16ComplexTensor:
            """Pure bitwise Z_8 |>< Z_2 action on rotation mask; (x, y) is untouched."""
            return D16ComplexTensor(self.x, self.y, apply_group_op_u16(self.rotation, op_id))

        def all_16_group_masks(self) -> torch.Tensor:
            return apply_all_16_d16_ops_u16(self.rotation)

        def conjugate(self) -> D16ComplexTensor:
            return self.apply_group_op(8)

        def negate(self) -> D16ComplexTensor:
            return self.apply_group_op(4)

        def to_indifferent_where(self, cond: torch.Tensor) -> D16ComplexTensor:
            """Set mask to 0xFFFF (State 16: Indifferent) where cond is True."""
            valid_cond = cond & (self.rotation != MASK_NAN)
            new_rot = torch.where(valid_cond, torch.full_like(self.rotation, MASK_ZERO), self.rotation)
            new_x = torch.where(valid_cond, torch.zeros_like(self.x), self.x)
            new_y = torch.where(valid_cond, torch.zeros_like(self.y), self.y)
            return D16ComplexTensor(new_x, new_y, new_rot)

        def __mul__(self, other: D16ComplexTensor) -> D16ComplexTensor:
            return self.multiply(other)

        def __neg__(self) -> D16ComplexTensor:
            return self.negate()

        def __repr__(self) -> str:
            return f"D16ComplexTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"


__all__ = [
    "SQRT2",
    "INV_SQRT2",
    "TAN_PI_8",
    "EPS_BOUND",
    "MASK_ZERO",
    "MASK_NAN",
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
