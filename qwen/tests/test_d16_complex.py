"""Unit tests for the d16_complex module (scalar D16Complex and PyTorch D16ComplexTensor).

Can be executed via:
    python tests/test_d16_complex.py
    python -m unittest discover -s tests
    pytest tests/test_d16_complex.py
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

# Автоматический поиск директории src с модулем d16_complex.py
_THIS_DIR = Path(__file__).resolve().parent
for _candidate in (
    _THIS_DIR.parent / "src",
    _THIS_DIR.parent / "qwen" / "src",
    _THIS_DIR,
):
    if (_candidate / "d16_complex.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

import d16_complex as d16
from d16_complex import (
    D16Complex,
    EPS_BOUND,
    INV_SQRT2,
    MASK_NAN,
    MASK_ZERO,
    SQRT2,
    STATE_INDIFFERENT,
    STATE_UNDEFINED,
    TAN_PI_8,
    above_bisector,
    bit_index,
    even_anchor,
    is_one_hot,
    on_bisector,
    reverse_bits,
    rol,
    ror,
    rotated_sum,
)

try:
    import torch
    from d16_complex import (
        D16ComplexTensor,
        apply_all_16_d16_ops_u16,
        apply_group_op_u16,
        d16_from_xy_u16,
        d16_make_u16,
        d16_multiply_u16,
        d16_to_xy_u16,
        state_u16,
    )
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


class TestScalarBitwiseAndHelpers(unittest.TestCase):
    """Проверка 16-битных побитовых операций и геометрических предикатов клина."""

    def test_rol_and_ror_identities(self) -> None:
        for s in range(16):
            m = 1 << s
            for shift in range(33):
                self.assertEqual(rol(m, shift), 1 << ((s + shift) & 15))
                self.assertEqual(ror(m, shift), 1 << ((s - shift) & 15))
                self.assertEqual(ror(rol(m, shift), shift), m)

        self.assertEqual(rol(MASK_ZERO, 5), MASK_ZERO)
        self.assertEqual(ror(MASK_NAN, 7), MASK_NAN)

    def test_reverse_bits_involution_and_mapping(self) -> None:
        for s in range(16):
            m = 1 << s
            rev = reverse_bits(m)
            self.assertEqual(rev, 1 << (15 - s))
            self.assertEqual(reverse_bits(rev), m)

    def test_one_hot_and_even_anchor(self) -> None:
        for s in range(16):
            m = 1 << s
            self.assertTrue(is_one_hot(m))
            self.assertEqual(bit_index(m), s)
            self.assertEqual(even_anchor(m), s)

        # Двухбитные границы: уникальный четный бит должен возвращаться как якорь
        for s in range(16):
            boundary_mask = (1 << s) | (1 << ((s + 1) & 15))
            self.assertFalse(is_one_hot(boundary_mask))
            expected_even = s if (s % 2 == 0) else ((s + 1) & 15)
            self.assertEqual(even_anchor(boundary_mask), expected_even)

    def test_bisector_predicates_and_rotated_sum(self) -> None:
        self.assertTrue(on_bisector(10.0, 10.0 * TAN_PI_8))
        self.assertFalse(above_bisector(10.0, 10.0 * TAN_PI_8))
        self.assertTrue(above_bisector(10.0, 10.0 * TAN_PI_8 + 1e-3))
        self.assertFalse(above_bisector(10.0, 10.0 * TAN_PI_8 - 1e-3))
        self.assertAlmostEqual(rotated_sum(3.0, 5.0), (3.0 + 5.0) * INV_SQRT2, places=14)
        self.assertEqual(rotated_sum(0.0, 0.0), 0.0)


class TestScalarD16ComplexEncodingAndMake(unittest.TestCase):
    """Проверка конструкторов, канонического клина [0°, 22.5°], границ и состояний 0..17."""

    def test_zero_and_nan_states(self) -> None:
        z0 = D16Complex.zero()
        self.assertTrue(z0.is_zero())
        self.assertFalse(z0.is_nan())
        self.assertEqual(z0.rotation, MASK_ZERO)
        self.assertEqual(z0.state(), STATE_INDIFFERENT)
        self.assertEqual(z0.to_xy(), (0.0, 0.0))

        zn = D16Complex.nan()
        self.assertTrue(zn.is_nan())
        self.assertFalse(zn.is_zero())
        self.assertEqual(zn.rotation, MASK_NAN)
        self.assertEqual(zn.state(), STATE_UNDEFINED)
        re, im = zn.to_xy()
        self.assertTrue(math.isnan(re) and math.isnan(im))

        # Порог eps в from_xy переводит малые входы в состояние Indifferent (0xFFFF)
        z_eps = D16Complex.from_xy(0.05, -0.04, eps=0.1)
        self.assertTrue(z_eps.is_zero())
        self.assertEqual(z_eps.state(), STATE_INDIFFERENT)

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            D16Complex(-1.0, 0.2, 1)
        with self.assertRaises(ValueError):
            D16Complex(1.0, -0.2, 1)
        with self.assertRaises(ValueError):
            D16Complex(1.0, 0.0, MASK_ZERO)  # 0xFFFF зарезервирован для нуля
        with self.assertRaises(ValueError):
            D16Complex(1.0, 0.0, MASK_NAN)   # 0x0000 требует (0, 0)
        with self.assertRaises(ValueError):
            D16Complex(float("inf"), 0.0, 1)
        with self.assertRaises(ValueError):
            # Несоседние два бита маски
            D16Complex(1.0, 0.0, (1 << 0) | (1 << 2))

    def test_bijection_from_xy_to_xy_across_grid_and_rays(self) -> None:
        grid = [-7.5, -3.2, -1.0, -TAN_PI_8, -0.15, 0.0, 0.15, TAN_PI_8, 1.0, 3.2, 7.5]
        for l_val in grid:
            for e_val in grid:
                z = D16Complex.from_xy(l_val, e_val)
                # Проверка инварианта фундаментального клина: 0 <= y <= (sqrt(2)-1)*x
                self.assertGreaterEqual(z.x, 0.0)
                self.assertGreaterEqual(z.y, 0.0)
                self.assertLessEqual(z.y, TAN_PI_8 * z.x + 1e-12)

                rec_l, rec_e = z.to_xy()
                self.assertAlmostEqual(rec_l, l_val, places=13)
                self.assertAlmostEqual(rec_e, e_val, places=13)
                self.assertAlmostEqual(abs(z), math.hypot(l_val, e_val), places=13)

        # Проверка всех 16 секторов по углам (внутренние точки и границы кратные 22.5°)
        for s in range(16):
            mid_angle = math.radians(s * 22.5 + 11.25)
            l_val = 4.0 * math.cos(mid_angle)
            e_val = 4.0 * math.sin(mid_angle)
            z = D16Complex.from_xy(l_val, e_val)
            self.assertEqual(z.state(), s)
            self.assertFalse(z.is_boundary)
            self.assertEqual(z.rotation, 1 << s)

            # Точный луч границы s * 22.5°
            bnd_angle = math.radians(s * 22.5)
            z_bnd = D16Complex.from_xy(4.0 * math.cos(bnd_angle), 4.0 * math.sin(bnd_angle))
            self.assertTrue(z_bnd.is_boundary)
            rl, re = z_bnd.to_xy()
            self.assertAlmostEqual(rl, 4.0 * math.cos(bnd_angle), places=13)
            self.assertAlmostEqual(re, 4.0 * math.sin(bnd_angle), places=13)

    def test_make_folds_noncanonical_coordinates_in_all_16_sectors(self) -> None:
        # Подаем неканонические пары (включая y > x и выше биссектрисы 22.5°)
        noncanonical_pairs = [(3.0, 5.0), (1.5, 4.2), (4.0, 2.5), (2.0, 1.9), (3.5, 0.0)]
        for s in range(16):
            for x_in, y_in in noncanonical_pairs:
                # Эталонное значение через аналитическую развертку карты сектора s
                q, r = s >> 2, s & 3
                is_diag = r in (1, 2)
                wx = rotated_sum(x_in, y_in) if is_diag else x_in
                wy = ((x_in - y_in) * INV_SQRT2) if is_diag else y_in
                u = wy if r >= 2 else wx
                v = wx if r >= 2 else wy
                ref_re, ref_im = (
                    (u, v) if q == 0 else (-v, u) if q == 1 else (-u, -v) if q == 2 else (v, -u)
                )

                z = D16Complex.make(x_in, y_in, 1 << s)
                self.assertGreaterEqual(z.x, 0.0)
                self.assertGreaterEqual(z.y, 0.0)
                self.assertLessEqual(z.y, TAN_PI_8 * z.x + 1e-12)
                out_re, out_im = z.to_xy()
                self.assertAlmostEqual(out_re, ref_re, places=13)
                self.assertAlmostEqual(out_im, ref_im, places=13)


class TestScalarD16AlgebraAndGroupAction(unittest.TestCase):
    """Проверка умножения, возведения в квадрат, локального сложения и группы Z_8 |>< Z_2."""

    def test_pure_bitwise_group_action_preserves_wedge_xy(self) -> None:
        z = D16Complex.from_xy(3.7, 1.2)
        orig_x, orig_y = z.x, z.y
        l0, e0 = z.to_xy()

        for op_id in range(16):
            z_op = z.apply_group_op(op_id)
            # Главный инвариант D16Complex: координаты (x, y) в клине НЕ меняются ни на бит!
            self.assertEqual(z_op.x, orig_x)
            self.assertEqual(z_op.y, orig_y)

            # Проверяем геометрическое соответствие повороту/отражению в плоскости (L, E)
            k = op_id & 7
            ang = math.radians(45.0 * k)
            c, s = math.cos(ang), math.sin(ang)
            base_l, base_e = (l0, e0) if op_id < 8 else (l0, -e0)
            exp_l = c * base_l - s * base_e
            exp_e = s * base_l + c * base_e

            act_l, act_e = z_op.to_xy()
            self.assertAlmostEqual(act_l, exp_l, places=13)
            self.assertAlmostEqual(act_e, exp_e, places=13)

    def test_multiply_and_square_inside_wedge(self) -> None:
        samples = [
            (0.0, 0.0),
            (2.5, 0.0),
            (0.0, -3.0),
            (2.0, 2.0),
            (-2.0, 2.0 * TAN_PI_8),
            (3.1, 0.7),
            (-1.8, 2.9),
            (-2.4, -1.1),
            (1.4, -3.3),
        ]
        for l1, e1 in samples:
            z1 = D16Complex.from_xy(l1, e1)
            # Проверка квадрата z1.square()
            z_sq = z1.square()
            sq_re, sq_im = z_sq.to_xy()
            self.assertAlmostEqual(sq_re, l1 * l1 - e1 * e1, places=12)
            self.assertAlmostEqual(sq_im, 2.0 * l1 * e1, places=12)

            for l2, e2 in samples:
                z2 = D16Complex.from_xy(l2, e2)
                zm = z1 * z2
                self.assertLessEqual(zm.y, TAN_PI_8 * zm.x + 1e-12)
                m_re, m_im = zm.to_xy()
                self.assertAlmostEqual(m_re, l1 * l2 - e1 * e2, places=12)
                self.assertAlmostEqual(m_im, l1 * e2 + e1 * l2, places=12)

    def test_local_chart_add_and_sub(self) -> None:
        samples = [
            (0.0, 0.0),
            (3.0, 0.5),
            (-2.1, 1.4),
            (-1.5, -2.8),
            (0.9, -3.2),
            (2.0, 2.0 * TAN_PI_8),
        ]
        for l1, e1 in samples:
            z1 = D16Complex.from_xy(l1, e1)
            self.assertTrue((z1 - z1).is_zero())
            for l2, e2 in samples:
                z2 = D16Complex.from_xy(l2, e2)
                z_sum = z1 + z2
                s_re, s_im = z_sum.to_xy()
                self.assertAlmostEqual(s_re, l1 + l2, places=12)
                self.assertAlmostEqual(s_im, e1 + e2, places=12)

                z_diff = z1 - z2
                d_re, d_im = z_diff.to_xy()
                self.assertAlmostEqual(d_re, l1 - l2, places=12)
                self.assertAlmostEqual(d_im, e1 - e2, places=12)


@unittest.skipUnless(_HAS_TORCH, "PyTorch is required for D16ComplexTensor tests")
class TestVectorizedD16ComplexTensor(unittest.TestCase):
    """Проверка полного совпадения векторизованного D16ComplexTensor со скалярным эталоном."""

    def test_tensor_matches_scalar_on_full_grid(self) -> None:
        vals = torch.linspace(-4.0, 4.0, 17, dtype=torch.float64)
        L, E = torch.meshgrid(vals, vals, indexing="ij")

        zt = D16ComplexTensor.from_xy(L, E)
        rec_L, rec_E = zt.to_xy()

        self.assertLess(float((rec_L - L).abs().max()), 1e-14)
        self.assertLess(float((rec_E - E).abs().max()), 1e-14)
        self.assertTrue(bool((zt.x >= 0.0).all()))
        self.assertTrue(bool((zt.y >= 0.0).all()))
        self.assertTrue(bool((zt.y <= TAN_PI_8 * zt.x + 1e-12).all()))

        # Поэлементное сравнение со скалярным классом D16Complex
        st_tensor = zt.state()
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                zs = D16Complex.from_xy(float(L[i, j]), float(E[i, j]))
                self.assertAlmostEqual(float(zt.x[i, j]), zs.x, places=14)
                self.assertAlmostEqual(float(zt.y[i, j]), zs.y, places=14)
                self.assertEqual(int(zt.rotation[i, j]), zs.rotation)
                self.assertEqual(int(st_tensor[i, j]), zs.state())

    def test_tensor_multiply_and_square(self) -> None:
        torch.manual_seed(42)
        L1 = torch.randn(32, 64, dtype=torch.float64) * 3.0
        E1 = torch.randn(32, 64, dtype=torch.float64) * 3.0
        L2 = torch.randn(32, 64, dtype=torch.float64) * 3.0
        E2 = torch.randn(32, 64, dtype=torch.float64) * 3.0

        z1 = D16ComplexTensor.from_xy(L1, E1)
        z2 = D16ComplexTensor.from_xy(L2, E2)

        prod = z1 * z2
        p_re, p_im = prod.to_xy()
        true_re = L1 * L2 - E1 * E2
        true_im = L1 * E2 + E1 * L2
        self.assertLess(float((p_re - true_re).abs().max()), 1e-13)
        self.assertLess(float((p_im - true_im).abs().max()), 1e-13)

        sq = z1.square()
        sq_re, sq_im = sq.to_xy()
        self.assertLess(float((sq_re - (L1.square() - E1.square())).abs().max()), 1e-13)
        self.assertLess(float((sq_im - (2.0 * L1 * E1)).abs().max()), 1e-13)

    def test_tensor_make_and_group_ops_and_indifferent_masking(self) -> None:
        x_in = torch.tensor([3.0, 1.5, 4.0, 2.0], dtype=torch.float64)
        y_in = torch.tensor([5.0, 4.2, 2.5, 1.9], dtype=torch.float64)

        for s in range(16):
            mask_in = torch.full_like(x_in, 1 << s, dtype=torch.long)
            ref_re, ref_im = d16_to_xy_u16(x_in, y_in, mask_in)
            zt = D16ComplexTensor.make(x_in, y_in, mask_in)
            out_re, out_im = zt.to_xy()
            self.assertLess(float((out_re - ref_re).abs().max()), 1e-14)
            self.assertLess(float((out_im - ref_im).abs().max()), 1e-14)

        # Проверка 16 групповых операций и перевода в состояние Indifferent (0xFFFF)
        z_batch = D16ComplexTensor.from_xy(x_in, -y_in)
        all_masks = z_batch.all_16_group_masks()
        self.assertEqual(all_masks.shape, (4, 16))
        for op_id in range(16):
            z_op = z_batch.apply_group_op(op_id)
            self.assertTrue(bool((z_op.x == z_batch.x).all()))
            self.assertTrue(bool((z_op.y == z_batch.y).all()))

        cond = torch.tensor([True, False, True, False])
        z_masked = z_batch.to_indifferent_where(cond)
        self.assertEqual(int(z_masked.state()[0]), STATE_INDIFFERENT)
        self.assertEqual(int(z_masked.state()[2]), STATE_INDIFFERENT)
        self.assertEqual(int(z_masked.state()[1]), int(z_batch.state()[1]))
        re_m, im_m = z_masked.to_xy()
        self.assertEqual(float(re_m[0]), 0.0)
        self.assertEqual(float(im_m[0]), 0.0)


if __name__ == "__main__":
    print("=" * 88)
    print("RUNNING UNIT TESTS FOR d16_complex.py (SCALAR D16Complex & TORCH D16ComplexTensor)")
    print("=" * 88)
    unittest.main(verbosity=2)
