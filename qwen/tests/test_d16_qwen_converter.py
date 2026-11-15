"""Unit tests for d16_qwen_converter.py (1-to-1 D16Complex conversion and Z_8 |>< Z_2 orbit discovery).

Can be executed via:
    python tests/test_d16_qwen_converter.py
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

_THIS_DIR = Path(__file__).resolve().parent
for _candidate in (
    _THIS_DIR.parent / "src",
    _THIS_DIR.parent / "qwen" / "src",
    _THIS_DIR,
):
    if (_candidate / "d16_qwen_converter.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

import torch
import torch.nn.functional as F

from d16_qwen_converter import (
    OP_QUAD_SIGN,
    QwenToD16Converter,
    compute_d16_pairwise_alignment,
    convert_mlp_layer_to_d16,
    d16_group_matrices,
    select_d16_quadruplets,
)
from d16_qwen_model import D16QwenConfig, D16QwenMLP


class TestD16GroupMatricesAndOrbitDiscovery(unittest.TestCase):
    """Проверка 16 матриц группы Z_8 |>< Z_2, сохранения всех N объектов D16Complex и поиска орбит."""

    def test_group_matrices_orthogonality_and_quadratic_identities(self) -> None:
        mats = d16_group_matrices(dtype=torch.float64)
        self.assertEqual(mats.shape, (16, 2, 2))
        eye = torch.eye(2, dtype=torch.float64)

        L = torch.tensor(2.3, dtype=torch.float64)
        E = torch.tensor(-1.4, dtype=torch.float64)
        vec = torch.stack([L, E])
        im_half = L * E
        re_half = 0.5 * (L * L - E * E)

        for op in range(16):
            M = mats[op]
            self.assertLess(float((M.T @ M - eye).abs().max().item()), 1e-14)
            u_p, g_p = M @ vec
            prod = float((u_p * g_p).item())
            expected = OP_QUAD_SIGN[op] * (im_half if op < 8 else re_half)
            self.assertAlmostEqual(prod, float(expected.item()), places=13)

    def test_synthetic_orbit_discovery_and_exact_layer_preservation(self) -> None:
        torch.manual_seed(7)
        hidden_size = 32
        K_true = 6
        N = K_true * 4  # 24 независимых объекта D16Complex, лежащих на 6 орбитах по 4 точки

        mats = d16_group_matrices(dtype=torch.float32)
        base_B = F.normalize(torch.randn(K_true, 2, hidden_size), dim=-1) * 1.5

        planted_ops = torch.tensor([[0, 1, 8, 9]] * K_true, dtype=torch.long)
        U_list, G_list = [], []
        for q in range(K_true):
            for m in range(4):
                op_id = int(planted_ops[q, m].item())
                mapped = mats[op_id] @ base_B[q]
                U_list.append(mapped[0])
                G_list.append(mapped[1])

        U_real = torch.stack(U_list, dim=0)  # [24, 32]
        G_real = torch.stack(G_list, dim=0)  # [24, 32]
        D_real = torch.randn(hidden_size, N) * 0.2

        params, report = convert_mlp_layer_to_d16(U_real, G_real, D_real)
        # Главное требование: количество объектов D16Complex НЕ уменьшается (24 -> 24)!
        self.assertEqual(report.orig_intermediate_size, N)
        self.assertEqual(report.d16_intermediate_size, N)
        self.assertEqual(params["up_proj.weight"].shape, (N, hidden_size))
        self.assertEqual(params["gate_proj.weight"].shape, (N, hidden_size))
        self.assertEqual(params["down_proj.weight"].shape, (hidden_size, N))

        # При этом анализатор орбит точно находит все 6 орбит со счетом 1.0
        self.assertEqual(report.quadruplets.shape, (K_true, 4))
        self.assertEqual(len(set(report.quadruplets.flatten().tolist())), N)
        self.assertAlmostEqual(report.mean_orbit_score, 1.0, places=4)

    def test_exact_1to1_layer_equivalence_to_qwen_mlp(self) -> None:
        torch.manual_seed(11)
        hidden_size = 48
        N = 64
        U_real = torch.randn(N, hidden_size) / (hidden_size ** 0.5)
        G_real = torch.randn(N, hidden_size) / (hidden_size ** 0.5)
        D_real = torch.randn(hidden_size, N) / (N ** 0.5)

        params, report = convert_mlp_layer_to_d16(U_real, G_real, D_real)
        self.assertEqual(report.d16_intermediate_size, N)

        cfg = D16QwenConfig.tiny_test_config(hidden_size=hidden_size, intermediate_size=N)
        d16_mlp = D16QwenMLP(cfg).eval()
        d16_mlp.load_state_dict(params)

        x = torch.randn(8, 16, hidden_size)
        with torch.no_grad():
            out_d16 = d16_mlp(x)
            out_qwen = F.linear(F.linear(x, U_real) * F.silu(F.linear(x, G_real)), D_real)

        max_err = float((out_d16 - out_qwen).abs().max().item())
        self.assertLess(max_err, 1e-5)


class TestFullModelStateDictAndCheckpointConversion(unittest.TestCase):
    """Проверка полной конвертации модели Qwen в D16QwenForCausalLM (в памяти и с диска)."""

    @staticmethod
    def _create_mock_qwen_checkpoint(vocab_size=64, hidden_size=32, inter_size=48, num_layers=2):
        cfg_dict = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": inter_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
        }
        kv_dim = (hidden_size // 4) * 2
        sd = {
            "model.embed_tokens.weight": torch.randn(vocab_size, hidden_size) * 0.02,
            "model.norm.weight": torch.ones(hidden_size),
        }
        for i in range(num_layers):
            p = f"model.layers.{i}."
            sd[p + "input_layernorm.weight"] = torch.ones(hidden_size)
            sd[p + "post_attention_layernorm.weight"] = torch.ones(hidden_size)
            sd[p + "self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size) * 0.02
            sd[p + "self_attn.q_proj.bias"] = torch.zeros(hidden_size)
            sd[p + "self_attn.k_proj.weight"] = torch.randn(kv_dim, hidden_size) * 0.02
            sd[p + "self_attn.k_proj.bias"] = torch.zeros(kv_dim)
            sd[p + "self_attn.v_proj.weight"] = torch.randn(kv_dim, hidden_size) * 0.02
            sd[p + "self_attn.v_proj.bias"] = torch.zeros(kv_dim)
            sd[p + "self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size) * 0.02
            sd[p + "mlp.up_proj.weight"] = torch.randn(inter_size, hidden_size) * 0.1
            sd[p + "mlp.gate_proj.weight"] = torch.randn(inter_size, hidden_size) * 0.1
            sd[p + "mlp.down_proj.weight"] = torch.randn(hidden_size, inter_size) * 0.1
        return cfg_dict, sd

    def test_convert_state_dict_and_from_pretrained_dir(self) -> None:
        torch.manual_seed(99)
        cfg_dict, sd = self._create_mock_qwen_checkpoint()

        model_d16, rep = QwenToD16Converter.convert_state_dict(sd, cfg_dict)
        self.assertEqual(model_d16.config.intermediate_size, 48)
        self.assertEqual(rep.d16_intermediate_size, 48)
        self.assertEqual(rep.global_op_histogram.shape, (16,))

        inp = torch.randint(0, 64, (2, 10))
        out = model_d16(inp)
        self.assertEqual(out.logits.shape, (2, 10, 64))
        self.assertTrue(bool(torch.isfinite(out.logits).all()))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "config.json").write_text(json.dumps(cfg_dict), encoding="utf-8")
            torch.save(sd, tmp_path / "pytorch_model.bin")

            model_disk, rep_disk = QwenToD16Converter.from_pretrained_dir(tmp_path)
            self.assertEqual(model_disk.config.intermediate_size, 48)
            self.assertEqual(len(rep_disk.layer_reports), 2)
            gen = model_disk.generate(inp[:, :4], max_new_tokens=4)
            self.assertEqual(gen.shape, (2, 8))


if __name__ == "__main__":
    print("=" * 88)
    print("RUNNING UNIT TESTS FOR d16_qwen_converter.py")
    print("=" * 88)
    unittest.main(verbosity=2)
