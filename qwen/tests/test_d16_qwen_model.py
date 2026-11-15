"""Unit and training-from-scratch tests for d16_qwen_model.py.

Can be executed via:
    python tests/test_d16_qwen_model.py
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

_THIS_DIR = Path(__file__).resolve().parent
for _candidate in (
    _THIS_DIR.parent / "src",
    _THIS_DIR.parent / "qwen" / "src",
    _THIS_DIR,
):
    if (_candidate / "d16_qwen_model.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

import torch
import torch.nn.functional as F

from d16_qwen_model import (
    D16QwenConfig,
    D16QwenForCausalLM,
    D16QwenMLP,
    D16RMSNorm,
    D16RotaryEmbedding,
)


class TestD16QwenComponents(unittest.TestCase):
    """Тестирование узлов нативной архитектуры: RMSNorm, D16-RoPE и D16QwenMLP."""

    def test_rmsnorm_unit_variance(self) -> None:
        torch.manual_seed(0)
        norm = D16RMSNorm(hidden_size=64, eps=1e-6)
        x = torch.randn(4, 16, 64) * 5.0
        y = norm(x)
        rms = y.detach().pow(2).mean(dim=-1).sqrt()
        self.assertLess(float((rms - 1.0).abs().max().item()), 1e-5)

    def test_d16_rotary_embedding_matches_exact_complex_rotation(self) -> None:
        torch.manual_seed(1)
        rope = D16RotaryEmbedding(head_dim=16, max_position_embeddings=128, base=10000.0)
        q = torch.randn(2, 4, 12, 16, dtype=torch.float64)
        k = torch.randn(2, 2, 12, 16, dtype=torch.float64)
        pos_ids = torch.arange(12).unsqueeze(0).expand(2, -1)

        cos, sin = rope(q, pos_ids)
        q_rot, k_rot = D16RotaryEmbedding.apply_d16_rotary_pos_emb(q, k, cos, sin)

        cos_b, sin_b = cos.unsqueeze(1), sin.unsqueeze(1)
        q_re, q_im = q[..., :8], q[..., 8:]
        q_ref = torch.cat([q_re * cos_b - q_im * sin_b, q_re * sin_b + q_im * cos_b], dim=-1)
        k_re, k_im = k[..., :8], k[..., 8:]
        k_ref = torch.cat([k_re * cos_b - k_im * sin_b, k_re * sin_b + k_im * cos_b], dim=-1)

        self.assertLess(float((q_rot - q_ref).abs().max().item()), 1e-13)
        self.assertLess(float((k_rot - k_ref).abs().max().item()), 1e-13)
        self.assertLess(float((q_rot.norm(dim=-1) - q.norm(dim=-1)).abs().max().item()), 1e-13)

    def test_native_d16_mlp_wedge_branching_and_gradients(self) -> None:
        torch.manual_seed(2)
        cfg = D16QwenConfig.tiny_test_config()
        mlp = D16QwenMLP(cfg).double()
        x = torch.randn(3, 10, cfg.hidden_size, dtype=torch.float64, requires_grad=True)

        out = mlp(x)
        self.assertEqual(out.shape, x.shape)

        # Проверяем, что секторное ветвление _eval_d16(z) внутри клина [0°, 22.5°]
        # (x*y на осевых секторах и 0.5*(x^2 - y^2) на диагональных) дает точность < 1e-13
        with torch.no_grad():
            L = mlp.up_proj(x)
            E = mlp.gate_proj(x)
            ref_out = mlp.down_proj(L * F.silu(E))
            self.assertLess(float((out.detach() - ref_out).abs().max().item()), 1e-13)

        out.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(mlp.up_proj.weight.grad)
        self.assertIsNotNone(mlp.gate_proj.weight.grad)
        self.assertIsNotNone(mlp.down_proj.weight.grad)

    def test_bitwise_orbit_links_preserve_object_count_and_rotate_mask(self) -> None:
        torch.manual_seed(5)
        cfg = D16QwenConfig.tiny_test_config(hidden_size=32, intermediate_size=16)
        mlp = D16QwenMLP(cfg).double()

        # Задаем нейрон 1 как 45°-партнера Адамара (op_id=8: rol на 2 бита) нейрона 0
        s = 2.0 ** -0.5
        with torch.no_grad():
            u0 = mlp.up_proj.weight[0].clone()
            g0 = mlp.gate_proj.weight[0].clone()
            mlp.up_proj.weight[1].copy_(s * (u0 - g0))
            mlp.gate_proj.weight[1].copy_(s * (u0 + g0))

        x = torch.randn(4, 6, 32, dtype=torch.float64)
        with torch.no_grad():
            out_independent = mlp(x)
            # Теперь включаем побитовую связь орбиты: объект #1 берет (x, y) от #0 и сдвигает только маску rotation!
            mlp.set_orbit_links(
                partner_idx=torch.tensor([1]),
                anchor_idx=torch.tensor([0]),
                op_ids=torch.tensor([8]),
            )
            z_objs = mlp.forward_d16_objects(x)
            out_linked = mlp(x)

        # Количество объектов D16Complex осталось ровно 16!
        self.assertEqual(z_objs.shape, (4, 6, 16))
        # Координаты (x, y) у объектов #0 и #1 в клине [0°, 22.5°] идентичны, а rotation сдвинут на 2 бита:
        self.assertLess(float((z_objs.x[..., 0] - z_objs.x[..., 1]).abs().max().item()), 1e-14)
        self.assertLess(float((z_objs.y[..., 0] - z_objs.y[..., 1]).abs().max().item()), 1e-14)
        self.assertLess(float((out_independent - out_linked).abs().max().item()), 1e-13)


class TestD16QwenInferenceCacheAndTelemetry(unittest.TestCase):
    """Тестирование полной модели: KV-кэш, генерация, телеметрия 18 состояний и управление маской."""

    def test_kv_cache_equivalence_and_generation(self) -> None:
        torch.manual_seed(3)
        cfg = D16QwenConfig.tiny_test_config()
        model = D16QwenForCausalLM(cfg).eval()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            full_out = model(input_ids, use_cache=False)

            prefill = model(input_ids[:, :-1], use_cache=True)
            step_out = model(input_ids[:, -1:], past_key_values=prefill.past_key_values, use_cache=True)

            self.assertLess(float((full_out.logits[:, -1, :] - step_out.logits[:, -1, :]).abs().max().item()), 1e-4)

            gen_cached = model.generate(input_ids[:, :4], max_new_tokens=6, do_sample=False, use_cache=True)
            gen_nocache = model.generate(input_ids[:, :4], max_new_tokens=6, do_sample=False, use_cache=False)
            self.assertTrue(bool((gen_cached == gen_nocache).all()))
            self.assertEqual(gen_cached.shape, (2, 10))

    def test_state_telemetry_indifferent_sparsity_and_steering(self) -> None:
        torch.manual_seed(4)
        cfg = D16QwenConfig.tiny_test_config()
        model = D16QwenForCausalLM(cfg).eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 12))

        model.set_collect_d16_stats(True, reset=True)
        model.set_indifferent_threshold(0.0)
        _ = model(input_ids)
        summary_zero = model.get_d16_state_summary()
        expected_total = 2 * 12 * cfg.num_hidden_layers * cfg.intermediate_size
        self.assertEqual(summary_zero["total_elements"], expected_total)
        self.assertEqual(int(summary_zero["state_counts"].sum()), expected_total)

        model.set_collect_d16_stats(True, reset=True)
        model.set_indifferent_threshold(0.25)
        _ = model(input_ids)
        summary_eps = model.get_d16_state_summary()
        self.assertGreater(summary_eps["indifferent_pct"], 0.0)
        self.assertEqual(int(summary_eps["state_counts"][16]), summary_eps["indifferent_elements"])

        model.set_indifferent_threshold(0.0)
        with torch.no_grad():
            out_id = model(input_ids).logits[:, -1, :]
            model.set_sector_steering(mode="active_swap_le")
            out_steered = model(input_ids).logits[:, -1, :]
            model.set_sector_steering(mode="identity")
        self.assertGreater(float((out_id - out_steered).abs().max().item()), 1e-4)
        self.assertTrue(bool(torch.isfinite(out_steered).all()))


class TestD16QwenTrainingFromScratch(unittest.TestCase):
    """Проверка обучения нативной модели D16Qwen с нуля (без внешних весов)."""

    def test_train_from_scratch_native_d16(self) -> None:
        torch.manual_seed(42)
        cfg = D16QwenConfig.tiny_test_config(
            vocab_size=32,
            hidden_size=48,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model = D16QwenForCausalLM(cfg)
        model.train()

        pattern = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 1, 5, 9, 13]], dtype=torch.long)
        batch = pattern.repeat(4, 1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-2, weight_decay=1e-4)
        initial_loss = None
        final_loss = None

        for step in range(45):
            optimizer.zero_grad()
            out = model(input_ids=batch, labels=batch)
            loss = out.loss
            assert loss is not None
            if step == 0:
                initial_loss = float(loss.item())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        assert initial_loss is not None and final_loss is not None
        self.assertLess(final_loss, initial_loss * 0.05)
        self.assertLess(final_loss, 0.05)

        model.eval()
        prompt = pattern[:, :4]
        generated = model.generate(prompt, max_new_tokens=8, do_sample=False)
        self.assertTrue(bool((generated == pattern[:, :12]).all()))


if __name__ == "__main__":
    print("=" * 88)
    print("RUNNING UNIT & TRAINING-FROM-SCRATCH TESTS FOR NATIVE d16_qwen_model.py")
    print("=" * 88)
    unittest.main(verbosity=2)
