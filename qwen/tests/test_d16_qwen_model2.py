"""Unit, encapsulation, and training tests for d16_qwen_model2.py, d16_parameter.py, and d16_adamw.py."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

_THIS_DIR = Path(__file__).resolve().parent
for _candidate in (
    _THIS_DIR.parent / "src",
    _THIS_DIR.parent / "qwen" / "src",
    _THIS_DIR,
):
    if (_candidate / "d16_qwen_model2.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

import torch

from d16_adamw import D16AdamW
from d16_complex import D16ComplexTensor
from d16_parameter import D16Parameter
from d16_qwen_model2 import D16QwenConfig, D16QwenForCausalLM, encode_token_ids


class TestD16EncapsulationAndAutograd(unittest.TestCase):
    """Проверка строгой инкапсуляции D16ComplexTensor и точности автограда (+, -, *)."""

    def test_strict_encapsulation_no_leaked_internals(self) -> None:
        banned_names = {
            "MASK_ZERO",
            "MASK_NAN",
            "REAL_POSITIVE",
            "REAL_NEGATIVE",
            "reverse_bits_u16",
            "rol_u16",
            "ror_u16",
            "even_anchor",
            "even_anchor_u16",
            "d16_reduced_u16",
            "d16_make_u16",
            "d16_from_xy_u16",
            "d16_to_xy_u16",
            "_chart_table",
            "_chart_parameters",
            "_rotation_basis",
            "_from_chart",
            "_into_chart",
            "_PackComplex",
            "_MultiplyComplex",
            "_AddComplex",
        }
        src_dir = Path(sys.path[0]) if Path(sys.path[0]).is_dir() else _THIS_DIR
        for fname in ("d16_parameter.py", "d16_adamw.py", "d16_qwen_model2.py"):
            fpath = src_dir / fname if (src_dir / fname).is_file() else _THIS_DIR / fname
            tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=fname)
            used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            self.assertEqual(used_names & banned_names, set())

    def test_d16_complex_tensor_autograd_add_mul_sub(self) -> None:
        torch.manual_seed(7)
        ar = torch.randn(16, dtype=torch.float64, requires_grad=True)
        ai = torch.randn(16, dtype=torch.float64, requires_grad=True)
        br = torch.randn(16, dtype=torch.float64, requires_grad=True)
        bi = torch.randn(16, dtype=torch.float64, requires_grad=True)

        za = D16ComplexTensor.encode_coordinates(ar, ai)
        zb = D16ComplexTensor.encode_coordinates(br, bi)
        zc = (za * zb) + (za - zb)
        out_r, out_i = zc.unpack()
        loss_d16 = (out_r.real_signed().square() + 2.0 * out_i.real_signed().square()).sum()
        loss_d16.backward()

        ar_ref = ar.detach().clone().requires_grad_(True)
        ai_ref = ai.detach().clone().requires_grad_(True)
        br_ref = br.detach().clone().requires_grad_(True)
        bi_ref = bi.detach().clone().requires_grad_(True)
        c_r = (ar_ref * br_ref - ai_ref * bi_ref) + (ar_ref - br_ref)
        c_i = (ar_ref * bi_ref + ai_ref * br_ref) + (ai_ref - bi_ref)
        loss_ref = (c_r.square() + 2.0 * c_i.square()).sum()
        loss_ref.backward()

        self.assertLess(float((loss_d16 - loss_ref).abs().item()), 1e-12)
        self.assertLess(float((ar.grad - ar_ref.grad).abs().max().item()), 1e-12)
        self.assertLess(float((ai.grad - ai_ref.grad).abs().max().item()), 1e-12)
        self.assertLess(float((br.grad - br_ref.grad).abs().max().item()), 1e-12)
        self.assertLess(float((bi.grad - bi_ref.grad).abs().max().item()), 1e-12)


class TestD16Model2TrainingAndGeneration(unittest.TestCase):
    """Проверка KV-кэша, генерации и обучения весов D16Parameter через D16AdamW."""

    def test_kv_cache_and_generation_equivalence(self) -> None:
        torch.manual_seed(11)
        cfg = D16QwenConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            weight_storage="d16",
            indiff_eps="auto",
            indiff_rel=0.65,
        )
        model = D16QwenForCausalLM(cfg).eval()
        raw_ids = torch.randint(0, 32, (2, 7), dtype=torch.long)
        d16_ids = encode_token_ids(raw_ids)

        with torch.no_grad():
            full_out = model(d16_ids, use_cache=False)
            prefill = model(d16_ids[:, :-1], use_cache=True)
            step_out = model(d16_ids[:, -1:], past_key_values=prefill.past_key_values, use_cache=True)

            # Verify prefix causal invariance (future tokens must not leak into auto threshold)
            prefix_re, prefix_im = prefill.logits.to_xy()
            full_pref_re, full_pref_im = full_out.logits[:, :-1, :].to_xy()
            self.assertLess(float((full_pref_re - prefix_re).abs().max().item()), 1e-5)
            self.assertLess(float((full_pref_im - prefix_im).abs().max().item()), 1e-5)

            full_re, full_im = full_out.logits[:, -1, :].to_xy()
            step_re, step_im = step_out.logits[:, -1, :].to_xy()
            self.assertLess(float((full_re - step_re).abs().max().item()), 1e-5)
            self.assertLess(float((full_im - step_im).abs().max().item()), 1e-5)

            gen_cached = model.generate(d16_ids[:, :3], max_new_tokens=5, do_sample=False, use_cache=True)
            gen_nocache = model.generate(d16_ids[:, :3], max_new_tokens=5, do_sample=False, use_cache=False)
            self.assertTrue(bool((gen_cached.x == gen_nocache.x).all()))
            self.assertTrue(bool((gen_cached.rotation == gen_nocache.rotation).all()))

    def test_train_native_d16_weights_with_d16_adamw(self) -> None:
        torch.manual_seed(42)
        cfg = D16QwenConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            weight_storage="d16",
        )
        model = D16QwenForCausalLM(cfg)
        optimizer = D16AdamW(model.d16_parameters(), lr=2.5e-2, weight_decay=1e-3)

        raw_tokens = torch.tensor(
            [
                [1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14],
                [3, 7, 11, 15, 19, 23, 27, 31, 4, 8, 12, 16],
            ],
            dtype=torch.long,
        )
        d16_tokens = encode_token_ids(raw_tokens)
        init_rot = model.model.layers[0].mlp.up_proj.weight.rotation.clone()

        first_loss = None
        last_loss = None
        for step in range(30):
            optimizer.zero_grad()
            out = model(input_ids=d16_tokens, labels=d16_tokens)
            assert out.loss is not None
            loss_val = float(out.loss.x.item())
            if step == 0:
                first_loss = loss_val
            out.loss.x.backward()
            optimizer.step(max_grad_norm=1.0)
            last_loss = loss_val

        assert first_loss is not None and last_loss is not None
        self.assertLess(last_loss, first_loss * 0.1)
        changed = int((model.model.layers[0].mlp.up_proj.weight.rotation != init_rot).sum().item())
        self.assertGreater(changed, init_rot.numel() // 2)

        model.eval()
        gen = model.generate(d16_tokens[:, :4], max_new_tokens=8, do_sample=False, use_cache=True)
        self.assertTrue(bool((gen.x.to(torch.long) == raw_tokens).all()))


    def test_d16_activate_modes_and_mlp_variants(self) -> None:
        torch.manual_seed(31)
        re = torch.randn(3, 8, dtype=torch.float64)
        im = torch.randn(3, 8, dtype=torch.float64)
        z = D16ComplexTensor.encode_coordinates(re, im)

        # 1. Verify standalone .sigmoid() compresses radius into r / (1 + r) in [0, 1) and preserves rotation,
        #    while .activate("sigmoid") executes z * z.sigmoid()
        sig_z = z.sigmoid()
        self.assertTrue(bool((sig_z.rotation == z.rotation).all()))
        self.assertTrue(bool((sig_z.norm_squared().sqrt() < 1.0).all()))

        act_sig = z.activate("sigmoid")
        prod_sig = z * z.sigmoid()
        self.assertTrue(
            bool((act_sig.x == prod_sig.x).all() and (act_sig.y == prod_sig.y).all() and (act_sig.rotation == prod_sig.rotation).all())
        )

        # 2. Verify all 3 MLP variants ('sigmoid', 'bilinear', 'sum', 'rel') run forward & backward cleanly
        raw_ids = torch.randint(0, 16, (2, 6), dtype=torch.long)
        d16_ids = encode_token_ids(raw_ids)
        for act_mode in ("sigmoid", "bilinear", "sum", "rel"):
            cfg = D16QwenConfig(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                weight_storage="d16",
                indiff_eps="auto",
                indiff_rel=0.25,
                mlp_activation=act_mode,
            )
            model = D16QwenForCausalLM(cfg)
            opt = D16AdamW(model.d16_parameters(), lr=1e-2)
            out = model(d16_ids, labels=d16_ids)
            out.loss.x.backward()
            opt.step(max_grad_norm=1.0)
            self.assertTrue(bool(torch.isfinite(out.loss.x)))

    def test_rotation_free_wedge_softmax_invariance_and_cutoff(self) -> None:
        torch.manual_seed(23)
        re = torch.randn(4, 12, dtype=torch.float64)
        im = torch.randn(4, 12, dtype=torch.float64)
        z = D16ComplexTensor.encode_coordinates(re, im)

        # Rotating masks across all 16 dihedral operations must not change wedge_softmax or wedge_argmax!
        p_base = z.wedge_softmax(eps="auto", rel=0.30)
        idx_base = z.wedge_argmax(eps="auto", rel=0.30)
        for op_id in range(1, 16):
            z_rot = z.apply_group_op(op_id)
            p_rot = z_rot.wedge_softmax(eps="auto", rel=0.30)
            idx_rot = z_rot.wedge_argmax(eps="auto", rel=0.30)
            self.assertTrue(bool((p_base.x == p_rot.x).all()))
            self.assertTrue(bool((idx_base == idx_rot).all()))

        # Sub-threshold signals are cut off to exactly 0 probability in wedge_softmax
        self.assertGreater(int((p_base.x == 0.0).sum().item()), 0)

    def test_activation_auto_indiff_threshold_and_runtime_switch(self) -> None:
        torch.manual_seed(19)
        cfg = D16QwenConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            weight_storage="d16",
            indiff_eps="auto",
            indiff_rel=0.25,
            collect_d16_stats=True,
        )
        model = D16QwenForCausalLM(cfg).eval()
        raw_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
        d16_ids = encode_token_ids(raw_ids)

        with torch.no_grad():
            model.set_collect_d16_stats(True, reset=True)
            _ = model(d16_ids)
            sp_25 = model.get_d16_state_summary()["indifferent_pct"]
            self.assertGreater(sp_25, 0.0)

            model.set_indifferent_threshold("auto", rel=0.40)
            model.set_collect_d16_stats(True, reset=True)
            _ = model(d16_ids)
            sp_40 = model.get_d16_state_summary()["indifferent_pct"]
            self.assertGreater(sp_40, sp_25)

            model.set_indifferent_threshold(0.0)
            model.set_collect_d16_stats(True, reset=True)
            _ = model(d16_ids)
            sp_0 = model.get_d16_state_summary()["indifferent_pct"]
            self.assertEqual(sp_0, 0.0)

    def test_d16_adamw_manual_and_auto_indiff_eps(self) -> None:
        for mode_eps in (0.04, "auto"):
            torch.manual_seed(42)
            cfg = D16QwenConfig(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                weight_storage="d16",
                indiff_eps="auto",
                indiff_rel=0.25,
            )
            model = D16QwenForCausalLM(cfg)
            optimizer = D16AdamW(
                model.d16_parameters(),
                lr=2.5e-2,
                weight_decay=1e-3,
                indiff_eps=mode_eps,
                indiff_rel=0.30,
            )
            raw_tokens = torch.tensor(
                [
                    [1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14],
                    [3, 7, 11, 15, 19, 23, 27, 31, 4, 8, 12, 16],
                ],
                dtype=torch.long,
            )
            d16_tokens = encode_token_ids(raw_tokens)

            for _ in range(30):
                optimizer.zero_grad()
                out = model(input_ids=d16_tokens, labels=d16_tokens)
                out.loss.x.backward()
                optimizer.step(max_grad_norm=1.0)

            sd = optimizer.state_dict()
            self.assertEqual(sd["indiff_eps"], mode_eps)
            optimizer.load_state_dict(sd)
            # Verify restored optimizer can execute a subsequent step without stride-0 copy errors
            optimizer.zero_grad()
            out_after = model(input_ids=d16_tokens, labels=d16_tokens)
            out_after.loss.x.backward()
            optimizer.step(max_grad_norm=1.0)

            up_w = model.model.layers[0].mlp.up_proj.weight.value()
            indiff_count = int(up_w.is_zero().sum().item())
            self.assertGreater(indiff_count, 0)

            model.eval()
            gen = model.generate(d16_tokens[:, :4], max_new_tokens=8, do_sample=False, use_cache=True)
            self.assertTrue(bool((gen.x.to(torch.long) == raw_tokens).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
