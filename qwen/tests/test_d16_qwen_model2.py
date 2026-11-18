"""Unit, encapsulation, and training tests for d16_qwen_model2.py, d16_parameter.py, and d16_adamw.py."""
from __future__ import annotations

import ast
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

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
from d16_qwen_importer import D16QwenImportConfig, import_qwen_weights, load_d16_checkpoint, save_d16_checkpoint
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
        for fname in ("d16_parameter.py", "d16_adamw.py", "d16_qwen_model2.py", "d16_qwen_importer.py"):
            fpath = src_dir / fname if (src_dir / fname).is_file() else _THIS_DIR / fname
            tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=fname)
            used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            self.assertEqual(used_names & banned_names, set())
            imported_modules = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            } | {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            self.assertNotIn("d16_qwen_model", imported_modules)

    def test_d16_complex_tensor_autograd_add_mul_sub(self) -> None:
        torch.manual_seed(7)
        ar = torch.randn(16, dtype=torch.float64, requires_grad=True)
        ai = torch.randn(16, dtype=torch.float64, requires_grad=True)
        br = torch.randn(16, dtype=torch.float64, requires_grad=True)
        bi = torch.randn(16, dtype=torch.float64, requires_grad=True)

        za = D16ComplexTensor.pack(ar, ai)
        zb = D16ComplexTensor.pack(br, bi)
        zc = (za * zb) + (za - zb)
        out_r, out_i = zc.unpack()
        self.assertIsInstance(out_r, torch.Tensor)
        self.assertIsInstance(out_i, torch.Tensor)
        loss_d16 = (out_r.square() + 2.0 * out_i.square()).sum()
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
        z_cpu = zc.detach().cpu()
        self.assertEqual(z_cpu.device.type, "cpu")
        self.assertTrue(hasattr(z_cpu, "cuda"))


    def test_d16_complex_tensor_matmul_operator(self) -> None:
        torch.manual_seed(13)
        ar = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        ai = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        br = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
        bi = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)

        za = D16ComplexTensor.pack(ar, ai)
        zb = D16ComplexTensor.pack(br, bi)
        zc = za @ zb
        out_r, out_i = zc.unpack()
        loss_d16 = (out_r.square() + 2.0 * out_i.square()).sum()
        loss_d16.backward()

        ar_ref = ar.detach().clone().requires_grad_(True)
        ai_ref = ai.detach().clone().requires_grad_(True)
        br_ref = br.detach().clone().requires_grad_(True)
        bi_ref = bi.detach().clone().requires_grad_(True)
        c_r = ar_ref @ br_ref - ai_ref @ bi_ref
        c_i = ar_ref @ bi_ref + ai_ref @ br_ref
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


    @staticmethod
    def _qwen_state_dict() -> dict[str, torch.Tensor]:
        torch.manual_seed(7)
        V, H, I, L = 32, 16, 24, 2
        sd = {
            "model.embed_tokens.weight": torch.randn(V, H) * 0.02,
            "model.norm.weight": torch.ones(H),
        }
        for l_idx in range(L):
            p = f"model.layers.{l_idx}"
            sd[f"{p}.input_layernorm.weight"] = torch.ones(H)
            sd[f"{p}.post_attention_layernorm.weight"] = torch.ones(H)
            sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(H, H) * 0.1
            sd[f"{p}.self_attn.q_proj.bias"] = torch.zeros(H)
            sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(H // 2, H) * 0.1
            sd[f"{p}.self_attn.k_proj.bias"] = torch.zeros(H // 2)
            sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(H // 2, H) * 0.1
            sd[f"{p}.self_attn.v_proj.bias"] = torch.zeros(H // 2)
            sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(H, H) * 0.1

            up = torch.randn(I, H) * 0.1
            gate = torch.randn(I, H) * 0.1
            for q in range(3):
                base = q * 4
                u0, g0 = up[base].clone(), gate[base].clone()
                up[base + 1], gate[base + 1] = -g0, u0
                up[base + 2], gate[base + 2] = (u0 - g0) * 0.70710678, (u0 + g0) * 0.70710678
                up[base + 3], gate[base + 3] = u0, -g0
            sd[f"{p}.mlp.up_proj.weight"] = up
            sd[f"{p}.mlp.gate_proj.weight"] = gate
            sd[f"{p}.mlp.down_proj.weight"] = torch.randn(H, I) * 0.1
        return sd

    def test_1to1_qwen_importer_quartets_and_unpaired_weights(self) -> None:
        sd = self._qwen_state_dict()
        cfg = D16QwenImportConfig.from_state_dict(sd, num_attention_heads=4, indiff_eps="auto", indiff_rel=0.30)
        d16_model, report = import_qwen_weights(sd, config=cfg, min_orbit_score=0.60)

        # Verify 1-to-1 coordinates and Re (in x) vs Im (in y) placement
        up_orig = sd["model.layers.0.mlp.up_proj.weight"]
        up_d16 = d16_model.model.layers[0].mlp.up_proj.weight.value()
        re_unpacked, im_unpacked = up_d16.unpack()
        im_mask = report.layer_reports[0].is_im_neuron
        self.assertGreater(int(im_mask.sum().item()), 0)
        self.assertGreater(int((~im_mask).sum().item()), 0)
        self.assertLess(float((re_unpacked[~im_mask] - up_orig[~im_mask]).abs().max().item()), 1e-6)
        self.assertLess(float(im_unpacked[~im_mask].abs().max().item()), 1e-6)
        self.assertLess(float(re_unpacked[im_mask].abs().max().item()), 1e-6)
        self.assertLess(float((im_unpacked[im_mask] - up_orig[im_mask]).abs().max().item()), 1e-6)

        # Verify fast save_d16_checkpoint and load_d16_checkpoint round-trip
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "d16_ckpt"
            save_d16_checkpoint(d16_model, ckpt_dir, report=report)
            restored_model, restored_rep = load_d16_checkpoint(ckpt_dir, return_report=True)
            self.assertIsNotNone(restored_rep)
            r_up = restored_model.model.layers[0].mlp.up_proj.weight.value()
            self.assertTrue(bool((r_up.x == up_d16.x).all()))
            self.assertTrue(bool((r_up.y == up_d16.y).all()))
            self.assertTrue(bool((r_up.rotation == up_d16.rotation).all()))

    def test_import_checks_tensor_shapes_before_model_allocation(self) -> None:
        sd = self._qwen_state_dict()
        cfg = D16QwenImportConfig.from_state_dict(sd, num_attention_heads=4)
        sd.update({
            "model.embed_tokens.weight": torch.zeros(4097, 16),
            "model.norm.weight": torch.zeros(1, 16),
            "model.layers.1.self_attn.v_proj.bias": torch.zeros(7),
            "model.layers.1.mlp.down_proj.weight": torch.zeros(24, 16),
            "lm_head.weight": torch.zeros(4099, 15),
        })
        with patch("d16_qwen_importer.D16QwenForCausalLM") as allocate:
            with self.assertRaises(ValueError) as exc:
                import_qwen_weights(sd, config=replace(cfg, vocab_size=4099), verbose=False)
            allocate.assert_not_called()
        for name, expected, actual in (
            ("model.embed_tokens.weight", (4099, 16), (4097, 16)),
            ("model.norm.weight", (16,), (1, 16)),
            ("model.layers.1.self_attn.v_proj.bias", (8,), (7,)),
            ("model.layers.1.mlp.down_proj.weight", (16, 24), (24, 16)),
            ("lm_head.weight", (4099, 16), (4099, 15)),
        ):
            self.assertIn(f"{name}: expected {expected}, got {actual}", str(exc.exception))

    def test_import_detects_partial_biases_and_missing_gate_or_layer(self) -> None:
        sd = {key.removeprefix("model."): value for key, value in self._qwen_state_dict().items()}
        for key, weight in list(sd.items()):
            if ".mlp." in key:
                sd[key.replace(".weight", ".bias")] = torch.zeros(weight.shape[0])
        complete, _ = import_qwen_weights(sd, mlp_activation="gate", verbose=False)
        self.assertIsNotNone(complete.model.layers[1].mlp.gate_proj.bias)
        no_bias = {key: value for key, value in sd.items() if not key.endswith(".bias")}
        no_bias["lm_head.weight"] = -sd["embed_tokens.weight"]
        untied, _ = import_qwen_weights(no_bias, verbose=False)
        self.assertIsNone(untied.model.layers[0].self_attn.q_proj.bias)
        self.assertIsNot(untied.lm_head.weight, untied.model.embed_tokens.weight)
        torch.testing.assert_close(untied.lm_head.weight.value().unpack()[0], no_bias["lm_head.weight"])
        with self.assertRaisesRegex(ValueError, "missing lm_head.weight"):
            import_qwen_weights({k: v for k, v in no_bias.items() if k != "lm_head.weight"}, config=untied.config)
        for missing in (
            "layers.0.self_attn.q_proj.bias",
            "layers.0.mlp.up_proj.bias",
            "layers.1.mlp.gate_proj.weight",
        ):
            with self.subTest(missing=missing):
                incomplete = dict(sd)
                del incomplete[missing]
                with patch("d16_qwen_importer.D16QwenForCausalLM") as allocate:
                    with self.assertRaisesRegex(ValueError, f"missing model.{missing}"):
                        import_qwen_weights(incomplete, mlp_activation="gate", verbose=False)
                    allocate.assert_not_called()
        with self.assertRaisesRegex(ValueError, "missing model.layers.2.input_layernorm.weight"):
            import_qwen_weights(sd, num_hidden_layers=3, verbose=False)

    def test_mixed_precision_import_is_independent_of_tile_size(self) -> None:
        sd = self._qwen_state_dict()
        q_key = "model.layers.0.self_attn.q_proj.weight"
        sd[q_key] = 1 + 0.004 * torch.randn_like(sd[q_key])
        cfg = D16QwenImportConfig.from_state_dict(sd, num_attention_heads=4)

        def imported(source, mode, tile_size):
            with patch("d16_qwen_importer._PACK_ROW_TILE", tile_size):
                model, _ = import_qwen_weights(
                    source, config=cfg, dtype=torch.bfloat16, second_coord_mode=mode, verbose=False,
                )
            return model.state_dict()

        for mode in ("d16_sat_det", "d16_sat_rope_gate"):
            with self.subTest(mode=mode):
                whole = imported(sd, mode, 4096)
                tiled = imported(sd, mode, 3)
                for key in whole:
                    torch.testing.assert_close(tiled[key], whole[key], rtol=0, atol=0, msg=key)
                rounded = {**sd, q_key: sd[q_key].to(torch.bfloat16)}
                reference = imported(rounded, mode, 4096)
                torch.testing.assert_close(tiled[q_key + ".y"], reference[q_key + ".y"], rtol=0, atol=0)

    def test_sampling_nll_for_cut_uniform_and_ignored_targets(self) -> None:
        scores = torch.tensor([[0.1, 1.0], [0.1, 0.4], [float("nan"), 0.1]], requires_grad=True)
        z = D16ComplexTensor.from_real(scores)
        target = torch.tensor([0])
        loss = z[:1].wedge_cross_entropy(target, eps=0.5)
        loss.x.backward()
        self.assertTrue(math.isfinite(loss.x.item()))
        self.assertLess(scores.grad[0, 0].item(), 0)
        self.assertEqual(z[:1].sampling_nll(target, eps=0.5).item(), math.inf)
        self.assertEqual(z[:1].sampling_nll(torch.tensor([1]), eps=0.5).item(), 0.0)
        nll = z.sampling_nll(torch.tensor([-100, 1, -100]), eps=0.5)
        self.assertFalse(nll.requires_grad)
        self.assertAlmostEqual(nll.item(), math.log(2))
        self.assertTrue(z.sampling_nll(torch.tensor([-100, -100, 1]), eps=0.5).isnan().item())
        self.assertTrue(z.sampling_nll(torch.full((3,), -100), eps=0.5).isnan().item())

    def test_reported_perplexity_matches_generate_at_requested_temperature(self) -> None:
        torch.manual_seed(23)
        cfg = D16QwenConfig(vocab_size=8, hidden_size=8, intermediate_size=12, num_hidden_layers=1,
                            num_attention_heads=2, num_key_value_heads=1)
        model = D16QwenForCausalLM(cfg).eval()
        ids = encode_token_ids(torch.tensor([[2, 5, 3]]))
        labels = encode_token_ids(torch.tensor([[-100, -100, 3]]))
        self.assertIsNone(model(ids, labels=labels).perplexity)
        for temperature in (0.5, 1.5):
            with self.subTest(temperature=temperature), patch("torch.multinomial", wraps=torch.multinomial) as draw:
                model.generate(ids[:, :-1], max_new_tokens=1, do_sample=True, temperature=temperature)
                sample_probability = draw.call_args.args[0][0, 3].item()
                out = model(ids, labels=labels, sampling_temperature=temperature)
                self.assertAlmostEqual(out.nll.item(), -math.log(sample_probability), places=5)
                self.assertAlmostEqual(out.perplexity.item(), 1 / sample_probability, places=5)

    def test_d16_adamw_resume_from_packed_weights_and_legacy_masks(self) -> None:
        packed = D16ComplexTensor.pack(torch.tensor([1., -2., 0., 3.]), torch.tensor([2., 1., 0., -4.]))

        def update(parameter, optimizer, step):
            optimizer.zero_grad()
            real, imag = parameter.value().unpack()
            (step * real + imag).sum().backward()
            optimizer.step()

        for mask_dtype in (torch.int32, torch.int64):
            with self.subTest(mask_dtype=mask_dtype):
                p = D16Parameter(packed)
                opt = D16AdamW([p], lr=0.02)
                update(p, opt, 1)
                q = D16Parameter.zeros(packed.shape)
                q.load_state_dict(p.state_dict())
                restored = D16AdamW([q])
                saved = opt.state_dict()
                saved["moments"][0]["first_rotation"] = saved["moments"][0]["first_rotation"].to(mask_dtype)
                restored.load_state_dict(saved)
                update(p, opt, 2)
                update(q, restored, 2)
                torch.testing.assert_close(q.state_dict(), p.state_dict(), rtol=0, atol=0)
                torch.testing.assert_close(restored.state_dict()["moments"], opt.state_dict()["moments"], rtol=0, atol=0)
                for invalid in (torch.full(packed.shape, 65536), torch.zeros(packed.shape)):
                    saved["moments"][0]["first_rotation"] = invalid
                    with self.assertRaisesRegex(ValueError, "sector mask"):
                        restored.load_state_dict(saved)

    def test_single_branch_d16_rmsnorm_unit_variance_and_complex_gain(self) -> None:
        torch.manual_seed(17)
        re = torch.randn(2, 5, 16, dtype=torch.float64) * 4.0
        im = torch.randn(2, 5, 16, dtype=torch.float64) * 4.0
        z = D16ComplexTensor.pack(re, im)

        normed = z.rms_normalize(eps=1e-12)
        mean_sq = normed.norm_squared().mean(dim=-1)
        self.assertLess(float((mean_sq - 1.0).abs().max().item()), 1e-10)
        self.assertTrue(bool((normed.rotation == z.rotation).all()))

    def test_d16_activate_modes_and_mlp_variants(self) -> None:
        torch.manual_seed(31)
        re = torch.randn(3, 8, dtype=torch.float64)
        im = torch.randn(3, 8, dtype=torch.float64)
        z = D16ComplexTensor.pack(re, im)

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
        z = D16ComplexTensor.pack(re, im)

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
