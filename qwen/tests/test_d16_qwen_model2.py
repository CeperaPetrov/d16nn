"""Regression tests for the D16Complex Qwen model."""

import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d16_complex import D16ComplexTensor
from d16_qwen_model2 import D16Linear, D16QwenConfig, D16QwenForCausalLM, D16RMSNorm, encode_token_ids


def tiny_config() -> D16QwenConfig:
    return D16QwenConfig.tiny_test_config(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
    )


class TestD16QwenModel2(unittest.TestCase):
    def test_accepts_real_axis_d16_input(self) -> None:
        torch.manual_seed(7)
        model = D16QwenForCausalLM(tiny_config()).eval()
        real = torch.tensor([[0.0, 2.0, 31.0]])
        prefix = D16ComplexTensor.from_xy(real, torch.zeros_like(real))
        torch.testing.assert_close(prefix.y, torch.zeros_like(prefix.y))
        torch.testing.assert_close(prefix.rotation[0], torch.tensor([0xFFFF, 0x8001, 0x8001]))
        with torch.no_grad():
            output = model(prefix)
            self.assertEqual(output.hidden_states.shape, (1, 3, model.config.complex_hidden_size))
            self.assertFalse(bool(output.logits.is_nan().any()))
            self.assertTrue(bool((output.hidden_states.imag().abs() > 1e-4).any()))
            cached = model.generate(prefix, max_new_tokens=3)
            uncached = model.generate(prefix, max_new_tokens=3, use_cache=False)
            self.assertIsInstance(cached, D16ComplexTensor)
            self.assertEqual(cached.shape, (1, 6))
            torch.testing.assert_close(cached.to_xy(), uncached.to_xy())
            torch.testing.assert_close(cached.x[:, 3].long(), output.logits.real()[:, -1].argmax(-1))
        with self.assertRaises(TypeError):
            model(torch.tensor([[0, 2, 31]]))
        for invalid in (-1.0, 0.5, 32.0):
            x = torch.tensor([[invalid]])
            with self.assertRaises(ValueError):
                model(D16ComplexTensor.from_xy(x, torch.zeros_like(x)))
        with self.assertRaises(ValueError):
            model(D16ComplexTensor.from_xy(torch.ones((1, 1)), torch.ones((1, 1))))
        with self.assertRaises(ValueError):
            model(D16ComplexTensor(torch.ones((1, 1)), torch.zeros((1, 1)), torch.tensor([[1]])))

    def test_d16_messages_across_layers(self) -> None:
        torch.manual_seed(3)
        model = D16QwenForCausalLM(tiny_config()).eval()
        inputs = encode_token_ids(torch.tensor([[1, 4, 7], [2, 5, 8]]))
        called = set()
        hooks = []

        def check_d16(name):
            def hook(module, args, kwargs, output):
                called.add(name)
                message = args[0] if args else kwargs["hidden_states"]
                self.assertIsInstance(message, D16ComplexTensor, name)
                message = output[0] if isinstance(output, tuple) else output
                self.assertIsInstance(message, D16ComplexTensor, name)

            return hook

        modules = dict(model.named_modules())
        names = ["model.embed_tokens", "model.norm", "lm_head"]
        for i in range(model.config.num_hidden_layers):
            layer = f"model.layers.{i}"
            names.extend(
                [layer, f"{layer}.input_layernorm", f"{layer}.post_attention_layernorm"]
            )
            attention = f"{layer}.self_attn"
            mlp = f"{layer}.mlp"
            names.extend([attention, mlp])
            names.extend(f"{attention}.{proj}_proj" for proj in ("q", "k", "v", "o"))
            names.extend(f"{mlp}.{proj}_proj" for proj in ("up", "down"))

        for name in names:
            hooks.append(modules[name].register_forward_hook(check_d16(name), with_kwargs=True))
        try:
            with torch.no_grad():
                output = model(inputs, use_cache=True)
        finally:
            for handle in hooks:
                handle.remove()

        self.assertEqual(called, set(names))
        self.assertIsInstance(output.hidden_states, D16ComplexTensor)
        self.assertEqual(output.hidden_states.shape, (2, 3, model.config.complex_hidden_size))
        self.assertTrue(bool((output.hidden_states.imag().abs() > 1e-4).any()))
        self.assertIsInstance(output.logits, D16ComplexTensor)
        self.assertEqual(output.logits.shape, (2, 3, model.config.vocab_size))
        self.assertTrue(bool((output.logits.imag().abs() > 1e-4).any()))
        self.assertEqual(len(output.past_key_values), model.config.num_hidden_layers)
        for key, value in output.past_key_values:
            self.assertIsInstance(key, D16ComplexTensor)
            self.assertIsInstance(value, D16ComplexTensor)
            self.assertEqual(key.shape, (2, 1, 3, model.config.head_dim))
            self.assertEqual(value.shape, (2, 1, 3, model.config.head_dim))
            self.assertTrue(bool((key.imag().abs() > 1e-4).any()))
            self.assertTrue(bool((value.imag().abs() > 1e-4).any()))

    def test_checkpoint_round_trip_with_tied_weights(self) -> None:
        torch.manual_seed(17)
        config = tiny_config()
        model = D16QwenForCausalLM(config).eval()
        restored = D16QwenForCausalLM(config).eval()
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        self.assertIs(restored.model.embed_tokens.weight_real, restored.lm_head.weight_real)
        self.assertIs(restored.model.embed_tokens.weight_imag, restored.lm_head.weight_imag)

        inputs = encode_token_ids(torch.tensor([[1, 5, 2, 3], [2, 6, 3, 4]]))
        with torch.no_grad():
            reference = model(inputs).logits
            output = restored(inputs).logits
        torch.testing.assert_close(output.to_xy(), reference.to_xy())
        torch.testing.assert_close(output.rotation, reference.rotation)

    def test_rmsnorm_uses_both_components(self) -> None:
        norm = D16RMSNorm(hidden_size=4, eps=1e-6)
        gains = torch.tensor([0.5, 0.25, 0.6, 1.5])
        with torch.no_grad():
            norm.log_gain.copy_(gains.log())
        real = torch.tensor([[3.0, -2.0, 0.0, 1.0]])
        imag = torch.tensor([[4.0, 0.0, 5.0, -1.0]])
        hidden = D16ComplexTensor.from_xy(real, imag)

        output = norm(hidden)
        self.assertIsInstance(output, D16ComplexTensor)
        torch.testing.assert_close(output.rotation, hidden.rotation)
        actual_real, actual_imag = output.to_xy()
        scale = torch.rsqrt((real.square() + imag.square()).mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(actual_real, real * scale * gains)
        torch.testing.assert_close(actual_imag, imag * scale * gains)

    def test_linear_matches_complex_product(self) -> None:
        torch.manual_seed(29)
        config = tiny_config()
        config.attention_bias = False
        model = D16QwenForCausalLM(config)
        projection = model.model.layers[0].self_attn.q_proj
        real = torch.randn(2, 3, config.complex_hidden_size)
        imag = torch.randn_like(real)
        output = projection(D16ComplexTensor.from_xy(real, imag))

        self.assertIsInstance(output, D16ComplexTensor)
        actual_real, actual_imag = output.to_xy()
        linear = torch.nn.functional.linear
        expected_real = linear(real, projection.weight_real) - linear(imag, projection.weight_imag)
        expected_imag = linear(imag, projection.weight_real) + linear(real, projection.weight_imag)
        torch.testing.assert_close(actual_real, expected_real, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual_imag, expected_imag, atol=1e-5, rtol=1e-5)

    def test_loss_backpropagates_to_complex_weights(self) -> None:
        torch.manual_seed(19)
        model = D16QwenForCausalLM(tiny_config())
        inputs = torch.tensor([[1, 5, 2, 3], [2, 6, 3, 4]])
        labels = inputs.clone()
        labels[0, 1] = -100
        output = model(encode_token_ids(inputs), labels=encode_token_ids(labels))
        reference_loss = torch.nn.functional.cross_entropy(
            output.logits.real()[:, :-1].reshape(-1, model.config.vocab_size),
            labels[:, 1:].reshape(-1),
        )
        self.assertIsInstance(output.loss, D16ComplexTensor)
        self.assertFalse(bool(output.loss.is_nan()))
        torch.testing.assert_close(output.loss.x, reference_loss)
        output.loss.x.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        attention = model.model.layers[0].self_attn
        for module in (model.model.embed_tokens, attention.q_proj, attention.v_proj):
            self.assertTrue(bool((module.weight_real.grad != 0).any()))
            self.assertTrue(bool((module.weight_imag.grad != 0).any()))

    def test_projection_gradient_at_exact_zero(self) -> None:
        projection = D16Linear(1, 1, bias=False)
        with torch.no_grad():
            projection.weight_real.zero_()
            projection.weight_imag.zero_()
        source = D16ComplexTensor.from_xy(torch.zeros(1, 1), torch.full((1, 1), 2.0))
        projection(source).x.sum().backward()
        torch.testing.assert_close(projection.weight_real.grad, torch.zeros(1, 1))
        torch.testing.assert_close(projection.weight_imag.grad, torch.full((1, 1), -2.0))

    def test_padding_and_cached_generation(self) -> None:
        torch.manual_seed(23)
        model = D16QwenForCausalLM(tiny_config()).eval()
        input_ids = encode_token_ids(torch.tensor([[0, 0, 1, 5, 2], [2, 4, 8, 6, 3]]))
        padding = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        positions = (padding.cumsum(dim=-1) - 1).clamp_min(0)

        with torch.no_grad():
            full = model(input_ids, attention_mask=padding, position_ids=positions)
            unpadded = model(input_ids[:1, 2:])
            prefix = model(
                input_ids[:, :-1],
                attention_mask=padding[:, :-1],
                position_ids=positions[:, :-1],
                use_cache=True,
            )
            step = model(
                input_ids[:, -1:],
                attention_mask=padding,
                position_ids=positions[:, -1:],
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )
        full_real, full_imag = full.logits.to_xy()
        self.assertTrue(bool(torch.isfinite(full_real).all() & torch.isfinite(full_imag).all()))
        torch.testing.assert_close(
            (full_real[:1, 2:], full_imag[:1, 2:]),
            unpadded.logits.to_xy(),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            step.logits.to_xy(), (full_real[:, -1:], full_imag[:, -1:]), atol=1e-4, rtol=1e-4
        )

        cached = model.generate(input_ids, max_new_tokens=4, use_cache=True)
        uncached = model.generate(input_ids, max_new_tokens=4, use_cache=False)
        self.assertEqual(cached.shape, (2, 9))
        torch.testing.assert_close(cached.to_xy(), uncached.to_xy())


if __name__ == "__main__":
    unittest.main()
