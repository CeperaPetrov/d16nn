"""Complex Qwen decoder with D16 wedge coordinates throughout the forward pass.

A hidden width of 896 represents 448 complex channels. The model accepts
real-axis D16 token IDs of shape [batch, sequence]. Use encode_token_ids to
convert tokenizer output. A learned complex embedding maps each token to the
hidden width. Hidden states, KV cache, logits and generated IDs stay in D16.
Token selection and cross-entropy use the real-axis component in D16.

Complex parameters store two trainable wedge coordinates and a persistent
sector mask via D16Parameter. Embedding lookup and linear kernels consume
these triples purely through D16ComplexTensor arithmetic (+, -, *).
Select weight_storage="cartesian" to load checkpoints from the earlier layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat
import math
from typing import Literal, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from d16_complex import D16ComplexTensor
from d16_parameter import D16Parameter
from d16_qwen_model import (
    D16QwenConfig as _QwenConfig,
    D16QwenForCausalLM as _QwenControls,
    _apply_sector_steering,
)

WeightStorage = Literal["d16", "cartesian"]
IndiffThreshold = float | Literal["auto"]


def _stores_d16(weight_storage: WeightStorage) -> bool:
    if weight_storage not in ("d16", "cartesian"):
        raise ValueError(f"Unknown weight storage: {weight_storage!r}")
    return weight_storage == "d16"


def encode_token_ids(token_ids: torch.Tensor) -> D16ComplexTensor:
    """Encode a batch of integer IDs on the real axis without rounding vocabulary indices."""
    if token_ids.ndim != 2 or token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Token IDs must be an integer tensor of shape [batch, sequence]")
    return D16ComplexTensor.from_real(token_ids.to(torch.float64))


def _token_addresses(
    tokens: D16ComplexTensor, keys: D16ComplexTensor, *, allow_ignore: bool = False
) -> torch.Tensor:
    if not isinstance(tokens, D16ComplexTensor):
        raise TypeError("Expected D16 token IDs. Use encode_token_ids to encode integer IDs.")
    if tokens.x.ndim != 2 or tokens.x.shape != tokens.y.shape or tokens.x.shape != tokens.rotation.shape:
        raise ValueError("D16 token IDs must have shape [batch, sequence]")
    addresses = torch.searchsorted(keys.x, tokens.x.contiguous()).clamp_max(keys.shape[0] - 1)
    found = keys[addresses]
    valid = (tokens.x == found.x) & (tokens.y == found.y) & (tokens.rotation == found.rotation)
    if allow_ignore:
        ignored = tokens.is_negative_real() & (tokens.x == 100) & (tokens.y == 0)
        valid = valid | ignored
        addresses = addresses.masked_fill(ignored, -100)
    if not bool(valid.all()):
        raise ValueError("D16 token IDs must be integers on the real axis in the vocabulary range")
    return addresses


@dataclass
class D16QwenConfig(_QwenConfig):
    """Hidden size counts real coordinates, two per complex channel."""

    weight_storage: WeightStorage = "d16"
    indiff_eps: IndiffThreshold = 0.0
    indiff_rel: float = 0.25

    @property
    def complex_hidden_size(self) -> int:
        return self.hidden_size // 2

    @property
    def head_dim(self) -> int:
        return self.complex_hidden_size // self.num_attention_heads


D16KeyValue = tuple[D16ComplexTensor, D16ComplexTensor]


class D16Linear(nn.Module):
    """Complex-linear projection of D16 activations by learned D16 weights."""

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
                _shared_weight
                if _shared_weight is not None
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
        eps: IndiffThreshold,
        rel: float,
    ) -> D16ComplexTensor:
        wr, wi = weight.map_tensors(lambda t: t.transpose(0, 1)).to(re.dtype).unpack()
        out_re = re.real_matmul(wr).real_sub(im.real_matmul(wi))
        out_im = re.real_matmul(wi).real_add(im.real_matmul(wr))
        z = D16ComplexTensor.pack(out_re, out_im)
        if bias is not None:
            z = z + bias.to(re.dtype)
        if eps == "auto" or (isinstance(eps, (int, float)) and eps > 0):
            z = z.apply_Manifold_Threshold_L1(eps, rel=rel, dim=-1)
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
        eps: IndiffThreshold,
        rel: float,
    ) -> D16ComplexTensor:
        weight = D16ComplexTensor.encode_coordinates(weight_real.to(re.dtype), weight_imag.to(re.dtype))
        bias = (
            None
            if bias_real is None
            else D16ComplexTensor.encode_coordinates(bias_real.to(re.dtype), bias_imag.to(re.dtype))
        )
        return self._project_tile(re, im, weight, bias, dtype, eps, rel)

    def forward(
        self,
        hidden_states: D16ComplexTensor,
        *,
        eps: IndiffThreshold = 0.0,
        rel: float = 0.25,
    ) -> D16ComplexTensor:
        re, im = hidden_states.unpack()
        weight_dtype = self.weight.x.dtype if self._native_weights else self.weight_real.dtype
        dtype = torch.promote_types(hidden_states.dtype, weight_dtype)
        calc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        re, im = re.to(calc_dtype), im.to(calc_dtype)

        tiles = []
        tile_size = max(1, self._MAX_WEIGHT_TILE // self.in_features)
        if self._native_weights:
            project = self._project_tile
            bias_tiles = self.bias.value().split(tile_size) if self.bias is not None else repeat(None)
            parameter_tiles = zip(self.weight.value().split(tile_size), bias_tiles)
        else:
            project = self._project_cartesian_tile
            bias_real = self.bias_real.split(tile_size) if self.bias_real is not None else repeat(None)
            bias_imag = self.bias_imag.split(tile_size) if self.bias_imag is not None else repeat(None)
            parameter_tiles = zip(
                self.weight_real.split(tile_size), self.weight_imag.split(tile_size), bias_real, bias_imag
            )
        recompute = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for parameter_tile in parameter_tiles:
            args = (re, im, *parameter_tile, dtype, eps, rel)
            if recompute:
                z = checkpoint(project, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                z = project(*args)
            tiles.append(z)

        if len(tiles) == 1:
            return tiles[0]
        return D16ComplexTensor.cat(tiles, dim=-1)


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
        return D16ComplexTensor.encode_coordinates(re, im)


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
        hs = hidden_states.to(dtype)
        mean_square = hs.norm_squared().mean(dim=-1, keepdim=True)
        undefined = hidden_states.is_nan().any(dim=-1, keepdim=True)
        if self._native_weights:
            log_gain, _ = self.gain.value().to(dtype).unpack()
            gain = log_gain.real_signed().exp()
        else:
            gain = self.log_gain.to(dtype).exp()
        scale = (mean_square + self.eps).rsqrt() * gain
        return hidden_states * scale.masked_fill(undefined, float("nan"))


class D16RotaryEmbedding(nn.Module):
    """One D16 rotation per complex head channel."""

    def __init__(self, head_dim: int, base: float = 1000000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("frequency_indices", torch.arange(head_dim), persistent=False)

    def forward(
        self, hidden_states: D16ComplexTensor, position_ids: torch.Tensor
    ) -> D16ComplexTensor:
        dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32
        inverse_frequencies = self.base ** (-self.frequency_indices.to(dtype) / self.head_dim)
        phase = position_ids.to(dtype).unsqueeze(-1) * inverse_frequencies
        return D16ComplexTensor.encode_coordinates(phase.cos(), phase.sin()).to(hidden_states.dtype)


class D16QwenAttention(nn.Module):
    """Grouped-query attention with a real-axis Hermitian score and complex values."""

    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = config.num_attention_heads // self.num_kv_heads
        self.hidden_size = config.complex_hidden_size
        kv_size = self.num_kv_heads * self.head_dim
        ws = config.weight_storage
        self.indiff_eps: IndiffThreshold = config.indiff_eps
        self.indiff_rel: float = config.indiff_rel
        self.q_proj = D16Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias, weight_storage=ws)
        self.k_proj = D16Linear(self.hidden_size, kv_size, bias=config.attention_bias, weight_storage=ws)
        self.v_proj = D16Linear(self.hidden_size, kv_size, bias=config.attention_bias, weight_storage=ws)
        self.o_proj = D16Linear(self.hidden_size, self.hidden_size, bias=False, weight_storage=ws)

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
        query = self.q_proj(hidden_states).map_tensors(
            lambda t: t.reshape(batch, length, kv_heads, self.num_groups, head_dim).permute(0, 2, 3, 1, 4)
        )
        key = self.k_proj(hidden_states).map_tensors(
            lambda t: t.reshape(batch, length, kv_heads, head_dim).transpose(1, 2)
        )
        value = self.v_proj(hidden_states).map_tensors(
            lambda t: t.reshape(batch, length, kv_heads, head_dim).transpose(1, 2)
        )

        key_phases = position_embeddings.map_tensors(lambda t: t[:, None, :, :])
        query_phases = key_phases.map_tensors(lambda t: t.unsqueeze(2))
        query = query * query_phases
        key = key * key_phases
        if past_key_value is not None:
            key = D16ComplexTensor.cat((past_key_value[0], key), dim=-2)
            value = D16ComplexTensor.cat((past_key_value[1], value), dim=-2)
        present = (key, value) if use_cache else None

        dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        q_re, q_im = query.to(dtype).unpack()
        k_re, k_im = key.to(dtype).map_tensors(lambda t: t.transpose(-1, -2).unsqueeze(2)).unpack()
        # Full complex Hermitian inner product Q @ K^H packed into the [0°, 22.5°] wedge
        scores_re = q_re.real_matmul(k_re).real_add(q_im.real_matmul(k_im))
        scores_im = q_im.real_matmul(k_re).real_sub(q_re.real_matmul(k_im))
        scores = D16ComplexTensor.pack(scores_re, scores_im) * (head_dim ** -0.5)
        allowed = None if attention_mask is None else attention_mask.unsqueeze(2)
        weights = scores.wedge_softmax(allowed, eps=self.indiff_eps, rel=self.indiff_rel)

        v_re, v_im = value.to(dtype).map_tensors(lambda t: t.unsqueeze(2)).unpack()
        output = D16ComplexTensor.pack(weights.real_matmul(v_re), weights.real_matmul(v_im))
        output = output.map_tensors(
            lambda t: t.permute(0, 3, 1, 2, 4).reshape(batch, length, self.hidden_size)
        )
        return self.o_proj(output.to(hidden_states.dtype)), present


class D16QwenMLP(nn.Module):
    """Apply 0.5 * z^2 * sigmoid(Im(z)) and project the full complex result."""

    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        ws = config.weight_storage
        self.up_proj = D16Linear(
            config.complex_hidden_size, config.intermediate_size, bias=config.mlp_bias, weight_storage=ws
        )
        self.down_proj = D16Linear(
            config.intermediate_size, config.complex_hidden_size, bias=config.mlp_bias, weight_storage=ws
        )
        self.indiff_eps: IndiffThreshold = config.indiff_eps
        self.indiff_rel: float = config.indiff_rel
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
        z = self.up_proj(hidden_states, eps=self.indiff_eps, rel=self.indiff_rel)
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
        _, imag = z.unpack()
        gate = imag.real_sigmoid()
        squared = z * z
        return self.down_proj((squared * (0.5 * gate.x)).to(hidden_states.dtype))


class D16QwenDecoderLayer(nn.Module):
    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        ws = config.weight_storage
        self.input_layernorm = D16RMSNorm(config.complex_hidden_size, config.rms_norm_eps, weight_storage=ws)
        self.self_attn = D16QwenAttention(config)
        self.post_attention_layernorm = D16RMSNorm(
            config.complex_hidden_size, config.rms_norm_eps, weight_storage=ws
        )
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
        hidden_states = hidden_states + attended
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + mlp_output, present


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
        ws = config.weight_storage
        self.embed_tokens = D16Embedding(config.vocab_size, config.complex_hidden_size, weight_storage=ws)
        self.layers = nn.ModuleList(D16QwenDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = D16RMSNorm(config.complex_hidden_size, config.rms_norm_eps, weight_storage=ws)
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
        nn.Module.__init__(self)
        self.config = config
        self.model = D16QwenModel(config)
        ws = config.weight_storage
        shared_weight = (
            self.model.embed_tokens.weight
            if config.tie_word_embeddings and _stores_d16(ws)
            else None
        )
        self.lm_head = D16Linear(
            config.complex_hidden_size,
            config.vocab_size,
            bias=False,
            weight_storage=ws,
            _shared_weight=shared_weight,
        )
        if config.tie_word_embeddings and not _stores_d16(ws):
            self.lm_head.weight_real = self.model.embed_tokens.weight_real
            self.lm_head.weight_imag = self.model.embed_tokens.weight_imag

    def d16_parameters(self) -> list[D16Parameter]:
        """Return all unique D16Parameter modules in the model for D16AdamW."""
        return list(dict.fromkeys(m for m in self.modules() if isinstance(m, D16Parameter)))

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
            output.loss = output.logits[..., :-1, :].wedge_cross_entropy(
                targets[..., 1:], eps=self.config.indiff_eps, rel=self.config.indiff_rel
            )
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
                if do_sample:
                    scaled_logits = logits.to(torch.float32) * (1.0 / max(temperature, 1e-5))
                    probabilities = scaled_logits.wedge_softmax(
                        eps=self.config.indiff_eps, rel=self.config.indiff_rel
                    )
                    addresses = torch.multinomial(probabilities.x, num_samples=1)
                else:
                    addresses = logits.wedge_argmax(
                        eps=self.config.indiff_eps, rel=self.config.indiff_rel
                    )
                generated = D16ComplexTensor.cat((generated, keys[addresses]), dim=-1)
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
