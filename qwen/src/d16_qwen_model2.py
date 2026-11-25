"""Standalone Complex Qwen decoder with D16 wedge coordinates throughout the forward pass.

A hidden width of 896 represents 448 complex channels. The model accepts
real-axis D16 token IDs of shape [batch, sequence]. Use encode_token_ids to
convert tokenizer output. A learned complex embedding maps each token to the
hidden width. Hidden states, KV cache, logits and generated IDs stay in D16.

All parameters store two trainable wedge coordinates and a persistent 16-bit
sector mask via D16Parameter. Embedding lookup, RMSNorm, Attention, and MLP
consume these triples purely through D16ComplexTensor operations.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import repeat
import math
import operator
from threading import Lock
from typing import Callable, Dict, Literal, Optional, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from d16_complex import D16ComplexTensor
from d16_parameter import D16Parameter

IndiffThreshold = float | Literal["auto"]
MLPActivation = Union[
    Literal["sigmoid", "bilinear", "gate", "sum", "rel"],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]


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
class D16QwenConfig:
    """Configuration for standalone native D16QwenModel and D16QwenForCausalLM.

    Hidden size counts real coordinates, two per complex channel.
    """

    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = True
    mlp_bias: bool = False

    indiff_eps: IndiffThreshold = 0.0
    indiff_rel: float = 0.25
    mlp_activation: MLPActivation = "bilinear"
    collect_d16_stats: bool = False

    @property
    def complex_hidden_size(self) -> int:
        return self.hidden_size // 2

    @property
    def head_dim(self) -> int:
        return self.complex_hidden_size // self.num_attention_heads


D16KeyValue = tuple[D16ComplexTensor, D16ComplexTensor]


class D16Linear(nn.Module):
    """Complex-linear projection of D16 activations by learned D16Parameter weights."""

    _MAX_WEIGHT_TILE = 1 << 20

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        _shared_weight: Optional[D16Parameter] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gradient_checkpointing = False
        shape = (out_features, in_features)
        bound = 1 / math.sqrt(2 * in_features)
        self.weight = (
            _shared_weight
            if _shared_weight is not None
            else D16Parameter.random_matrix(shape, std=bound / math.sqrt(3), normal=False)
        )
        self.bias = D16Parameter.zeros((out_features,)) if bias else None

    def _project_tile(
        self,
        hidden_states: D16ComplexTensor,
        weight: D16ComplexTensor,
        bias: Optional[D16ComplexTensor],
        dtype: torch.dtype,
    ) -> D16ComplexTensor:
        z = hidden_states @ weight.to(hidden_states.dtype).transpose(0, 1)
        if bias is not None:
            z = z + bias.to(hidden_states.dtype)
        return z.to(dtype)

    def forward(
        self,
        hidden_states: D16ComplexTensor,
        *,
        eps: IndiffThreshold = 0.0,
        rel: float = 0.25,
    ) -> D16ComplexTensor:
        dtype = torch.promote_types(hidden_states.dtype, self.weight.x.dtype)
        calc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        hs = hidden_states.to(calc_dtype)

        tiles = []
        tile_size = max(1, self._MAX_WEIGHT_TILE // self.in_features)
        bias_tiles = self.bias.value().split(tile_size) if self.bias is not None else repeat(None)
        parameter_tiles = zip(self.weight.value().split(tile_size), bias_tiles)
        recompute = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for parameter_tile in parameter_tiles:
            args = (hs, *parameter_tile, dtype)
            if recompute:
                z = checkpoint(self._project_tile, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                z = self._project_tile(*args)
            tiles.append(z)

        out = tiles[0] if len(tiles) == 1 else D16ComplexTensor.cat(tiles, dim=-1)
        if eps == "auto" or (isinstance(eps, (int, float)) and eps > 0):
            out = out.apply_Manifold_Threshold_L1(eps, rel=rel, dim=-1)
        return out


class D16Embedding(nn.Module):
    """Look up learned D16Parameter complex vectors by exact D16 token keys."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.register_buffer("_key_numbers", torch.arange(vocab_size), persistent=False)
        self.weight = D16Parameter.random_matrix((vocab_size, hidden_size), std=0.02, normal=True)

    def token_keys(self) -> D16ComplexTensor:
        return encode_token_ids(self._key_numbers.unsqueeze(0))[0]

    def forward(self, input_ids: D16ComplexTensor) -> D16ComplexTensor:
        addresses = _token_addresses(input_ids, self.token_keys())
        return self.weight.value()[addresses]


class D16RMSNorm(nn.Module):
    """RMS normalization in the D16 wedge with a native D16Parameter complex gain."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gain = D16Parameter.ones((hidden_size,))
        self.eps = eps

    def forward(self, hidden_states: D16ComplexTensor) -> D16ComplexTensor:
        dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32
        normed = hidden_states.to(dtype).rms_normalize(self.eps)
        return (normed * self.gain.value().to(dtype)).to(hidden_states.dtype)


def _algebraic_inv_root(base: float, n: int) -> float:
    """Solve r**n = 1 / base using only +, -, *, / (monotone Newton from 1.0)."""
    if base <= 0.0:
        raise ValueError("base must be positive")
    if base == 1.0 or n <= 0:
        return 1.0
    inv = base > 1.0
    target = 1.0 / float(base) if inv else float(base)
    r = 1.0
    for _ in range(100):
        p, b, e = 1.0, r, n - 1
        while e > 0:
            if e & 1:
                p *= b
            b *= b
            e >>= 1
        r_next = ((n - 1) * r + target / p) / n
        if abs(r_next - r) <= 1e-16 * r:
            r = r_next
            break
        r = r_next
    return r if inv else (1.0 / r)


@dataclass(frozen=True, eq=False)
class _RotaryTable:
    """The first N rotors and the power for position N, published together."""

    grid: torch.Tensor
    version: Optional[int]
    grid_copy: Optional[torch.Tensor]
    base: float
    table: D16ComplexTensor
    next_power: D16ComplexTensor
    stream: Optional[torch.cuda.Stream] = None
    ready: Optional[torch.cuda.Event] = None


class D16RotaryEmbedding(nn.Module):
    """D16 rotary table generated by doubling a unit Cayley rotor."""

    _version = 2  # Earlier state dicts kept the RoPE grid only in memory.

    def __init__(self, head_dim: int, base: float = 1000000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("frequency_indices", torch.arange(head_dim))
        self._cache: Optional[_RotaryTable] = None
        self._cache_lock = Lock()

    def __getstate__(self):
        # A copied or deserialized module builds its own table, CUDA event and lock.
        state = super().__getstate__()
        state.pop("_cache_lock")
        state.pop("_cache")
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        for name in ("_rotor_table", "_next_power", "_cache_meta"):
            self.__dict__.pop(name, None)
        self._cache_lock = Lock()
        self._cache = None

    def _apply(self, fn, *args, **kwargs):
        with self._cache_lock:
            self._cache = None
            return super()._apply(fn, *args, **kwargs)

    def _cached_table(self, device: torch.device) -> Optional[_RotaryTable]:
        cache = self._cache
        grid = self.frequency_indices
        if cache is None or cache.table.device != device or cache.base != self.base or cache.grid is not grid:
            return None
        if cache.grid_copy is not None:
            if not torch.equal(grid, cache.grid_copy):
                return None
        elif cache.version != grid._version:
            return None
        if cache.ready is not None:
            reader = torch.cuda.current_stream(device)
            if reader != cache.stream:
                # Order the device work and protect storage if another call evicts this table.
                reader.wait_event(cache.ready)
                cache.table.record_stream(reader)
                cache.next_power.record_stream(reader)
        return cache

    def use_qwen_half_head_frequencies(self) -> None:
        """Give corresponding channels in the two halves of a Qwen head the same frequency."""
        if self.head_dim % 2:
            raise ValueError("Qwen half-head RoPE requires an even head_dim")
        with self._cache_lock:
            self._cache = None
            even_indices = torch.arange(0, self.head_dim, 2, device=self.frequency_indices.device)
            self.frequency_indices.copy_(even_indices.repeat(2))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args):
        with self._cache_lock:
            self._cache = None
            if local_metadata.get("version", 1) < 2:
                state_dict.setdefault(prefix + "frequency_indices", self.frequency_indices)
            super()._load_from_state_dict(state_dict, prefix, local_metadata, *args)

    @staticmethod
    def _renormalize_unit(rotor: D16ComplexTensor) -> D16ComplexTensor:
        """Pure algebraic Newton step keeping unit rotors at |z| = 1 across doublings."""
        return rotor * (1.5 - 0.5 * rotor.norm_squared())

    @torch.no_grad()
    def _build_step_rotor(self, grid: torch.Tensor, base: float, device: torch.device) -> D16ComplexTensor:
        r = _algebraic_inv_root(base, self.head_dim)
        indices = grid.tolist()
        max_idx = max(indices, default=0)
        # s_k is the Cayley parameter for a quarter step, so R(1) = C(s_k)^4.
        ladder = [0.125]
        for _ in range(max_idx):
            ladder.append(ladder[-1] * r)
        s = torch.tensor([ladder[i] for i in indices], dtype=torch.float64, device=device)
        return self._renormalize_unit(D16ComplexTensor.from_cayley(s).square().square())

    @torch.no_grad()
    def _ensure_table(self, required_length: int, device: torch.device) -> D16ComplexTensor:
        cache = self._cached_table(device)
        if cache is not None and cache.table.shape[0] >= required_length:
            return cache.table
        with self._cache_lock:
            cache = self._cached_table(device)
            if cache is None:
                grid, base = self.frequency_indices, self.base
                version = None if grid.is_inference() else grid._version
                grid_copy = grid.clone() if version is None else None
                seed = self._build_step_rotor(grid if grid_copy is None else grid_copy, base, device)
                cache = _RotaryTable(
                    grid, version, grid_copy, base,
                    D16ComplexTensor.ones((1, self.head_dim), dtype=torch.float64, device=device),
                    seed.map_tensors(lambda t: t.unsqueeze(0)),
                )
            elif cache.table.shape[0] >= required_length:
                return cache.table

            table, power = cache.table, cache.next_power
            while table.shape[0] < required_length:
                table = D16ComplexTensor.cat((table, table * power), dim=0)
                power = self._renormalize_unit(power.square())
            stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
            ready = stream.record_event() if stream is not None else None
            self._cache = replace(cache, table=table, next_power=power, stream=stream, ready=ready)
            return table

    def forward(
        self,
        hidden_states: D16ComplexTensor,
        position_ids: Optional[torch.Tensor],
        *,
        max_position: Optional[int] = None,
    ) -> D16ComplexTensor:
        """Look up integer positions up to the inclusive max_position, or a contiguous suffix for None."""
        if max_position is not None:
            max_position = operator.index(max_position)
            if max_position < 0:
                raise ValueError("max_position must be nonnegative")
        if position_ids is None:
            length = hidden_states.shape[-2]
            stop = length if max_position is None else max_position + 1
            if stop < length:
                raise ValueError("max_position is too small for the sequence length")
            table = self._ensure_table(stop, hidden_states.device)
            return table[None, stop - length:stop].to(hidden_states.dtype)
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("position_ids must have dtype int32 or int64")
        required = 0
        if position_ids.numel():
            low, high = torch.aminmax(position_ids)
            if low.item() < 0:
                raise ValueError("position_ids must be nonnegative")
            highest = int(high.item())
            if max_position is not None and highest > max_position:
                raise ValueError("position_ids exceed max_position")
            required = highest + 1
        table = self._ensure_table(required, hidden_states.device)
        return table[position_ids].to(hidden_states.dtype)


class D16QwenAttention(nn.Module):
    """Grouped-query attention with a full D16 Hermitian inner product and wedge_softmax."""

    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = config.num_attention_heads // self.num_kv_heads
        self.hidden_size = config.complex_hidden_size
        kv_size = self.num_kv_heads * self.head_dim
        self.indiff_eps: IndiffThreshold = config.indiff_eps
        self.indiff_rel: float = config.indiff_rel
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
        q = query.to(dtype)
        k_h = key.to(dtype).conjugate().map_tensors(lambda t: t.transpose(-1, -2).unsqueeze(2))
        scores = (q @ k_h) * (head_dim ** -0.5)
        allowed = None if attention_mask is None else attention_mask.unsqueeze(2)
        weights = scores.wedge_softmax(allowed, eps=self.indiff_eps, rel=self.indiff_rel)

        v = value.to(dtype).map_tensors(lambda t: t.unsqueeze(2))
        output = weights @ v
        output = output.map_tensors(
            lambda t: t.permute(0, 3, 1, 2, 4).reshape(batch, length, self.hidden_size)
        )
        return self.o_proj(output.to(hidden_states.dtype)), present


class D16QwenMLP(nn.Module):
    """Wedge-native D16 MLP supporting:
    - Variant 1 ('sigmoid'): z.activate('sigmoid')  [executes z * z.sigmoid()]
    - Variant 2 (default, 'bilinear'): z * g  with up_proj and gate_proj
    - Variant 3 ('sum', 'rel', or callable): z.activate(mode) directly over wedge coordinates (x, y)
    """

    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        self.mlp_activation: MLPActivation = config.mlp_activation
        self.up_proj = D16Linear(
            config.complex_hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.gate_proj: Optional[D16Linear] = (
            D16Linear(
                config.complex_hidden_size, config.intermediate_size, bias=config.mlp_bias
            )
            if self.mlp_activation in ("bilinear", "gate")
            else None
        )
        self.down_proj = D16Linear(
            config.intermediate_size, config.complex_hidden_size, bias=config.mlp_bias
        )
        self.indiff_eps: IndiffThreshold = config.indiff_eps
        self.indiff_rel: float = config.indiff_rel
        self.collect_stats = config.collect_d16_stats
        self.register_buffer("state_counts", torch.zeros(18, dtype=torch.long), persistent=False)
        self.total_elements = 0
        self.indifferent_elements = 0

    def reset_stats(self) -> None:
        self.state_counts.zero_()
        self.total_elements = 0
        self.indifferent_elements = 0

    def forward(self, hidden_states: D16ComplexTensor) -> D16ComplexTensor:
        z = self.up_proj(hidden_states, eps=self.indiff_eps, rel=self.indiff_rel)

        if self.collect_stats:
            with torch.no_grad():
                counts = torch.bincount(z.state().reshape(-1), minlength=18)
                self.state_counts += counts
                self.total_elements += z.x.numel()
                self.indifferent_elements += int(counts[16])

        dtype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
        z = z.to(dtype)

        if self.mlp_activation in ("bilinear", "gate"):
            if self.gate_proj is None:
                raise RuntimeError("gate_proj is not initialized for bilinear MLP mode")
            g = self.gate_proj(hidden_states, eps=self.indiff_eps, rel=self.indiff_rel).to(dtype)
            activated = (z * g.sigmoid()) if self.mlp_activation == "gate" else (z * g)
        else:
            activated = z.activate(self.mlp_activation)

        return self.down_proj(activated.to(hidden_states.dtype))


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
        max_pos = total_length - 1 if position_ids is None and total_length else None
        phases = self.rotary_emb(hidden_states, position_ids, max_position=max_pos)
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
    """Backpropagate through loss.x. Optional sampling NLL and perplexity are scalar metrics."""

    logits: D16ComplexTensor
    hidden_states: D16ComplexTensor
    past_key_values: Optional[list[D16KeyValue]] = None
    loss: Optional[D16ComplexTensor] = None
    nll: Optional[torch.Tensor] = None

    @property
    def perplexity(self) -> Optional[torch.Tensor]:
        return None if self.nll is None else self.nll.exp()


class D16QwenForCausalLM(nn.Module):
    """Standalone D16 language model with a complex readout at each vocabulary position."""

    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.model = D16QwenModel(config)
        shared_weight = (
            self.model.embed_tokens.weight if config.tie_word_embeddings else None
        )
        self.lm_head = D16Linear(
            config.complex_hidden_size,
            config.vocab_size,
            bias=False,
            _shared_weight=shared_weight,
        )

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
        *,
        sampling_temperature: Optional[float] = None,
    ) -> D16CausalLMOutput:
        """With labels and a sampling temperature, also evaluate NLL at that temperature."""
        if sampling_temperature is not None and labels is None:
            raise ValueError("Sampling NLL requires labels")
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
            logits, targets = output.logits[..., :-1, :], targets[..., 1:]
            eps, rel = self.config.indiff_eps, self.config.indiff_rel
            output.loss = logits.wedge_cross_entropy(targets, eps=eps, rel=rel)
            if sampling_temperature is not None:
                output.nll = logits.sampling_nll(targets, sampling_temperature, eps=eps, rel=rel)
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
                    probabilities = logits.sampling_probabilities(
                        temperature, eps=self.config.indiff_eps, rel=self.config.indiff_rel
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

    # --- D16 Telemetry & Indifferent Sparsity Controls ---

    def set_indifferent_threshold(
        self, eps: IndiffThreshold, rel: Optional[float] = None
    ) -> None:
        if eps != "auto" and (not isinstance(eps, (int, float)) or float(eps) < 0.0):
            raise ValueError("eps must be a nonnegative float or 'auto'")
        self.config.indiff_eps = eps
        if rel is not None:
            self.config.indiff_rel = float(rel)
        for layer in self.model.layers:
            layer.mlp.indiff_eps = eps
            layer.self_attn.indiff_eps = eps
            if rel is not None:
                layer.mlp.indiff_rel = float(rel)
                layer.self_attn.indiff_rel = float(rel)

    def set_collect_d16_stats(self, enabled: bool = True, reset: bool = True) -> None:
        self.config.collect_d16_stats = enabled
        for layer in self.model.layers:
            layer.mlp.collect_stats = enabled
            if reset:
                layer.mlp.reset_stats()

    def get_d16_state_summary(self) -> Dict[str, Union[torch.Tensor, float, int]]:
        total_counts = torch.zeros(18, dtype=torch.long)
        total_elems = 0
        indiff_elems = 0
        for layer in self.model.layers:
            total_counts += layer.mlp.state_counts.cpu()
            total_elems += layer.mlp.total_elements
            indiff_elems += layer.mlp.indifferent_elements
        sparsity_pct = (indiff_elems / max(total_elems, 1)) * 100.0
        return {
            "state_counts": total_counts,
            "total_elements": total_elems,
            "indifferent_elements": indiff_elems,
            "indifferent_pct": sparsity_pct,
        }


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
