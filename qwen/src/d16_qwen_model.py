"""Standalone D16Qwen Transformer Architecture built on D16ComplexTensor.

This module implements the complete Qwen2.5-compatible Causal Language Model
(D16QwenConfig, D16RMSNorm, D16RotaryEmbedding, D16QwenAttention, D16QwenMLP,
D16QwenDecoderLayer, D16QwenModel, D16QwenForCausalLM) from scratch in pure
PyTorch using D16ComplexTensor from d16_complex.py, with zero dependency on
external model libraries.

Key architectural features:
1. RoPE in D16QwenAttention is computed as complex multiplication inside the
   [0°, 22.5°] fundamental wedge: z_rot = z_qk * z_freq via D16ComplexTensor.
2. Non-linearity in D16QwenMLP operates explicitly on D16ComplexTensor objects
   (z = D16ComplexTensor.from_xy(L, E), z2 = z.square() inside [0°, 22.5°]),
   supporting both:
   - "d16_swiglu": 1-to-1 Qwen2.5 compatible mode (0.5 * Im(z^2) * sigmoid(Im(z)))
   - "d16_bipolar": compact native complex mode outputting both (Re(z^2), Im(z^2))
     with 18-state sector gains.
3. Full Straight-Through Estimator (STE) autograd bridge so the network can be
   trained from scratch with standard optimizers while executing exact D16Complex
   wedge arithmetic on the forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from d16_complex import (
    D16ComplexTensor,
    MASK_NAN,
    MASK_ZERO,
    STATE_INDIFFERENT,
    apply_group_op_u16,
    rol_u16,
    reverse_bits_u16,
    state_u16,
)


# =====================================================================
# 1. CONFIGURATION (QWEN2.5 COMPATIBLE + D16 EXTENSIONS)
# =====================================================================

@dataclass
class D16QwenConfig:
    """Configuration class for D16QwenModel and D16QwenForCausalLM."""

    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = True
    mlp_bias: bool = False

    # D16Complex specific parameters
    mlp_mode: str = "d16_swiglu"  # "d16_swiglu" (1-to-1 Qwen) or "d16_bipolar" (native Re/Im z^2)
    indiff_eps: float = 0.0       # Threshold for transitioning weak signals to 0xFFFF (State 16: Indifferent)
    collect_d16_stats: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def tiny_test_config(
        cls,
        vocab_size: int = 256,
        hidden_size: int = 64,
        intermediate_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        mlp_mode: str = "d16_swiglu",
        indiff_eps: float = 0.0,
    ) -> "D16QwenConfig":
        """Compact configuration for fast unit tests and training from scratch."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            attention_bias=True,
            mlp_bias=False,
            mlp_mode=mlp_mode,
            indiff_eps=indiff_eps,
        )


# =====================================================================
# 2. DIFFERENTIABLE BRIDGE (STE) FOR D16ComplexTensor ARITHMETIC
# =====================================================================

def d16_complex_square_ste(
    logic: torch.Tensor,
    emotion: torch.Tensor,
    indiff_eps: Union[float, torch.Tensor] = 0.0,
    steer_mode: str = "identity",
    steer_shift: int = 0,
    steer_channels: Optional[torch.Tensor] = None,
) -> Tuple[D16ComplexTensor, D16ComplexTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute z and z^2 = z * z in the [0°, 22.5°] wedge with STE autograd support.

    On the forward pass, values are computed strictly through D16ComplexTensor
    inside the [0°, 22.5°] fundamental wedge (including 0xFFFF Indifferent gating
    and optional pure-mask steering). On the backward pass (when training),
    gradients flow smoothly through the exact algebraic identities.
    """
    with torch.no_grad():
        z = D16ComplexTensor.from_xy(logic.detach(), emotion.detach(), eps=indiff_eps)

        if steer_mode != "identity":
            z = _apply_sector_steering(z, steer_mode, steer_shift, steer_channels)

        z2 = z.square()
        re_z2_d16, im_z2_d16 = z2.to_xy()
        _, im_z_d16 = z.to_xy()

        re_z2_d16 = torch.nan_to_num(re_z2_d16, 0.0).to(logic.dtype)
        im_z2_d16 = torch.nan_to_num(im_z2_d16, 0.0).to(logic.dtype)
        im_z_d16 = torch.nan_to_num(im_z_d16, 0.0).to(logic.dtype)

    if (logic.requires_grad or emotion.requires_grad) and steer_mode == "identity":
        active_mask = (~z.is_zero()).to(logic.dtype)
        re_cont = (logic.square() - emotion.square()) * active_mask
        im_cont = (2.0 * logic * emotion) * active_mask
        e_cont = emotion * active_mask

        re_z2 = re_cont + (re_z2_d16 - re_cont).detach()
        im_z2 = im_cont + (im_z2_d16 - im_cont).detach()
        im_z = e_cont + (im_z_d16 - e_cont).detach()
    else:
        re_z2, im_z2, im_z = re_z2_d16, im_z2_d16, im_z_d16

    return z, z2, re_z2, im_z2, im_z


def _apply_sector_steering(
    z: D16ComplexTensor,
    mode: str,
    shift: int = 0,
    channels: Optional[torch.Tensor] = None,
) -> D16ComplexTensor:
    """Modify only the 16-bit sector mask while keeping (x, y) in [0°, 22.5°] untouched."""
    rot = z.rotation.clone()
    if channels is not None:
        target = rot[..., -1, channels]
    else:
        target = rot

    s = state_u16(target)
    special = (target == MASK_ZERO) | (target == MASK_NAN)
    is_active_upper = (s >= 0) & (s <= 7)

    rev = reverse_bits_u16(target)
    if mode == "rotate":
        steered = rol_u16(target, shift & 15)
        cond = ~special
    elif mode == "group_op":
        steered = apply_group_op_u16(target, shift & 15)
        cond = ~special
    elif mode == "active_not_logic":
        steered = rol_u16(rev, 8)
        cond = is_active_upper
    elif mode == "active_to_dislike":
        steered = rev
        cond = is_active_upper
    elif mode == "active_swap_le":
        steered = rol_u16(rev, 4)
        cond = is_active_upper
    elif mode == "active_to_indifferent":
        steered = torch.full_like(target, MASK_ZERO)
        cond = is_active_upper
    else:
        raise ValueError(f"Unknown steering mode: {mode}")

    updated = torch.where(cond, steered, target)
    if channels is not None:
        rot[..., -1, channels] = updated
    else:
        rot = updated

    return D16ComplexTensor(z.x, z.y, rot)


# =====================================================================
# 3. NORMALIZATION & D16 ROTARY POSITION EMBEDDINGS (RoPE)
# =====================================================================

class D16RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Qwen2.5 compatible)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class D16RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) executed via D16ComplexTensor multiplication.

    Each pair of coordinates (x_m, x_{m + D/2}) in a query or key head is encoded
    as a D16ComplexTensor and multiplied by the unit rotary phase D16ComplexTensor
    inside the [0°, 22.5°] fundamental wedge.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int = 32768, base: float = 1000000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for complex RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float64) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq_len]
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        inv_freq_expanded = self.inv_freq[None, :, None].to(calc_dtype).expand(position_ids.shape[0], -1, 1)
        pos_expanded = position_ids[:, None, :].to(calc_dtype)
        freqs = (inv_freq_expanded @ pos_expanded).transpose(1, 2)  # [batch, seq_len, head_dim // 2]
        cos = freqs.cos().to(dtype=x.dtype)
        sin = freqs.sin().to(dtype=x.dtype)
        return cos, sin

    @staticmethod
    def apply_d16_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate Q and K heads via D16ComplexTensor multiplication in [0°, 22.5°]."""
        # q, k shape: [batch, num_heads, seq_len, head_dim]
        # cos, sin shape: [batch, seq_len, head_dim // 2] -> broadcast over heads
        cos_b = cos.unsqueeze(1)
        sin_b = sin.unsqueeze(1)

        def _rotate_tensor_d16(t: torch.Tensor) -> torch.Tensor:
            half = t.shape[-1] // 2
            t_re = t[..., :half]
            t_im = t[..., half:]

            with torch.no_grad():
                z_t = D16ComplexTensor.from_xy(t_re.detach(), t_im.detach())
                z_rot = D16ComplexTensor.from_xy(cos_b.detach(), sin_b.detach())
                z_out = z_t * z_rot
                out_re_d16, out_im_d16 = z_out.to_xy()
                out_re_d16 = torch.nan_to_num(out_re_d16, 0.0).to(t.dtype)
                out_im_d16 = torch.nan_to_num(out_im_d16, 0.0).to(t.dtype)

            if t.requires_grad:
                # Differentiable STE path for training
                out_re_cont = t_re * cos_b - t_im * sin_b
                out_im_cont = t_re * sin_b + t_im * cos_b
                out_re = out_re_cont + (out_re_d16 - out_re_cont).detach()
                out_im = out_im_cont + (out_im_d16 - out_im_cont).detach()
            else:
                out_re, out_im = out_re_d16, out_im_d16

            return torch.cat([out_re, out_im], dim=-1)

        return _rotate_tensor_d16(q), _rotate_tensor_d16(k)


# =====================================================================
# 4. GROUPED-QUERY ATTENTION (D16QwenAttention)
# =====================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for Grouped-Query Attention (GQA)."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class D16QwenAttention(nn.Module):
    """Multi-headed Grouped-Query Attention with D16ComplexTensor RoPE."""

    def __init__(self, config: D16QwenConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = D16RotaryEmbedding.apply_d16_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_states_rep = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), present_key_value


# =====================================================================
# 5. D16Complex MLP BLOCK (D16QwenMLP)
# =====================================================================

class D16QwenMLP(nn.Module):
    """MLP layer built explicitly on D16ComplexTensor objects.

    Supports two operating modes configured via config.mlp_mode:
    - "d16_swiglu": 1-to-1 Qwen2.5 compatible mode where each neuron encodes
      (up_proj(x), gate_proj(x)) -> z in D16ComplexTensor, computes z2 = z.square()
      in the [0°, 22.5°] wedge, and outputs 0.5 * Im(z2) * sigmoid(Im(z)).
    - "d16_bipolar": Native compact complex mode where each D16Complex neuron
      outputs both Re(z^2) and Im(z^2) weighted by an 18-state sector profile.
    """

    def __init__(self, config: D16QwenConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.mlp_mode = config.mlp_mode
        self.indiff_eps = config.indiff_eps
        self.collect_stats = config.collect_d16_stats

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

        if self.mlp_mode == "d16_bipolar":
            self.down_im_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            self.sector_gain = nn.Parameter(torch.ones(self.intermediate_size, 18))
        else:
            self.register_parameter("down_im_proj", None)
            self.register_parameter("sector_gain", None)

        # Runtime steering and telemetry attributes
        self.steer_mode: str = "identity"
        self.steer_shift: int = 0
        self.steer_channels: Optional[torch.Tensor] = None
        self.register_buffer("state_counts", torch.zeros(18, dtype=torch.long), persistent=False)
        self.total_elements: int = 0
        self.indifferent_elements: int = 0

    def reset_stats(self) -> None:
        self.state_counts.zero_()
        self.total_elements = 0
        self.indifferent_elements = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logic = self.up_proj(x)
        emotion = self.gate_proj(x)

        z, _z2, re_z2, im_z2, im_z = d16_complex_square_ste(
            logic,
            emotion,
            indiff_eps=self.indiff_eps,
            steer_mode=self.steer_mode,
            steer_shift=self.steer_shift,
            steer_channels=self.steer_channels,
        )

        if self.collect_stats:
            with torch.no_grad():
                st = z.state().flatten()
                self.state_counts += torch.bincount(st.cpu(), minlength=18).to(self.state_counts.device)
                self.total_elements += int(z.rotation.numel())
                self.indifferent_elements += int((z.rotation == MASK_ZERO).sum())

        if self.mlp_mode == "d16_swiglu":
            # 0.5 * Im(z^2) * sigmoid(Im(z)) == L * SiLU(E) computed via [0°, 22.5°] wedge square
            act = 0.5 * im_z2 * torch.sigmoid(im_z)
            return self.down_proj(act)

        if self.mlp_mode == "d16_bipolar":
            s_idx = z.state()  # [..., K] in 0..17
            flat_s = s_idx.reshape(-1, self.intermediate_size)
            gain = torch.gather(
                self.sector_gain.unsqueeze(0).expand(flat_s.shape[0], -1, -1),
                dim=-1,
                index=flat_s.unsqueeze(-1),
            ).squeeze(-1).view_as(re_z2)
            return self.down_proj(re_z2 * gain) + self.down_im_proj(im_z2 * gain)

        raise ValueError(f"Unsupported mlp_mode: {self.mlp_mode}")


# =====================================================================
# 6. DECODER LAYER & FULL MODEL (D16QwenModel, D16QwenForCausalLM)
# =====================================================================

class D16QwenDecoderLayer(nn.Module):
    """Single Transformer Decoder layer of D16Qwen."""

    def __init__(self, config: D16QwenConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = D16QwenAttention(config, layer_idx=layer_idx)
        self.mlp = D16QwenMLP(config, layer_idx=layer_idx)
        self.input_layernorm = D16RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = D16RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_kv = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv


@dataclass
class D16CausalLMOutput:
    """Structured output for D16QwenForCausalLM."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[torch.Tensor] = None


class D16QwenModel(nn.Module):
    """Base D16Qwen Transformer backbone."""

    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [D16QwenDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = D16RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = D16RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    @staticmethod
    def _build_causal_mask(
        input_ids: torch.Tensor,
        dtype: torch.dtype,
        past_length: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        total_len = past_length + seq_len
        device = input_ids.device

        causal = torch.full((seq_len, total_len), float("-inf"), dtype=dtype, device=device)
        causal = torch.triu(causal, diagonal=past_length + 1)
        causal = causal.unsqueeze(0).unsqueeze(1).expand(bsz, 1, seq_len, total_len)

        if attention_mask is not None and attention_mask.shape[-1] == total_len:
            pad_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * float("-inf")
            causal = causal + pad_mask

        return causal

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        bsz, seq_len = input_ids.shape
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(bsz, -1)

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = self._build_causal_mask(
            input_ids, hidden_states.dtype, past_length=past_length, attention_mask=attention_mask
        )

        next_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present_kv = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache and next_cache is not None and present_kv is not None:
                next_cache.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache


class D16QwenForCausalLM(nn.Module):
    """Standalone D16Qwen Causal Language Model with training, generation, and D16 telemetry."""

    def __init__(self, config: D16QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.model = D16QwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> D16CausalLMOutput:
        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return D16CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=next_cache,
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive token generation using KV-cache and D16Complex arithmetic."""
        was_training = self.training
        self.eval()

        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            model_inputs = generated[:, -1:] if (use_cache and past_key_values is not None) else generated
            outputs = self.forward(
                input_ids=model_inputs,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            if do_sample:
                probs = F.softmax(next_token_logits / max(temperature, 1e-5), dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_tokens], dim=-1)
            if eos_token_id is not None and bool((next_tokens == eos_token_id).all()):
                break

        if was_training:
            self.train()
        return generated

    # --- D16 Telemetry, Indifferent Sparsity & Sector Steering Controls ---

    def set_indifferent_threshold(self, eps: float) -> None:
        """Set radial threshold eps for State 16 (0xFFFF) across all layers."""
        self.config.indiff_eps = eps
        for layer in self.model.layers:
            layer.mlp.indiff_eps = eps

    def set_collect_d16_stats(self, enabled: bool = True, reset: bool = True) -> None:
        """Enable or disable 18-state D16Complex histogram collection across all layers."""
        self.config.collect_d16_stats = enabled
        for layer in self.model.layers:
            layer.mlp.collect_stats = enabled
            if reset:
                layer.mlp.reset_stats()

    def get_d16_state_summary(self) -> Dict[str, Union[torch.Tensor, float, int]]:
        """Aggregate 18-state D16Complex counts and 0xFFFF sparsity across all layers."""
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

    def set_sector_steering(
        self,
        mode: str = "identity",
        shift: int = 0,
        channels_by_layer: Optional[Dict[int, torch.Tensor]] = None,
        target_layers: Optional[List[int]] = None,
    ) -> None:
        """Configure pure bitwise sector-mask steering on specified decoder layers."""
        layers_to_set = target_layers if target_layers is not None else list(range(len(self.model.layers)))
        for idx, layer in enumerate(self.model.layers):
            if idx in layers_to_set:
                layer.mlp.steer_mode = mode
                layer.mlp.steer_shift = shift
                layer.mlp.steer_channels = (
                    None if channels_by_layer is None else channels_by_layer.get(idx)
                )
            else:
                layer.mlp.steer_mode = "identity"
                layer.mlp.steer_shift = 0
                layer.mlp.steer_channels = None


__all__ = [
    "D16QwenConfig",
    "D16CausalLMOutput",
    "d16_complex_square_ste",
    "D16RMSNorm",
    "D16RotaryEmbedding",
    "D16QwenAttention",
    "D16QwenMLP",
    "D16QwenDecoderLayer",
    "D16QwenModel",
    "D16QwenForCausalLM",
]
