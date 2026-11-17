"""Standalone D16Qwen Transformer Architecture built purely on D16ComplexTensor.

This module implements the complete D16Qwen Causal Language Model
(D16QwenConfig, D16RMSNorm, D16RotaryEmbedding, D16QwenAttention, D16QwenMLP,
D16QwenDecoderLayer, D16QwenModel, D16QwenForCausalLM) from scratch in pure
PyTorch using D16ComplexTensor from d16_complex.py.

Key architectural principles:
1. Every neuron i in D16QwenMLP is an independent D16Complex object
   z_i = (x_i, y_i, rotation_i) with coordinates in the closed [0°, 22.5°]
   fundamental wedge and a 16-bit dihedral mask rotation_i.
2. Inside D16QwenMLP._eval_d16(z), the 16-bit sector state s = z.state()
   directly switches the calculation branch over wedge coordinates (x, y):
   - Axial sectors ((s & 3) in (0, 3))    -> x * y          (Im(z^2)/2 branch)
   - Diagonal sectors ((s & 3) in (1, 2)) -> 0.5*(x^2 - y^2) (Re(z^2)/2 branch)
3. All dihedral group operations of Z_8 |>< Z_2 (including orbit partner links
   and runtime sector steering) act purely via bitwise shifts/reversals on the
   16-bit rotation mask without modifying (x, y).
4. RoPE in D16QwenAttention rotates Q and K heads via D16ComplexTensor
   multiplication inside the [0°, 22.5°] fundamental wedge.
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
    INV_SQRT2,
    MASK_NAN,
    MASK_ZERO,
    apply_all_16_d16_ops_u16,
    apply_group_op_u16,
    reverse_bits_u16,
    rol_u16,
    rotated_sum_d16,
    state_u16,
)


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

@dataclass
class D16QwenConfig:
    """Configuration class for D16QwenModel and D16QwenForCausalLM."""

    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864  # Each of the 4864 neurons is an independent D16Complex object
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = True
    mlp_bias: bool = False

    # D16Complex runtime parameters
    indiff_eps: Union[float, str] = 0.0
    indiff_rel: float = 0.25
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
        indiff_eps: Union[float, str] = 0.0,
        indiff_rel: float = 0.25,
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
            indiff_eps=indiff_eps,
            indiff_rel=indiff_rel,
        )


# =====================================================================
# 2. PURE BITWISE SECTOR STEERING ON D16ComplexTensor
# =====================================================================

def _apply_sector_steering(
    z: D16ComplexTensor,
    mode: str,
    shift: int = 0,
    channels: Optional[torch.Tensor] = None,
) -> D16ComplexTensor:
    """Apply mask-based steering; Indifferent also resets the wedge coordinates."""
    rot = z.rotation.clone()
    target = rot[..., -1, channels] if channels is not None else rot

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

    result = D16ComplexTensor(z.x, z.y, rot)
    if mode == "active_to_indifferent":
        return result.to_indifferent_where(rot == MASK_ZERO)
    return result


# =====================================================================
# 3. NORMALIZATION & D16 ROTARY POSITION EMBEDDINGS (RoPE)
# =====================================================================

class D16RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

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
    """Rotary Position Embedding (RoPE) executed via D16ComplexTensor multiplication."""

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
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        inv_freq_expanded = self.inv_freq[None, :, None].to(calc_dtype).expand(position_ids.shape[0], -1, 1)
        pos_expanded = position_ids[:, None, :].to(calc_dtype)
        freqs = (inv_freq_expanded @ pos_expanded).transpose(1, 2)
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
        cos_b = cos.unsqueeze(1)
        sin_b = sin.unsqueeze(1)

        def _rotate_tensor_d16(t: torch.Tensor) -> torch.Tensor:
            half = t.shape[-1] // 2
            t_re = t[..., :half]
            t_im = t[..., half:]

            z_t = D16ComplexTensor.from_xy(t_re, t_im)
            z_rot = D16ComplexTensor.from_xy(cos_b, sin_b)
            z_out = z_t * z_rot
            out_re, out_im = z_out.to_xy()
            out_re = torch.nan_to_num(out_re, 0.0).to(t.dtype)
            out_im = torch.nan_to_num(out_im, 0.0).to(t.dtype)
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
# 5. NATIVE D16Complex MLP BLOCK (D16QwenMLP)
# =====================================================================

class D16QwenMLP(nn.Module):
    """Native MLP layer where every neuron is a D16Complex(x, y, rotation) object.

    Forward execution:
    1. Computes (logic, emotion) = (up_proj(x), gate_proj(x)) and packs all N neurons
       into z = D16ComplexTensor.from_xy(logic, emotion, eps=self.indiff_eps) with
       coordinates (z.x, z.y) in the [0°, 22.5°] fundamental wedge and 16-bit mask z.rotation.
    2. Optional orbit links (if configured for orbit-sharing experiments) instantiate
       partner D16Complex objects at their respective indices by taking the anchor's
       (x, y) and applying a pure bitwise Z_8 |>< Z_2 shift/reversal to rotation.
    3. Evaluates each D16Complex object via _eval_d16(z), where the sector s = z.state()
       switches between the two halves of Hamilton's complex square formula:
       - Axial sectors (0, 3, 4, 7, 8, 11, 12, 15)    -> x * y
       - Diagonal sectors (1, 2, 5, 6, 9, 10, 13, 14) -> 0.5 * (x^2 - y^2)
    4. Projects the resulting N activations via down_proj.
    """

    def __init__(self, config: D16QwenConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.indiff_eps = config.indiff_eps
        self.indiff_rel = config.indiff_rel
        self.collect_stats = config.collect_d16_stats

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

        # Sector branch lookup tables (states 0..15, plus 16=Indifferent and 17=Undefined)
        is_diag = [1.0 if (s & 3) in (1, 2) else 0.0 for s in range(16)] + [0.0, 0.0]
        p_sign  = [1.0 if ((s >> 1) in (0, 1, 4, 5)) else -1.0 for s in range(16)] + [0.0, 0.0]
        e_is_wx = [1.0 if ((s >> 1) in (1, 2, 5, 6)) else 0.0 for s in range(16)] + [0.0, 0.0]
        e_pos   = [1.0 if s < 8 else 0.0 for s in range(16)] + [0.0, 0.0]

        self.register_buffer("IS_DIAG_SEC", torch.tensor(is_diag), persistent=False)
        self.register_buffer("PROD_SIGN",   torch.tensor(p_sign),  persistent=False)
        self.register_buffer("EMO_IS_WX",   torch.tensor(e_is_wx), persistent=False)
        self.register_buffer("EMO_POS",     torch.tensor(e_pos),   persistent=False)

        # Optional bitwise orbit-linking buffers (keep all N D16Complex objects!)
        self.register_buffer("orbit_partner_idx", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("orbit_anchor_idx",  torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("orbit_op_ids",      torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("orbit_scales",      torch.empty(0, dtype=torch.float32), persistent=False)

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

    def set_orbit_links(
        self,
        partner_idx: torch.Tensor,
        anchor_idx: torch.Tensor,
        op_ids: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
    ) -> None:
        """Configure partner D16Complex objects to derive their state by rotating anchor's mask."""
        dev = self.up_proj.weight.device
        self.orbit_partner_idx = partner_idx.to(device=dev, dtype=torch.long)
        self.orbit_anchor_idx = anchor_idx.to(device=dev, dtype=torch.long)
        self.orbit_op_ids = op_ids.to(device=dev, dtype=torch.long)
        if scales is None:
            self.orbit_scales = torch.ones(partner_idx.numel(), device=dev, dtype=self.up_proj.weight.dtype)
        else:
            self.orbit_scales = scales.to(device=dev, dtype=self.up_proj.weight.dtype)

    def clear_orbit_links(self) -> None:
        dev = self.up_proj.weight.device
        self.orbit_partner_idx = torch.empty(0, dtype=torch.long, device=dev)
        self.orbit_anchor_idx = torch.empty(0, dtype=torch.long, device=dev)
        self.orbit_op_ids = torch.empty(0, dtype=torch.long, device=dev)
        self.orbit_scales = torch.empty(0, dtype=self.up_proj.weight.dtype, device=dev)

    def _eval_d16(self, z: D16ComplexTensor) -> torch.Tensor:
        """Evaluate each D16Complex object directly from wedge (x, y) and sector state s16."""
        x = z.x
        y = z.y
        s16 = z.state()

        is_diag   = self.IS_DIAG_SEC.to(dtype=x.dtype)[s16]
        prod_sign = self.PROD_SIGN.to(dtype=x.dtype)[s16]
        emo_is_wx = self.EMO_IS_WX.to(dtype=x.dtype)[s16]
        emo_pos   = self.EMO_POS.to(dtype=x.dtype)[s16]

        # Sector s16 switches the calculation branch over wedge coordinates (x, y):
        #   Axial sector    -> x * y           (Im(z^2)/2 branch)
        #   Diagonal sector -> 0.5*(x^2 - y^2) (Re(z^2)/2 branch)
        oct_prod = torch.where(is_diag > 0.5, 0.5 * (x * x - y * y), x * y)
        wx = torch.where(is_diag > 0.5, rotated_sum_d16(x, y), x)
        wy = torch.where(is_diag > 0.5, (x - y) * INV_SQRT2, y)

        b = emo_is_wx * wx + (1.0 - emo_is_wx) * wy
        sign_e = 2.0 * emo_pos - 1.0
        act = prod_sign * oct_prod * torch.sigmoid(sign_e * b)
        return torch.where(s16 >= 16, torch.zeros_like(act), act)

    def forward_d16_objects(self, x: torch.Tensor) -> D16ComplexTensor:
        """Encode input x into the N D16Complex objects of this layer."""
        logic = self.up_proj(x)
        emotion = self.gate_proj(x)
        z = D16ComplexTensor.from_xy(logic, emotion, eps=self.indiff_eps, rel=self.indiff_rel)

        # If bitwise orbit partner links are active, instantiate partner D16Complex objects
        # by taking anchor (x, y) and rotating only the 16-bit mask via Z_8 |>< Z_2!
        if self.orbit_partner_idx.numel() > 0:
            zx = z.x.clone()
            zy = z.y.clone()
            zm = z.rotation.clone()

            anchor_m = zm[..., self.orbit_anchor_idx]
            all_16 = apply_all_16_d16_ops_u16(anchor_m)  # [..., M, 16]
            idx_shape = [1] * (anchor_m.ndim - 1) + [self.orbit_op_ids.numel(), 1]
            idx_gather = self.orbit_op_ids.view(*idx_shape).expand(*anchor_m.shape, 1)
            partner_m = torch.gather(all_16, dim=-1, index=idx_gather).squeeze(-1)

            sc = self.orbit_scales.to(zx.dtype)
            zx[..., self.orbit_partner_idx] = zx[..., self.orbit_anchor_idx] * sc
            zy[..., self.orbit_partner_idx] = zy[..., self.orbit_anchor_idx] * sc
            zm[..., self.orbit_partner_idx] = partner_m
            z = D16ComplexTensor(zx, zy, zm)

        if self.steer_mode != "identity":
            z = _apply_sector_steering(z, self.steer_mode, self.steer_shift, self.steer_channels)

        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_d16_objects(x)

        if self.collect_stats:
            with torch.no_grad():
                st = z.state().flatten()
                self.state_counts += torch.bincount(st.cpu(), minlength=18).to(self.state_counts.device)
                self.total_elements += int(z.rotation.numel())
                self.indifferent_elements += int((z.rotation == MASK_ZERO).sum())

        act = self._eval_d16(z)
        return self.down_proj(act)


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
        min_val = torch.finfo(dtype).min

        causal = torch.full((seq_len, total_len), min_val, dtype=dtype, device=device)
        causal = torch.triu(causal, diagonal=past_length + 1)
        causal = causal.unsqueeze(0).unsqueeze(1).expand(bsz, 1, seq_len, total_len).clone()

        if attention_mask is not None and attention_mask.shape[-1] == total_len:
            key_is_pad = (attention_mask[:, None, None, :] == 0)
            causal = causal.masked_fill(key_is_pad, min_val)

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
    """Standalone D16Qwen Causal Language Model."""

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

    def set_indifferent_threshold(self, eps: Union[float, str], rel: Optional[float] = None) -> None:
        if eps != "auto" and (not isinstance(eps, (int, float)) or float(eps) < 0.0):
            raise ValueError("eps must be a nonnegative float or 'auto'")
        self.config.indiff_eps = eps
        if rel is not None:
            self.config.indiff_rel = float(rel)
        for layer in self.model.layers:
            layer.mlp.indiff_eps = eps
            if rel is not None:
                layer.mlp.indiff_rel = float(rel)

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

    def set_sector_steering(
        self,
        mode: str = "identity",
        shift: int = 0,
        channels_by_layer: Optional[Dict[int, torch.Tensor]] = None,
        target_layers: Optional[List[int]] = None,
    ) -> None:
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
    "D16RMSNorm",
    "D16RotaryEmbedding",
    "D16QwenAttention",
    "D16QwenMLP",
    "D16QwenDecoderLayer",
    "D16QwenModel",
    "D16QwenForCausalLM",
]
