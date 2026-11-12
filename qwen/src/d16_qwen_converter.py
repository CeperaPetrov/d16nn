"""Converter and Orbit Analyzer from real Qwen2.5 weights to native D16QwenForCausalLM.

This module converts standard Qwen2/Qwen2.5 state_dicts or checkpoint directories
into the standalone D16QwenForCausalLM architecture (d16_qwen_model.py) without
requiring HuggingFace transformers at runtime.

It also produces detailed LayerOrbitReport diagnostics for visualizing how groups
of real neurons merge into single D16Complex units under the dihedral group Z_8 |>< Z_2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from d16_complex import INV_SQRT2
from d16_qwen_model import D16QwenConfig, D16QwenForCausalLM, D16QwenMLP


# =====================================================================
# 1. DIHEDRAL GROUP Z_8 |>< Z_2 MATRICES & METADATA (16 OPERATIONS)
# =====================================================================

OP_NAMES: Tuple[str, ...] = (
    "0: (+L, +E) [Im, 0°]",
    "1: (-L, -E) [Im, 180°]",
    "2: (+L, -E) [Im, conj]",
    "3: (-L, +E) [Im, -conj]",
    "4: (-E, +L) [Im, +90°]",
    "5: (+E, -L) [Im, -90°]",
    "6: (+E, +L) [Im, swap]",
    "7: (-E, -L) [Im, -swap]",
    "8: (L-E, L+E)/√2 [Re, +45°]",
    "9: (-L-E, L-E)/√2 [Re, +135°]",
    "10: (-L+E, -L-E)/√2 [Re, +225°]",
    "11: (L+E, -L+E)/√2 [Re, +315°]",
    "12: (L+E, L-E)/√2 [Re, refl 22.5°]",
    "13: (-L+E, L+E)/√2 [Re, refl 67.5°]",
    "14: (-L-E, -L+E)/√2 [Re, refl 112.5°]",
    "15: (L-E, -L-E)/√2 [Re, refl 157.5°]",
)

# Знак квадратичного произведения L' * E' относительно базового базиса (L, E):
# Для op in 0..7 : L'*E' = OP_QUAD_SIGN[op] * (L * E)         = OP_QUAD_SIGN[op] * 0.5 * Im(z^2)
# Для op in 8..15: L'*E' = OP_QUAD_SIGN[op] * 0.5*(L^2 - E^2) = OP_QUAD_SIGN[op] * 0.5 * Re(z^2)
OP_QUAD_SIGN: Tuple[float, ...] = (
    1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0,   # 0..7  (Axial Im branch)
    1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,   # 8..15 (Diagonal 45° Re branch)
)

RAYLEIGH_SIGMOID_KAPPA: float = 15.0 * math.sqrt(math.pi / 2.0) / 8.0  # ~2.349964


def d16_group_matrices(
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Return all 16 orthogonal 2x2 matrices of Z_8 |>< Z_2 matching d16_complex.py."""
    s = INV_SQRT2
    mats = [
        # 0..7: Axial D8 subgroup (Im(z^2) branch)
        [[1.0, 0.0], [0.0, 1.0]],    # 0:  (+L, +E)
        [[-1.0, 0.0], [0.0, -1.0]],  # 1:  (-L, -E)
        [[1.0, 0.0], [0.0, -1.0]],   # 2:  (+L, -E)
        [[-1.0, 0.0], [0.0, 1.0]],   # 3:  (-L, +E)
        [[0.0, -1.0], [1.0, 0.0]],   # 4:  (-E, +L)
        [[0.0, 1.0], [-1.0, 0.0]],   # 5:  (+E, -L)
        [[0.0, 1.0], [1.0, 0.0]],    # 6:  (+E, +L)
        [[0.0, -1.0], [-1.0, 0.0]],  # 7:  (-E, -L)
        # 8..15: 45-degree Hadamard coset (Re(z^2) branch)
        [[s, -s], [s, s]],           # 8:  ( L-E,  L+E)/sqrt(2)
        [[-s, -s], [s, -s]],         # 9:  (-L-E,  L-E)/sqrt(2)
        [[-s, s], [-s, -s]],         # 10: (-L+E, -L-E)/sqrt(2)
        [[s, s], [-s, s]],           # 11: ( L+E, -L+E)/sqrt(2)
        [[s, s], [s, -s]],           # 12: ( L+E,  L-E)/sqrt(2)
        [[-s, s], [s, s]],           # 13: (-L+E,  L+E)/sqrt(2)
        [[-s, -s], [-s, s]],         # 14: (-L-E, -L+E)/sqrt(2)
        [[s, -s], [-s, -s]],         # 15: ( L-E, -L-E)/sqrt(2)
    ]
    return torch.tensor(mats, dtype=dtype, device=device)


# =====================================================================
# 2. DIAGNOSTIC DATA STRUCTURES FOR VISUALIZATION
# =====================================================================

@dataclass
class LayerOrbitReport:
    """Detailed diagnostic report of Z_8 |>< Z_2 orbit extraction for one MLP layer."""

    layer_idx: int
    mode: str
    orig_intermediate_size: int
    d16_intermediate_size: int
    quadruplets: torch.Tensor        # [K, 4] original neuron indices in each D16 unit
    group_ops: torch.Tensor          # [K, 4] operation IDs (0..15) for each member
    member_scales: torch.Tensor      # [K, 4] projection scales onto canonical basis
    orbit_scores: torch.Tensor       # [K] coherence score in [0, 1] for each unit
    local_2x2: torch.Tensor          # [K, 4, 2, 2] local 2D coordinates of each member in (U_q, G_q)
    op_histogram: torch.Tensor       # [16] counts of operations 0..15 across the layer
    hadamard_45_pct: float           # Percentage of non-anchor members using 45° Re(z^2) ops (8..15)
    mean_orbit_score: float

    def top_orbits(self, k: int = 5) -> List[Dict[str, Union[int, float, List[int], List[str]]]]:
        """Return human-readable summary of the top-k highest-scoring D16Complex orbits."""
        k = min(k, int(self.orbit_scores.numel()))
        vals, idxs = torch.topk(self.orbit_scores, k=k, largest=True)
        result = []
        for rank in range(k):
            q = int(idxs[rank].item())
            neurons = [int(x) for x in self.quadruplets[q].tolist()]
            ops = [int(x) for x in self.group_ops[q].tolist()]
            op_labels = [OP_NAMES[o] for o in ops]
            result.append({
                "unit_idx": q,
                "score": float(vals[rank].item()),
                "neurons": neurons,
                "ops": ops,
                "op_labels": op_labels,
            })
        return result


@dataclass
class ConversionReport:
    """Global report across all converted layers of a D16QwenForCausalLM model."""

    mode: str
    num_layers: int
    orig_intermediate_size: int
    d16_intermediate_size: int
    layer_reports: Dict[int, LayerOrbitReport] = field(default_factory=dict)

    @property
    def global_op_histogram(self) -> torch.Tensor:
        hist = torch.zeros(16, dtype=torch.long)
        for rep in self.layer_reports.values():
            hist += rep.op_histogram.cpu()
        return hist

    @property
    def mean_orbit_score(self) -> float:
        if not self.layer_reports:
            return 0.0
        return sum(r.mean_orbit_score for r in self.layer_reports.values()) / len(self.layer_reports)

    @property
    def mean_hadamard_45_pct(self) -> float:
        if not self.layer_reports:
            return 0.0
        return sum(r.hadamard_45_pct for r in self.layer_reports.values()) / len(self.layer_reports)


# =====================================================================
# 3. ORBIT DISCOVERY & ANALYTICAL SECTOR GAIN COMPILATION
# =====================================================================

@torch.no_grad()
def compute_d16_pairwise_alignment(
    up_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    rms_weight: Optional[torch.Tensor] = None,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute best Z_8 |>< Z_2 alignment score and op_id (0..15) for all neuron pairs."""
    device = up_weight.device
    calc_dtype = torch.float32
    u = up_weight.detach().to(calc_dtype)
    g = gate_weight.detach().to(calc_dtype)

    if rms_weight is not None:
        w = rms_weight.detach().to(device=device, dtype=calc_dtype).unsqueeze(0)
        u = u * w
        g = g * w

    A = torch.stack([u, g], dim=1)
    norms = A.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    A_norm = A / norms
    N, _, D = A_norm.shape

    mats = d16_group_matrices(dtype=calc_dtype, device=device)
    A_flat = A_norm.reshape(N, 2 * D)

    best_scores = torch.zeros((N, N), dtype=calc_dtype, device=device)
    best_ops = torch.zeros((N, N), dtype=torch.long, device=device)

    MA = torch.einsum("oab,nbd->onad", mats, A_norm).reshape(16, N, 2 * D)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dots = torch.einsum("id,ond->oin", A_flat[start:end], MA)
        sc, op = dots.max(dim=0)
        best_scores[start:end] = sc.clamp(-1.0, 1.0)
        best_ops[start:end] = op

    best_scores.fill_diagonal_(-1.0)
    return best_scores, best_ops


@torch.no_grad()
def select_d16_quadruplets(
    best_scores: torch.Tensor,
    best_ops: torch.Tensor,
    num_units: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily partition N neurons into K quadruplets linked by Z_8 |>< Z_2 operations."""
    N = best_scores.shape[0]
    K = (N // 4) if num_units is None else min(num_units, N // 4)
    device = best_scores.device

    top3_vals, _ = torch.topk(best_scores, k=min(3, N - 1), dim=1)
    anchor_priority = top3_vals.mean(dim=1)
    order = torch.argsort(anchor_priority, descending=True)

    used = torch.zeros(N, dtype=torch.bool, device=device)
    quads = torch.zeros((K, 4), dtype=torch.long, device=device)
    q_ops = torch.zeros((K, 4), dtype=torch.long, device=device)
    q_scores = torch.zeros(K, dtype=best_scores.dtype, device=device)

    is_diag_op = best_ops >= 8
    q_idx = 0

    for anchor_t in order:
        if q_idx >= K:
            break
        i0 = int(anchor_t.item())
        if bool(used[i0]):
            continue
        used[i0] = True

        row_scores = best_scores[i0].clone()
        row_scores[used] = -2.0

        diag_mask = (~used) & is_diag_op[i0]
        axial_mask = (~used) & (~is_diag_op[i0])

        chosen: List[int] = [i0]
        chosen_ops: List[int] = [0]
        score_sum = 0.0

        if bool(diag_mask.any()):
            diag_scores = torch.where(diag_mask, row_scores, torch.full_like(row_scores, -2.0))
            j_diag = int(torch.argmax(diag_scores).item())
            if float(diag_scores[j_diag]) > -1.5:
                used[j_diag] = True
                chosen.append(j_diag)
                chosen_ops.append(int(best_ops[i0, j_diag].item()))
                score_sum += float(best_scores[i0, j_diag].item())

        if bool(axial_mask.any()) and len(chosen) < 4:
            ax_scores = torch.where(axial_mask & (~used), row_scores, torch.full_like(row_scores, -2.0))
            j_ax = int(torch.argmax(ax_scores).item())
            if float(ax_scores[j_ax]) > -1.5:
                used[j_ax] = True
                chosen.append(j_ax)
                chosen_ops.append(int(best_ops[i0, j_ax].item()))
                score_sum += float(best_scores[i0, j_ax].item())

        while len(chosen) < 4:
            rem_scores = torch.where(~used, best_scores[i0], torch.full_like(row_scores, -2.0))
            j_next = int(torch.argmax(rem_scores).item())
            used[j_next] = True
            chosen.append(j_next)
            chosen_ops.append(int(best_ops[i0, j_next].item()))
            score_sum += float(best_scores[i0, j_next].item())

        quads[q_idx] = torch.tensor(chosen, dtype=torch.long, device=device)
        q_ops[q_idx] = torch.tensor(chosen_ops, dtype=torch.long, device=device)
        q_scores[q_idx] = score_sum / 3.0
        q_idx += 1

    sort_order = torch.argsort(q_scores, descending=True)
    return quads[sort_order], q_ops[sort_order], q_scores[sort_order]


@torch.no_grad()
def compute_analytic_sector_gains(
    gate_sigma: torch.Tensor,
    op_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute analytical 18-state sector gain profile [K, 18] for D16QwenMLP."""
    device = gate_sigma.device
    dtype = gate_sigma.dtype
    K = gate_sigma.shape[0]

    angles = (torch.arange(16, dtype=torch.float64, device=device) + 0.5) * (math.pi / 8.0)
    cos_a = angles.cos()
    sin_a = angles.sin()

    if op_ids is None:
        dir_e = sin_a.unsqueeze(0).expand(K, 16)
    else:
        mats = d16_group_matrices(dtype=torch.float64, device=device)[op_ids]
        dir_e = mats[:, 1, 0:1] * cos_a.unsqueeze(0) + mats[:, 1, 1:2] * sin_a.unsqueeze(0)

    eff_arg = RAYLEIGH_SIGMOID_KAPPA * gate_sigma.to(torch.float64).unsqueeze(1) * dir_e
    gains_16 = torch.sigmoid(eff_arg).to(dtype)

    gains_18 = torch.zeros((K, 18), dtype=dtype, device=device)
    gains_18[:, :16] = gains_16
    return gains_18


# =====================================================================
# 4. LAYER CONVERSION (ORBIT 4->1 AND FULL 1->1)
# =====================================================================

@torch.no_grad()
def convert_mlp_layer_to_d16(
    up_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    rms_weight: Optional[torch.Tensor] = None,
    layer_idx: int = 0,
    mode: str = "orbit_4to1",
) -> Tuple[Dict[str, torch.Tensor], LayerOrbitReport]:
    """Convert a real Qwen MLP layer (up, gate, down) into D16QwenMLP parameters + report."""
    device = up_weight.device
    orig_dtype = up_weight.dtype
    calc_dtype = torch.float32

    U = up_weight.detach().to(calc_dtype)
    G = gate_weight.detach().to(calc_dtype)
    D_mat = down_weight.detach().to(calc_dtype)
    N, hidden_size = U.shape

    if rms_weight is not None:
        rms_w = rms_weight.detach().to(device=device, dtype=calc_dtype)
    else:
        rms_w = torch.ones(hidden_size, dtype=calc_dtype, device=device)

    best_scores, best_ops = compute_d16_pairwise_alignment(U, G, rms_w)
    mats = d16_group_matrices(dtype=calc_dtype, device=device)
    quad_signs = torch.tensor(OP_QUAD_SIGN, dtype=calc_dtype, device=device)

    if mode == "full_1to1":
        K = N
        top3_sc, top3_idx = torch.topk(best_scores, k=min(3, N - 1), dim=1)
        anchors = torch.arange(N, dtype=torch.long, device=device).unsqueeze(1)
        quads = torch.cat([anchors, top3_idx], dim=1)
        anchor_ops = torch.zeros((N, 1), dtype=torch.long, device=device)
        partner_ops = torch.gather(best_ops, dim=1, index=top3_idx)
        q_ops = torch.cat([anchor_ops, partner_ops], dim=1)
        q_scores = top3_sc.mean(dim=1)

        U_d16 = U.clone()
        G_d16 = G.clone()

        sigma_e = (G * rms_w.unsqueeze(0)).norm(dim=1).clamp_min(1e-6)
        sector_im_gain = compute_analytic_sector_gains(sigma_e)
        sector_re_gain = torch.zeros_like(sector_im_gain)

        down_im_proj = D_mat.clone()
        down_re_proj = torch.zeros_like(D_mat)

        A_all = torch.stack([U, G], dim=1)
        basis_norm = A_all / A_all.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        member_A = A_all[quads]
        local_2x2 = torch.einsum("qad,qmbd->qmba", basis_norm, member_A)
        member_scales = torch.ones((N, 4), dtype=calc_dtype, device=device)
        member_scales[:, 1:] = top3_sc

        op_hist = torch.bincount(partner_ops.flatten().cpu(), minlength=16)
        hadamard_pct = float((partner_ops >= 8).float().mean().item() * 100.0)

        params = {
            "up_proj.weight": U_d16.to(orig_dtype),
            "gate_proj.weight": G_d16.to(orig_dtype),
            "sector_re_gain": sector_re_gain.to(orig_dtype),
            "sector_im_gain": sector_im_gain.to(orig_dtype),
            "down_re_proj.weight": down_re_proj.to(orig_dtype),
            "down_im_proj.weight": down_im_proj.to(orig_dtype),
        }
        report = LayerOrbitReport(
            layer_idx=layer_idx,
            mode=mode,
            orig_intermediate_size=N,
            d16_intermediate_size=K,
            quadruplets=quads.cpu(),
            group_ops=q_ops.cpu(),
            member_scales=member_scales.cpu(),
            orbit_scores=q_scores.cpu(),
            local_2x2=local_2x2.cpu(),
            op_histogram=op_hist,
            hadamard_45_pct=hadamard_pct,
            mean_orbit_score=float(q_scores.mean().item()),
        )
        return params, report

    if mode != "orbit_4to1":
        raise ValueError(f"Unsupported conversion mode: {mode}. Use 'orbit_4to1' or 'full_1to1'.")

    quads, q_ops, q_scores = select_d16_quadruplets(best_scores, best_ops)
    K = quads.shape[0]

    A_all = torch.stack([U, G], dim=1)
    norms_all = A_all.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    A_unit = A_all / norms_all

    member_unit = A_unit[quads]
    member_raw = A_all[quads]
    M_q = mats[q_ops]

    aligned_dirs = torch.einsum("qmba,qmbd->qmad", M_q, member_unit)
    canon_dir = aligned_dirs.mean(dim=1)
    canon_dir = canon_dir / canon_dir.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)

    anchor_norms = norms_all[quads[:, 0], 0, 0]
    B_canon = canon_dir * anchor_norms[:, None, None]
    U_d16 = B_canon[:, 0, :]
    G_d16 = B_canon[:, 1, :]

    mapped_B = torch.einsum("qmab,qbd->qmad", M_q, B_canon)
    denom = mapped_B.square().sum(dim=(2, 3)).clamp_min(1e-12)
    member_scales = (member_raw * mapped_B).sum(dim=(2, 3)) / denom

    local_2x2 = torch.einsum("qad,qmbd->qmba", canon_dir, member_raw)

    sigma_base = (G_d16 * rms_w.unsqueeze(0)).norm(dim=1).clamp_min(1e-6)

    down_im_proj = torch.zeros((hidden_size, K), dtype=calc_dtype, device=device)
    down_re_proj = torch.zeros((hidden_size, K), dtype=calc_dtype, device=device)
    sector_im_accum = torch.zeros((K, 18), dtype=calc_dtype, device=device)
    sector_re_accum = torch.zeros((K, 18), dtype=calc_dtype, device=device)
    im_weight_sum = torch.zeros(K, dtype=calc_dtype, device=device)
    re_weight_sum = torch.zeros(K, dtype=calc_dtype, device=device)

    for m in range(4):
        idx_m = quads[:, m]
        op_m = q_ops[:, m]
        scale_m = member_scales[:, m]
        quad_scale = scale_m.square() * quad_signs[op_m]
        d_m = D_mat[:, idx_m]
        d_norm_m = d_m.norm(dim=0).clamp_min(1e-8)

        gains_m = compute_analytic_sector_gains(sigma_base * scale_m.abs(), op_m)
        is_re_branch = op_m >= 8

        im_mask = (~is_re_branch).to(calc_dtype)
        re_mask = is_re_branch.to(calc_dtype)

        down_im_proj += d_m * (quad_scale * im_mask).unsqueeze(0)
        down_re_proj += d_m * (0.5 * quad_scale * re_mask).unsqueeze(0)

        w_im = d_norm_m * scale_m.square() * im_mask
        w_re = d_norm_m * scale_m.square() * re_mask

        sector_im_accum += gains_m * w_im.unsqueeze(1)
        sector_re_accum += gains_m * w_re.unsqueeze(1)
        im_weight_sum += w_im
        re_weight_sum += w_re

    default_gains = compute_analytic_sector_gains(sigma_base)
    sector_im_gain = torch.where(
        (im_weight_sum > 1e-7).unsqueeze(1),
        sector_im_accum / im_weight_sum.clamp_min(1e-7).unsqueeze(1),
        default_gains,
    )
    sector_re_gain = torch.where(
        (re_weight_sum > 1e-7).unsqueeze(1),
        sector_re_accum / re_weight_sum.clamp_min(1e-7).unsqueeze(1),
        default_gains,
    )

    non_anchor_ops = q_ops[:, 1:].flatten()
    op_hist = torch.bincount(non_anchor_ops.cpu(), minlength=16)
    hadamard_pct = float((non_anchor_ops >= 8).float().mean().item() * 100.0)

    params = {
        "up_proj.weight": U_d16.to(orig_dtype),
        "gate_proj.weight": G_d16.to(orig_dtype),
        "sector_re_gain": sector_re_gain.to(orig_dtype),
        "sector_im_gain": sector_im_gain.to(orig_dtype),
        "down_re_proj.weight": down_re_proj.to(orig_dtype),
        "down_im_proj.weight": down_im_proj.to(orig_dtype),
    }
    report = LayerOrbitReport(
        layer_idx=layer_idx,
        mode=mode,
        orig_intermediate_size=N,
        d16_intermediate_size=K,
        quadruplets=quads.cpu(),
        group_ops=q_ops.cpu(),
        member_scales=member_scales.cpu(),
        orbit_scores=q_scores.cpu(),
        local_2x2=local_2x2.cpu(),
        op_histogram=op_hist,
        hadamard_45_pct=hadamard_pct,
        mean_orbit_score=float(q_scores.mean().item()),
    )
    return params, report


# =====================================================================
# 5. FULL MODEL CONVERTER CLASS (QwenToD16Converter)
# =====================================================================

class QwenToD16Converter:
    """Offline & in-memory converter from standard Qwen2/Qwen2.5 to D16QwenForCausalLM."""

    @staticmethod
    def config_from_qwen_dict(
        qwen_cfg: Dict[str, Union[int, float, bool, str]],
        mode: str = "orbit_4to1",
        indiff_eps: float = 0.0,
    ) -> D16QwenConfig:
        """Build D16QwenConfig from a standard Qwen config.json dictionary."""
        orig_inter = int(qwen_cfg.get("intermediate_size", 4864))
        if mode == "orbit_4to1":
            d16_inter = orig_inter // 4
        elif mode == "full_1to1":
            d16_inter = orig_inter
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return D16QwenConfig(
            vocab_size=int(qwen_cfg.get("vocab_size", 151936)),
            hidden_size=int(qwen_cfg.get("hidden_size", 896)),
            intermediate_size=d16_inter,
            num_hidden_layers=int(qwen_cfg.get("num_hidden_layers", 24)),
            num_attention_heads=int(qwen_cfg.get("num_attention_heads", 14)),
            num_key_value_heads=int(qwen_cfg.get("num_key_value_heads", 2)),
            max_position_embeddings=int(qwen_cfg.get("max_position_embeddings", 32768)),
            rms_norm_eps=float(qwen_cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(qwen_cfg.get("rope_theta", 1000000.0)),
            tie_word_embeddings=bool(qwen_cfg.get("tie_word_embeddings", True)),
            attention_bias=True,
            mlp_bias=False,
            indiff_eps=indiff_eps,
        )

    @classmethod
    @torch.no_grad()
    def convert_state_dict(
        cls,
        qwen_state_dict: Dict[str, torch.Tensor],
        qwen_config_dict: Dict[str, Union[int, float, bool, str]],
        mode: str = "orbit_4to1",
        indiff_eps: float = 0.0,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[D16QwenForCausalLM, ConversionReport]:
        """Convert an in-memory Qwen state_dict into a ready D16QwenForCausalLM + ConversionReport."""
        d16_cfg = cls.config_from_qwen_dict(qwen_config_dict, mode=mode, indiff_eps=indiff_eps)
        d16_model = D16QwenForCausalLM(d16_cfg).to(device)
        new_sd = d16_model.state_dict()

        new_sd["model.embed_tokens.weight"].copy_(qwen_state_dict["model.embed_tokens.weight"])
        new_sd["model.norm.weight"].copy_(qwen_state_dict["model.norm.weight"])
        if not d16_cfg.tie_word_embeddings and "lm_head.weight" in qwen_state_dict:
            new_sd["lm_head.weight"].copy_(qwen_state_dict["lm_head.weight"])
        else:
            new_sd["lm_head.weight"].copy_(qwen_state_dict["model.embed_tokens.weight"])

        orig_inter = int(qwen_state_dict["model.layers.0.mlp.up_proj.weight"].shape[0])
        report = ConversionReport(
            mode=mode,
            num_layers=d16_cfg.num_hidden_layers,
            orig_intermediate_size=orig_inter,
            d16_intermediate_size=d16_cfg.intermediate_size,
        )

        for l_idx in range(d16_cfg.num_hidden_layers):
            prefix = f"model.layers.{l_idx}."

            new_sd[prefix + "input_layernorm.weight"].copy_(
                qwen_state_dict[prefix + "input_layernorm.weight"]
            )
            post_norm_w = qwen_state_dict[prefix + "post_attention_layernorm.weight"]
            new_sd[prefix + "post_attention_layernorm.weight"].copy_(post_norm_w)

            for proj in ("q_proj", "k_proj", "v_proj"):
                new_sd[prefix + f"self_attn.{proj}.weight"].copy_(
                    qwen_state_dict[prefix + f"self_attn.{proj}.weight"]
                )
                bias_key = prefix + f"self_attn.{proj}.bias"
                if bias_key in qwen_state_dict and bias_key in new_sd:
                    new_sd[bias_key].copy_(qwen_state_dict[bias_key])

            new_sd[prefix + "self_attn.o_proj.weight"].copy_(
                qwen_state_dict[prefix + "self_attn.o_proj.weight"]
            )

            u_w = qwen_state_dict[prefix + "mlp.up_proj.weight"].to(device)
            g_w = qwen_state_dict[prefix + "mlp.gate_proj.weight"].to(device)
            d_w = qwen_state_dict[prefix + "mlp.down_proj.weight"].to(device)

            mlp_params, layer_rep = convert_mlp_layer_to_d16(
                up_weight=u_w,
                gate_weight=g_w,
                down_weight=d_w,
                rms_weight=post_norm_w.to(device),
                layer_idx=l_idx,
                mode=mode,
            )
            report.layer_reports[l_idx] = layer_rep

            for k_param, val_tensor in mlp_params.items():
                new_sd[prefix + f"mlp.{k_param}"].copy_(val_tensor)

        d16_model.load_state_dict(new_sd)
        return d16_model, report

    @classmethod
    def from_pretrained_dir(
        cls,
        model_dir: Union[str, Path],
        mode: str = "orbit_4to1",
        indiff_eps: float = 0.0,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[D16QwenForCausalLM, ConversionReport]:
        """Load a local Qwen checkpoint directory (.safetensors or .bin) and convert to D16Qwen."""
        path = Path(model_dir)
        cfg_path = path / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"config.json not found in {path}")

        qwen_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        state_dict: Dict[str, torch.Tensor] = {}

        safetensors_files = sorted(path.glob("*.safetensors"))
        if safetensors_files:
            from safetensors.torch import load_file
            for sf in safetensors_files:
                state_dict.update(load_file(str(sf), device=str(device)))
        else:
            bin_files = sorted(path.glob("*.bin")) + sorted(path.glob("*.pt"))
            if not bin_files:
                raise FileNotFoundError(f"No .safetensors, .bin, or .pt weight files found in {path}")
            for bf in bin_files:
                state_dict.update(torch.load(str(bf), map_location=device))

        return cls.convert_state_dict(
            qwen_state_dict=state_dict,
            qwen_config_dict=qwen_cfg,
            mode=mode,
            indiff_eps=indiff_eps,
            device=device,
        )


__all__ = [
    "OP_NAMES",
    "OP_QUAD_SIGN",
    "RAYLEIGH_SIGMOID_KAPPA",
    "d16_group_matrices",
    "LayerOrbitReport",
    "ConversionReport",
    "compute_d16_pairwise_alignment",
    "select_d16_quadruplets",
    "compute_analytic_sector_gains",
    "convert_mlp_layer_to_d16",
    "QwenToD16Converter",
]