"""Converter and Orbit Analyzer from real Qwen2.5 weights to native D16QwenForCausalLM.

Principles:
1. Every neuron i in a Qwen MLP layer (0..4863) is an independent D16Complex object
   z_i = (x_i, y_i, rotation_i). The converter preserves all N = 4864 D16Complex
   objects 1-to-1 without shrinking or deleting neurons.
2. In parallel, the converter analyzes the Z_8 |>< Z_2 orbit structure across the
   N D16Complex objects (finding which independent D16Complex objects lie on the
   same 16-element dihedral orbit and differ in their rotation mask) and packages
   this into LayerOrbitReport / ConversionReport for visualization.
3. For experimental verification of orbit sharing, helper methods allow linking
   the top-M highest-coherence orbit pairs so that the partner D16Complex object
   at index j derives its state by a pure bitwise shift/reversal of the anchor's
   16-bit rotation mask (while remaining a separate D16Complex object at index j).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
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

OP_QUAD_SIGN: Tuple[float, ...] = (
    1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0,   # 0..7  (Axial Im branch)
    1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,   # 8..15 (Diagonal 45° Re branch)
)


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
# 2. DIAGNOSTIC DATA STRUCTURES FOR ORBIT VISUALIZATION
# =====================================================================

@dataclass
class LayerOrbitReport:
    """Diagnostic report of Z_8 |>< Z_2 orbit relationships among the N D16Complex objects."""

    layer_idx: int
    mode: str
    orig_intermediate_size: int
    d16_intermediate_size: int
    quadruplets: torch.Tensor        # [K, 4] indices of 4 D16Complex objects on the same local orbit
    group_ops: torch.Tensor          # [K, 4] relative Z_8 |>< Z_2 operation IDs (0..15)
    member_scales: torch.Tensor      # [K, 4] relative norm scales ||A_m|| / ||A_0||
    orbit_scores: torch.Tensor       # [K] alignment cosine score in [0, 1]
    local_2x2: torch.Tensor          # [K, 4, 2, 2] local 2D coordinates of each object in anchor's frame
    op_histogram: torch.Tensor       # [16] counts of operations 0..15 across discovered orbits
    hadamard_45_pct: float           # Percentage of orbit links using 45° Re(z^2) ops (8..15)
    mean_orbit_score: float

    def top_orbits(self, k: int = 5) -> List[Dict[str, Union[int, float, List[int], List[str]]]]:
        """Return human-readable summary of the top-k highest-coherence D16Complex orbit groups."""
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
# 3. ORBIT DISCOVERY AMONG INDEPENDENT D16Complex OBJECTS
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
    """Group N D16Complex objects into K=N//4 disjoint 4-element orbit clusters for inspection."""
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
def select_d16_orbit_pairs(
    best_scores: torch.Tensor,
    best_ops: torch.Tensor,
    max_pairs: int = 256,
    min_score: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select disjoint pairs (anchor_i, partner_j, op_id, score) of D16Complex objects on the same orbit."""
    N = best_scores.shape[0]
    device = best_scores.device
    upper = torch.triu(best_scores, diagonal=1)
    flat_scores, flat_idx = torch.sort(upper.flatten(), descending=True)

    used = torch.zeros(N, dtype=torch.bool, device=device)
    anchors: List[int] = []
    partners: List[int] = []
    ops: List[int] = []
    scores: List[float] = []

    for sc_t, idx_t in zip(flat_scores, flat_idx):
        sc = float(sc_t.item())
        if sc <= min_score or len(anchors) >= max_pairs:
            break
        idx = int(idx_t.item())
        i, j = idx // N, idx % N
        if bool(used[i]) or bool(used[j]):
            continue
        used[i] = True
        used[j] = True
        anchors.append(i)
        partners.append(j)
        ops.append(int(best_ops[i, j].item()))
        scores.append(sc)

    return (
        torch.tensor(anchors, dtype=torch.long, device=device),
        torch.tensor(partners, dtype=torch.long, device=device),
        torch.tensor(ops, dtype=torch.long, device=device),
        torch.tensor(scores, dtype=best_scores.dtype, device=device),
    )


# =====================================================================
# 4. LAYER CONVERSION (PRESERVING ALL N D16Complex OBJECTS)
# =====================================================================

@torch.no_grad()
def convert_mlp_layer_to_d16(
    up_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    rms_weight: Optional[torch.Tensor] = None,
    layer_idx: int = 0,
    mode: str = "native_d16",
    analyze_orbits: bool = True,
) -> Tuple[Dict[str, torch.Tensor], LayerOrbitReport]:
    """Convert a Qwen MLP layer into N independent D16Complex objects + orbit report.

    Every neuron i in 0..N-1 remains an independent D16Complex object.
    """
    device = up_weight.device
    orig_dtype = up_weight.dtype
    calc_dtype = torch.float32

    U = up_weight.detach().to(calc_dtype)
    G = gate_weight.detach().to(calc_dtype)
    D_mat = down_weight.detach().to(calc_dtype)
    N, hidden_size = U.shape

    params = {
        "up_proj.weight": U.to(orig_dtype).clone(),
        "gate_proj.weight": G.to(orig_dtype).clone(),
        "down_proj.weight": D_mat.to(orig_dtype).clone(),
    }

    if not analyze_orbits:
        empty_i = torch.zeros((0, 4), dtype=torch.long)
        report = LayerOrbitReport(
            layer_idx=layer_idx,
            mode=mode,
            orig_intermediate_size=N,
            d16_intermediate_size=N,
            quadruplets=empty_i,
            group_ops=empty_i,
            member_scales=torch.zeros((0, 4)),
            orbit_scores=torch.zeros(0),
            local_2x2=torch.zeros((0, 4, 2, 2)),
            op_histogram=torch.zeros(16, dtype=torch.long),
            hadamard_45_pct=0.0,
            mean_orbit_score=0.0,
        )
        return params, report

    if rms_weight is not None:
        rms_w = rms_weight.detach().to(device=device, dtype=calc_dtype)
    else:
        rms_w = torch.ones(hidden_size, dtype=calc_dtype, device=device)

    best_scores, best_ops = compute_d16_pairwise_alignment(U, G, rms_w)
    quads, q_ops, q_scores = select_d16_quadruplets(best_scores, best_ops)

    A_all = torch.stack([U, G], dim=1)
    norms_all = A_all.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    A_unit = A_all / norms_all

    # Express each of the 4 objects in the anchor object's 2D reference frame
    anchor_frame = A_unit[quads[:, 0]]  # [K, 2, D]
    member_raw = A_all[quads]           # [K, 4, 2, D]
    local_2x2 = torch.einsum("qad,qmbd->qmba", anchor_frame, member_raw)

    anchor_norms = norms_all[quads[:, 0], 0, 0].unsqueeze(1).clamp_min(1e-12)
    member_scales = norms_all[quads, 0, 0] / anchor_norms

    non_anchor_ops = q_ops[:, 1:].flatten()
    op_hist = torch.bincount(non_anchor_ops.cpu(), minlength=16)
    hadamard_pct = float((non_anchor_ops >= 8).float().mean().item() * 100.0)

    report = LayerOrbitReport(
        layer_idx=layer_idx,
        mode=mode,
        orig_intermediate_size=N,
        d16_intermediate_size=N,  # All N D16Complex objects are preserved!
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
    """Converter and Z_8 |>< Z_2 Orbit Inspector from Qwen2/Qwen2.5 to D16QwenForCausalLM."""

    @staticmethod
    def config_from_qwen_dict(
        qwen_cfg: Dict[str, Union[int, float, bool, str]],
        indiff_eps: float = 0.0,
    ) -> D16QwenConfig:
        """Build D16QwenConfig preserving all N D16Complex objects per layer."""
        return D16QwenConfig(
            vocab_size=int(qwen_cfg.get("vocab_size", 151936)),
            hidden_size=int(qwen_cfg.get("hidden_size", 896)),
            intermediate_size=int(qwen_cfg.get("intermediate_size", 4864)),
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
        mode: str = "native_d16",
        indiff_eps: float = 0.0,
        analyze_orbits: bool = True,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[D16QwenForCausalLM, ConversionReport]:
        """Convert an in-memory Qwen state_dict into D16QwenForCausalLM + ConversionReport."""
        d16_cfg = cls.config_from_qwen_dict(qwen_config_dict, indiff_eps=indiff_eps)
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
                analyze_orbits=analyze_orbits,
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
        mode: str = "native_d16",
        indiff_eps: float = 0.0,
        analyze_orbits: bool = True,
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
            analyze_orbits=analyze_orbits,
            device=device,
        )

    @staticmethod
    @torch.no_grad()
    def activate_bitwise_orbit_links(
        model: D16QwenForCausalLM,
        report: ConversionReport,
        pairs_per_layer: int = 0,
        min_orbit_score: float = 0.35,
    ) -> int:
        """Link top orbit pairs so partner D16Complex objects rotate anchor's mask via Z_8 |>< Z_2.

        Note: The number of D16Complex objects in every layer remains N (4864).
        When pairs_per_layer == 0, clears all links so every object uses its own (x, y, rotation).
        """
        total_linked = 0
        for l_idx, layer in enumerate(model.model.layers):
            if pairs_per_layer <= 0 or l_idx not in report.layer_reports:
                layer.mlp.clear_orbit_links()
                continue

            rep = report.layer_reports[l_idx]
            # First member (col 1) of each sorted quadruplet is the strongest partner of anchor (col 0)
            valid = rep.orbit_scores >= min_orbit_score
            k = min(pairs_per_layer, int(valid.sum().item()))
            if k <= 0:
                layer.mlp.clear_orbit_links()
                continue

            anchors = rep.quadruplets[:k, 0]
            partners = rep.quadruplets[:k, 1]
            ops = rep.group_ops[:k, 1]
            scales = rep.member_scales[:k, 1]
            layer.mlp.set_orbit_links(
                partner_idx=partners,
                anchor_idx=anchors,
                op_ids=ops,
                scales=scales,
            )
            total_linked += k
        return total_linked


__all__ = [
    "OP_NAMES",
    "OP_QUAD_SIGN",
    "d16_group_matrices",
    "LayerOrbitReport",
    "ConversionReport",
    "compute_d16_pairwise_alignment",
    "select_d16_quadruplets",
    "select_d16_orbit_pairs",
    "convert_mlp_layer_to_d16",
    "QwenToD16Converter",
]
