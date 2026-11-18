"""Standalone 1-to-1 Weight Importer and Fast D16 Checkpoint Save/Load for d16_qwen_model2.

This module keeps `d16_qwen_model2.py` completely untouched and defines:
1. `D16QwenImportConfig(D16QwenConfig)`: overrides `complex_hidden_size` so every
   D16Parameter matrix in `D16QwenForCausalLM` has the exact 1-to-1 shape of the
   reference Qwen model (`hidden_size`, `intermediate_size`, `vocab_size`).
2. `find_mlp_quartets`: fast Gram-matrix detector of Z_8 |>< Z_2 orbit quartets
   (n0, n1, n2, n3) above `min_orbit_score`. Neurons that do not form a quartet
   above the threshold remain unpaired.
3. `pack_weights_1to1`: converts any reference weight tensor `W` 1-to-1 into a
   `D16ComplexTensor` at the exact same coordinates `(layer, neuron, synapse)`
   using the standard `D16ComplexTensor.pack(re, im)` constructor:
   - Unpaired weights and `Re`-role quartet weights -> `pack(w, 0.0)`
   - `Im`-role quartet weights                      -> `pack(0.0, w)`
4. `import_qwen_weights`: builds a `D16QwenForCausalLM` instance and imports a
   reference Qwen model (or state_dict / file / directory) 1-to-1.
5. `save_d16_checkpoint` / `load_d16_checkpoint`: fast serialization and instant
   restoration of packed `D16QwenForCausalLM` models (supporting both directory
   checkpoints with `.safetensors` / `d16_config.json` and single `.pt` files)
   without re-running quartet detection or polar random initialization.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from d16_complex import D16ComplexTensor, INV_SQRT2, TAN_PI_8
from d16_parameter import D16Parameter
from d16_qwen_model2 import D16QwenConfig, D16QwenForCausalLM, IndiffThreshold, MLPActivation


# =====================================================================
# 1. 1-TO-1 IMPORT CONFIGURATION FOR d16_qwen_model2
# =====================================================================

@dataclass
class D16QwenImportConfig(D16QwenConfig):
    """Configuration for 1-to-1 Qwen weight import into `d16_qwen_model2.D16QwenForCausalLM`.

    Overrides `complex_hidden_size` to equal `self.hidden_size` so all weight matrices
    match the reference Qwen dimensions 1-to-1 without modifying `d16_qwen_model2.py`.
    """

    mlp_activation: MLPActivation = "gate"

    @property
    def complex_hidden_size(self) -> int:
        return self.hidden_size

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_qwen_config(
        cls,
        qwen_cfg: Any,
        *,
        num_hidden_layers: Optional[int] = None,
        indiff_eps: IndiffThreshold = 0.0,
        indiff_rel: float = 0.25,
        mlp_activation: MLPActivation = "gate",
        collect_d16_stats: bool = False,
    ) -> "D16QwenImportConfig":
        """Build a 1-to-1 D16QwenImportConfig from a HuggingFace Qwen config dict or object."""
        get = (
            (lambda k, d: qwen_cfg.get(k, d))
            if isinstance(qwen_cfg, Mapping)
            else (lambda k, d: getattr(qwen_cfg, k, d))
        )
        layers = int(get("num_hidden_layers", 24)) if num_hidden_layers is None else int(num_hidden_layers)
        return cls(
            vocab_size=int(get("vocab_size", 151936)),
            hidden_size=int(get("hidden_size", 896)),
            intermediate_size=int(get("intermediate_size", 4864)),
            num_hidden_layers=layers,
            num_attention_heads=int(get("num_attention_heads", 14)),
            num_key_value_heads=int(get("num_key_value_heads", 2)),
            rms_norm_eps=float(get("rms_norm_eps", 1e-6)),
            rope_theta=float(get("rope_theta", 1000000.0)),
            tie_word_embeddings=bool(get("tie_word_embeddings", True)),
            attention_bias=bool(get("attention_bias", True)),
            mlp_bias=bool(get("mlp_bias", False)),
            indiff_eps=indiff_eps,
            indiff_rel=indiff_rel,
            mlp_activation=mlp_activation,
            collect_d16_stats=collect_d16_stats,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        num_hidden_layers: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        indiff_eps: IndiffThreshold = 0.0,
        indiff_rel: float = 0.25,
        mlp_activation: MLPActivation = "gate",
        collect_d16_stats: bool = False,
    ) -> "D16QwenImportConfig":
        """Infer 1-to-1 D16QwenImportConfig dimensions directly from a Qwen state_dict."""
        embed_w = state_dict.get("model.embed_tokens.weight", state_dict.get("embed_tokens.weight"))
        if embed_w is None:
            raise ValueError("state_dict must contain 'model.embed_tokens.weight'")
        if not isinstance(embed_w, torch.Tensor) or embed_w.ndim != 2 or min(embed_w.shape) < 1:
            raise ValueError("model.embed_tokens.weight must be a nonempty matrix")
        vocab_size, hidden_size = int(embed_w.shape[0]), int(embed_w.shape[1])

        up_w = state_dict.get("model.layers.0.mlp.up_proj.weight", state_dict.get("layers.0.mlp.up_proj.weight"))
        if up_w is None:
            raise ValueError("state_dict must contain 'model.layers.0.mlp.up_proj.weight'")
        if not isinstance(up_w, torch.Tensor) or up_w.ndim != 2:
            raise ValueError("model.layers.0.mlp.up_proj.weight must be a matrix")
        intermediate_size = int(up_w.shape[0])

        layer_indices = set()
        for key in state_dict.keys():
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_indices.add(int(parts[i + 1]))
        detected_layers = max(layer_indices) + 1 if layer_indices else 1
        layers = detected_layers if num_hidden_layers is None else int(num_hidden_layers)

        k_w = state_dict.get("model.layers.0.self_attn.k_proj.weight", state_dict.get("layers.0.self_attn.k_proj.weight"))
        if k_w is not None and (not isinstance(k_w, torch.Tensor) or k_w.ndim != 2):
            raise ValueError("model.layers.0.self_attn.k_proj.weight must be a matrix")
        keys = {name.removeprefix("model.") for name in state_dict}

        def has_bias(module: str, projections: tuple[str, ...]) -> bool:
            return any(
                f"layers.{idx}.{module}.{proj}.bias" in keys
                for idx in range(layers) for proj in projections
            )

        kv_dim = int(k_w.shape[0]) if k_w is not None else hidden_size
        if num_attention_heads is None:
            if hidden_size == 896:
                num_attention_heads = 14
            elif hidden_size % 64 == 0:
                num_attention_heads = hidden_size // 64
            else:
                num_attention_heads = max(1, hidden_size // max(kv_dim, 2))
        if num_attention_heads <= 0 or hidden_size % num_attention_heads:
            raise ValueError("num_attention_heads must be positive and divide hidden_size")
        head_dim = hidden_size // num_attention_heads
        num_key_value_heads = max(1, kv_dim // head_dim)

        lm_w = state_dict.get("lm_head.weight")
        if lm_w is not None and not isinstance(lm_w, torch.Tensor):
            raise ValueError("lm_head.weight must be a tensor")
        tie_word_embeddings = (lm_w is None) or bool(
            lm_w.shape == embed_w.shape and torch.equal(lm_w.cpu(), embed_w.cpu())
        )

        return cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
            attention_bias=has_bias("self_attn", ("q_proj", "k_proj", "v_proj")),
            mlp_bias=has_bias("mlp", ("up_proj", "gate_proj", "down_proj")),
            indiff_eps=indiff_eps,
            indiff_rel=indiff_rel,
            mlp_activation=mlp_activation,
            collect_d16_stats=collect_d16_stats,
        )


# =====================================================================
# 2. CHECKPOINT FILE & DIRECTORY LOADER (.safetensors / .bin / HF)
# =====================================================================

def load_qwen_checkpoint(
    source: Union[str, Path, nn.Module, Mapping[str, torch.Tensor]],
    *,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Mapping[str, torch.Tensor], Optional[Dict[str, Any]]]:
    """Load a Qwen state_dict and optional config.json from a file, directory, HF repo ID, or module."""
    if isinstance(source, nn.Module):
        cfg_dict = vars(source.config) if hasattr(source, "config") else None
        return source.state_dict(), cfg_dict
    if isinstance(source, Mapping):
        return source, None

    path = Path(source).expanduser()
    cfg_dict: Optional[Dict[str, Any]] = None

    # 1. Local directory containing model.safetensors / sharded safetensors / pytorch_model.bin
    if path.is_dir():
        cfg_file = path / "config.json"
        if cfg_file.is_file():
            cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8"))

        st_single = path / "model.safetensors"
        st_shards = sorted(path.glob("model-*-of-*.safetensors"))
        bin_single = path / "pytorch_model.bin"
        bin_shards = sorted(path.glob("pytorch_model-*-of-*.bin"))
        pt_files = sorted(path.glob("*.safetensors")) or sorted(path.glob("*.bin")) or sorted(path.glob("*.pt"))

        if st_single.is_file():
            from safetensors.torch import load_file
            return load_file(str(st_single), device=(f"cuda:{torch.cuda.current_device()}" if str(device) == "cuda" else str(device))), cfg_dict
        if st_shards:
            from safetensors.torch import load_file
            merged: Dict[str, torch.Tensor] = {}
            for shard in st_shards:
                merged.update(load_file(str(shard), device=(f"cuda:{torch.cuda.current_device()}" if str(device) == "cuda" else str(device))))
            return merged, cfg_dict
        if bin_single.is_file():
            return torch.load(str(bin_single), map_location=device, weights_only=True), cfg_dict
        if bin_shards:
            merged = {}
            for shard in bin_shards:
                merged.update(torch.load(str(shard), map_location=device, weights_only=True))
            return merged, cfg_dict
        if pt_files:
            f0 = pt_files[0]
            if f0.suffix == ".safetensors":
                from safetensors.torch import load_file
                return load_file(str(f0), device=(f"cuda:{torch.cuda.current_device()}" if str(device) == "cuda" else str(device))), cfg_dict
            return torch.load(str(f0), map_location=device, weights_only=True), cfg_dict
        raise FileNotFoundError(f"No .safetensors or .bin checkpoint files found in directory: {path}")

    # 2. Direct path to a .safetensors, .bin, or .pt file
    if path.is_file():
        cfg_file = path.parent / "config.json"
        if cfg_file.is_file():
            cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8"))
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(path), device=(f"cuda:{torch.cuda.current_device()}" if str(device) == "cuda" else str(device))), cfg_dict
        raw = torch.load(str(path), map_location=device, weights_only=True)
        if isinstance(raw, Mapping) and "state_dict" in raw and isinstance(raw["state_dict"], Mapping):
            return raw["state_dict"], cfg_dict
        return raw, cfg_dict

    # 3. Fallback: HuggingFace Hub model ID (e.g. "Qwen/Qwen2.5-0.5B")
    from transformers import AutoConfig, AutoModelForCausalLM
    hf_cfg = AutoConfig.from_pretrained(str(source))
    hf_model = AutoModelForCausalLM.from_pretrained(str(source), torch_dtype="auto", device_map=str(device))
    return hf_model.state_dict(), vars(hf_cfg)


# =====================================================================
# 3. ORBIT TELEMETRY REPORTS (COMPATIBLE WITH d16_vis_utils.py)
# =====================================================================

@dataclass
class LayerOrbitReport:
    """Per-layer report of detected Z_8 |>< Z_2 quartets and Re/Im role assignments."""

    layer_idx: int
    mode: str
    orig_intermediate_size: int
    d16_intermediate_size: int
    quadruplets: torch.Tensor          # [Q, 4] neuron indices of all candidate quartets
    group_ops: torch.Tensor            # [Q, 4] Z_8 |>< Z_2 operation IDs in 0..15
    orbit_scores: torch.Tensor         # [Q] mean cosine similarity per quartet
    active_quartet_mask: torch.Tensor  # [Q] True where quartet score >= min_orbit_score
    is_im_neuron: torch.Tensor         # [N] True for neurons assigned to Im (placed in y via pack(0, w))
    paired_neuron_mask: torch.Tensor   # [N] True for neurons belonging to an active quartet
    op_histogram: torch.Tensor         # [16] histogram of Z_8 |>< Z_2 operations
    local_2x2: torch.Tensor            # [Q, 4, 2, 2] local 2D projections for compass plots

    @property
    def mean_orbit_score(self) -> float:
        if self.orbit_scores.numel() == 0:
            return 0.0
        active = self.orbit_scores[self.active_quartet_mask]
        target = active if active.numel() > 0 else self.orbit_scores
        return float(target.mean().item())

    @property
    def hadamard_45_pct(self) -> float:
        tot = int(self.op_histogram[1:].sum().item())
        if tot == 0:
            return 0.0
        had = int(self.op_histogram[8:].sum().item())
        return 100.0 * had / tot

    @property
    def active_quartets_count(self) -> int:
        return int(self.active_quartet_mask.sum().item())

    @property
    def paired_neurons_count(self) -> int:
        return int(self.paired_neuron_mask.sum().item())

    @property
    def unpaired_neurons_count(self) -> int:
        return int((~self.paired_neuron_mask).sum().item())

    @property
    def im_neurons_count(self) -> int:
        return int(self.is_im_neuron.sum().item())

    @property
    def re_neurons_count(self) -> int:
        return int((~self.is_im_neuron).sum().item())

    def top_orbits(self, k: int = 10) -> List[Dict[str, Any]]:
        if self.orbit_scores.numel() == 0:
            return []
        k_eff = min(k, int(self.orbit_scores.numel()))
        vals, idxs = torch.topk(self.orbit_scores, k=k_eff)
        results: List[Dict[str, Any]] = []
        for sc, q_idx in zip(vals.tolist(), idxs.tolist()):
            results.append(
                {
                    "unit_idx": int(q_idx),
                    "score": float(sc),
                    "active": bool(self.active_quartet_mask[q_idx].item()),
                    "neurons": self.quadruplets[q_idx].tolist(),
                    "ops": self.group_ops[q_idx].tolist(),
                }
            )
        return results


@dataclass
class ConversionReport:
    """Full model import report across all layers."""

    min_orbit_score: float
    layer_reports: Dict[int, LayerOrbitReport] = field(default_factory=dict)
    total_imported_tensors: int = 0
    total_weights: int = 0
    total_re_weights: int = 0
    total_im_weights: int = 0


# =====================================================================
# 4. FAST GRAM-MATRIX Z_8 |>< Z_2 QUARTET DETECTION
# =====================================================================

@torch.no_grad()
def find_mlp_quartets(
    up_weight: torch.Tensor,
    gate_weight: Optional[torch.Tensor] = None,
    down_weight: Optional[torch.Tensor] = None,
    *,
    layer_idx: int = 0,
    min_orbit_score: float = 0.25,
) -> LayerOrbitReport:
    """Fast Gram-matrix Z_8 |>< Z_2 quartet finder for full-size Qwen MLP layers."""
    device = up_weight.device
    u = up_weight.detach().to(torch.float32)
    if gate_weight is not None:
        g = gate_weight.detach().to(torch.float32)
    elif down_weight is not None and down_weight.shape[1] == u.shape[0]:
        g = down_weight.detach().to(torch.float32).transpose(0, 1)
    else:
        g = torch.roll(u, shifts=1, dims=-1)

    n_neurons = u.shape[0]
    num_candidate_quartets = n_neurons // 4

    norms = (u.square().sum(dim=-1, keepdim=True) + g.square().sum(dim=-1, keepdim=True)).sqrt().clamp_min(1e-8)
    u_unit = u / norms
    g_unit = g / norms

    # Compute the 3 Gram matrices in R^{N x N}:
    s_uu = u_unit @ u_unit.transpose(0, 1)
    s_gg = g_unit @ g_unit.transpose(0, 1)
    s_ug = u_unit @ g_unit.transpose(0, 1)
    s_gu = s_ug.transpose(0, 1)

    A = s_uu + s_gg
    B = s_uu - s_gg
    C = s_ug - s_gu
    D = s_ug + s_gu
    del s_uu, s_gg, s_ug, s_gu

    # Evaluate the best dihedral operation (1..15) for every pair (i, j)
    best_sim = -A.clone()  # op 1: (-u, -g)
    best_op = torch.ones((n_neurons, n_neurons), dtype=torch.uint8, device=device)

    ops_exprs = (
        (2, B),
        (3, -B),
        (4, C),
        (5, -C),
        (6, D),
        (7, -D),
        (8, (A + C) * INV_SQRT2),
        (9, (-A + C) * INV_SQRT2),
        (10, (-A - C) * INV_SQRT2),
        (11, (A - C) * INV_SQRT2),
        (12, (B + D) * INV_SQRT2),
        (13, (-B + D) * INV_SQRT2),
        (14, (-B - D) * INV_SQRT2),
        (15, (B - D) * INV_SQRT2),
    )
    for op_id, mat in ops_exprs:
        better = mat > best_sim
        best_sim = torch.where(better, mat, best_sim)
        best_op = torch.where(better, op_id, best_op)
    del A, B, C, D

    best_sim.fill_diagonal_(-float("inf"))
    init_scores = (
        best_sim.topk(k=3, dim=-1).values.mean(dim=-1)
        if n_neurons >= 4
        else torch.zeros(n_neurons, device=device)
    )
    neuron_order = init_scores.argsort(descending=True).tolist()

    used = torch.zeros(n_neurons, dtype=torch.bool, device=device)
    quads: List[List[int]] = []
    ops_list: List[List[int]] = []
    scores_list: List[float] = []
    op_hist = torch.zeros(16, dtype=torch.long)

    ptr = 0
    for _ in range(num_candidate_quartets):
        while ptr < n_neurons and bool(used[neuron_order[ptr]]):
            ptr += 1
        if ptr >= n_neurons or int((~used).sum().item()) < 4:
            break
        anchor = neuron_order[ptr]
        used[anchor] = True

        row_sim = best_sim[anchor].masked_fill(used, -float("inf"))
        top_vals, top_partners = torch.topk(row_sim, k=3)
        p1, p2, p3 = [int(x) for x in top_partners.tolist()]
        op1 = int(best_op[anchor, p1].item())
        op2 = int(best_op[anchor, p2].item())
        op3 = int(best_op[anchor, p3].item())
        used[p1] = True
        used[p2] = True
        used[p3] = True

        quads.append([anchor, p1, p2, p3])
        ops_list.append([0, op1, op2, op3])
        scores_list.append(float(top_vals.mean().item()))

    if quads:
        quad_t = torch.tensor(quads, dtype=torch.long)
        ops_t = torch.tensor(ops_list, dtype=torch.long)
        scores_t = torch.tensor(scores_list, dtype=torch.float32)

        # Vectorized local 2x2 coordinates in each anchor's orthonormal frame (for compass plots)
        anchors_idx = quad_t[:, 0].to(device)
        e1 = F.normalize(u_unit[anchors_idx], dim=-1)
        a_g = g_unit[anchors_idx]
        e2 = a_g - (a_g * e1).sum(dim=-1, keepdim=True) * e1
        e2 = F.normalize(e2, dim=-1)
        basis = torch.stack([e1, e2], dim=1)  # [Q, 2, H]

        q_dev = quad_t.to(device)
        mem_u = u[q_dev]  # [Q, 4, H]
        mem_g = g[q_dev]  # [Q, 4, H]
        mem_planes = torch.stack([mem_u, mem_g], dim=2)  # [Q, 4, 2, H]
        loc_t = torch.einsum("qmrH,qkH->qmrk", mem_planes, basis).cpu()
    else:
        quad_t = torch.zeros((0, 4), dtype=torch.long)
        ops_t = torch.zeros((0, 4), dtype=torch.long)
        scores_t = torch.zeros((0,), dtype=torch.float32)
        loc_t = torch.zeros((0, 4, 2, 2), dtype=torch.float32)

    active_mask = scores_t >= float(min_orbit_score)
    is_im_neuron = torch.zeros(n_neurons, dtype=torch.bool)
    paired_neuron_mask = torch.zeros(n_neurons, dtype=torch.bool)

    for q_idx in range(quad_t.shape[0]):
        if not bool(active_mask[q_idx].item()):
            continue
        members = quad_t[q_idx].tolist()
        ops = ops_t[q_idx].tolist()
        for m_idx, nid in enumerate(members):
            paired_neuron_mask[nid] = True
            if m_idx > 0:
                op_hist[ops[m_idx]] += 1

        # ops 1..7 are axial (Im), ops 8..15 are 45° Hadamard (Re), anchor m=0 is Re
        im_flags = [False, ops[1] < 8, ops[2] < 8, ops[3] < 8]
        if not any(im_flags):
            im_flags[1] = True
            im_flags[3] = True
        for nid, is_im in zip(members, im_flags):
            is_im_neuron[nid] = is_im

    return LayerOrbitReport(
        layer_idx=layer_idx,
        mode="1to1_quartet_re_im",
        orig_intermediate_size=n_neurons,
        d16_intermediate_size=n_neurons,
        quadruplets=quad_t,
        group_ops=ops_t,
        orbit_scores=scores_t,
        active_quartet_mask=active_mask,
        is_im_neuron=is_im_neuron,
        paired_neuron_mask=paired_neuron_mask,
        op_histogram=op_hist,
        local_2x2=loc_t,
    )


# =====================================================================
# 5. STANDARD 1-TO-1 WEIGHT PACKING & MODEL IMPORT (TILED & VALIDATED)
# =====================================================================

_PACK_ROW_TILE: int = 4096

# Фундаментальные геометрические константы реконструкции 2-й координаты в клине [0°, 22.5°]:
# - WEDGE_SAT_SLOPE = TAN_PI_8 / 4 : четверть наклона биссектрисы (после билинейного умножения даёт середину клина TAN_PI_8 / 2)
# - WEDGE_HALF_SAT  = (TAN_PI_8 ** 2) * INV_SQRT2 : радиус полунасыщения в единицах глобального RMS матрицы
WEDGE_SAT_SLOPE: float = 0.25 * TAN_PI_8
WEDGE_HALF_SAT: float = (TAN_PI_8 ** 2) * INV_SQRT2


@torch.no_grad()
def compute_d16_second_coord(
    w: torch.Tensor,
    mode: str = "d16_sat_rope_gate",
    *,
    partner_w: Optional[torch.Tensor] = None,
    c_sat: float = WEDGE_SAT_SLOPE,
    c_w0: float = WEDGE_HALF_SAT,
    sign: float = 1.0,
) -> torch.Tensor:
    """Compute the complementary D16 coordinate w2 = f(w1) via D16 radial saturation."""
    if mode == "zero":
        return torch.zeros_like(w)

    wf = w.detach().to(torch.float32)
    sigma_w = wf.square().mean().sqrt().clamp_min(1e-8)
    y_sat = c_sat * sigma_w
    w0 = c_w0 * sigma_w
    envelope = y_sat * (wf.abs() / (w0 + wf.abs()))

    if mode == "d16_sat_rand":
        xi = torch.randn_like(wf)
    elif mode == "d16_sat_det":
        if wf.shape[-1] >= 3:
            dw = torch.roll(wf, -1, dims=-1) - torch.roll(wf, 1, dims=-1)
        else:
            dw = wf.clone()
        dw_std = dw.square().mean().sqrt().clamp_min(1e-8)
        # Шахматная чётность (-1)^(i+j) гарантирует идеальную симметрию S0=S15 и S7=S8
        idx_last = torch.arange(wf.shape[-1], device=wf.device)
        if wf.ndim >= 2:
            idx_prev = torch.arange(wf.shape[-2], device=wf.device).unsqueeze(-1)
            parity = torch.where(((idx_prev + idx_last) & 1) == 0, 1.0, -1.0)
        else:
            parity = torch.where((idx_last & 1) == 0, 1.0, -1.0)
        xi = (sign * (dw / dw_std) * parity).clamp(-3.0, 3.0)
    elif mode == "d16_sat_rope_gate":
        if partner_w is not None and partner_w.shape == wf.shape:
            pw = partner_w.detach().to(device=wf.device, dtype=torch.float32)
            pw_std = pw.square().mean().sqrt().clamp_min(1e-8)
        else:
            w_pairs = wf.view(*wf.shape[:-1], wf.shape[-1] // 2, 2)
            pw = torch.stack([-w_pairs[..., 1], w_pairs[..., 0]], dim=-1).reshape_as(wf)
            pw_std = sigma_w
        xi = (sign * (pw / pw_std)).clamp(-3.0, 3.0)
    else:
        raise ValueError(f"Unknown second_coord_mode: {mode!r}")

    return (envelope * xi).to(dtype=w.dtype)


@torch.no_grad()
def _rope_head_half_partner(w: torch.Tensor, head_dim: int) -> Optional[torch.Tensor]:
    """Pair each attention head's first half r in [0..head_dim//2) with its RoPE conjugate half r + head_dim//2."""
    if head_dim <= 0 or (head_dim % 2) != 0 or w.ndim < 1 or (w.shape[0] % head_dim) != 0:
        return None
    half = head_dim // 2
    w_heads = w.view(-1, 2, half, *w.shape[1:])
    return torch.cat((w_heads[:, 1], -w_heads[:, 0]), dim=1).reshape_as(w)


@torch.no_grad()
def _compute_w2_slice(
    w_slice: torch.Tensor,
    mode: str,
    *,
    sigma_w: torch.Tensor,
    partner_slice: Optional[torch.Tensor] = None,
    sigma_partner: Optional[torch.Tensor] = None,
    row_offset: int = 0,
    c_sat: float = WEDGE_SAT_SLOPE,
    c_w0: float = WEDGE_HALF_SAT,
    sign: float = 1.0,
) -> torch.Tensor:
    """Compute 2nd coordinate w2 for a row slice using global matrix statistics (sigma_w, sigma_partner)."""
    if mode == "zero" or c_sat == 0.0 or (w_slice.ndim < 2 and partner_slice is None):
        return torch.zeros_like(w_slice)

    wf = w_slice.to(torch.float32)
    y_sat = c_sat * sigma_w
    w0 = c_w0 * sigma_w
    envelope = y_sat * (wf.abs() / (w0 + wf.abs()))

    if mode == "d16_sat_rand":
        xi = torch.randn_like(wf)
    elif mode == "d16_sat_det":
        dw = torch.roll(wf, -1, dims=-1) - torch.roll(wf, 1, dims=-1) if wf.shape[-1] >= 3 else wf
        dw_std = (sigma_partner if sigma_partner is not None else dw.square().mean().sqrt()).clamp_min(1e-8)
        idx_r = torch.arange(row_offset, row_offset + wf.shape[-2], device=wf.device).unsqueeze(-1)
        idx_c = torch.arange(wf.shape[-1], device=wf.device)
        parity = torch.where(((idx_r + idx_c) & 1) == 0, 1.0, -1.0)
        xi = (sign * (dw / dw_std) * parity).clamp(-3.0, 3.0)
    elif mode == "d16_sat_rope_gate":
        if partner_slice is not None and partner_slice.shape == wf.shape:
            pw = partner_slice.to(device=wf.device, dtype=torch.float32)
            pw_std = (sigma_partner if sigma_partner is not None else pw.square().mean().sqrt()).clamp_min(1e-8)
        else:
            w_pairs = wf.view(*wf.shape[:-1], wf.shape[-1] // 2, 2)
            pw = torch.stack([-w_pairs[..., 1], w_pairs[..., 0]], dim=-1).reshape_as(wf)
            pw_std = sigma_w.clamp_min(1e-8)
        xi = (sign * (pw / pw_std)).clamp(-3.0, 3.0)
    else:
        raise ValueError(f"Unknown second_coord_mode: {mode!r}")

    return (envelope * xi).to(dtype=w_slice.dtype)


@torch.no_grad()
def _compute_global_sigmas_tiled(
    w: torch.Tensor,
    mode: str,
    partner_w: Optional[torch.Tensor] = None,
    *,
    dtype: torch.dtype,
    c_sat: float = WEDGE_SAT_SLOPE,
    row_tile: int = _PACK_ROW_TILE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate global RMS in tiles after rounding each input to the destination dtype."""
    if mode == "zero" or c_sat == 0.0 or (w.ndim < 2 and partner_w is None):
        one = torch.ones((), dtype=torch.float32, device=w.device)
        return one, one

    sq_w = torch.zeros((), dtype=torch.float64, device=w.device)
    sq_p = torch.zeros((), dtype=torch.float64, device=w.device)
    n_rows = w.shape[0]

    for start in range(0, n_rows, row_tile):
        end = min(start + row_tile, n_rows)
        sl = w[start:end].to(dtype).float()
        sq_w += sl.square().sum(dtype=torch.float64)
        if mode == "d16_sat_det" and sl.shape[-1] >= 3:
            dw = torch.roll(sl, -1, dims=-1) - torch.roll(sl, 1, dims=-1)
            sq_p += dw.square().sum(dtype=torch.float64)
        elif mode == "d16_sat_rope_gate" and partner_w is not None:
            psl = partner_w[start:end].to(device=w.device, dtype=dtype).float()
            sq_p += psl.square().sum(dtype=torch.float64)

    numel = max(w.numel(), 1)
    sigma_w = (sq_w / numel).sqrt().to(torch.float32).clamp_min(1e-8)
    sigma_p = (
        (sq_p / numel).sqrt().to(torch.float32).clamp_min(1e-8)
        if float(sq_p.item()) > 0.0
        else sigma_w
    )
    return sigma_w, sigma_p


@torch.no_grad()
def pack_weights_1to1(
    weight: torch.Tensor,
    is_im_mask: Optional[torch.Tensor] = None,
    *,
    im_dim: int = 0,
    second_coord_mode: str = "zero",
    partner_weight: Optional[torch.Tensor] = None,
    c_sat: float = WEDGE_SAT_SLOPE,
    c_w0: float = WEDGE_HALF_SAT,
    sign: float = 1.0,
    row_tile: int = _PACK_ROW_TILE,
    out: Optional[D16ComplexTensor] = None,
) -> D16ComplexTensor:
    """Pack row tiles directly into out, using its dtype and device for the imported weights."""
    w = weight.detach()
    if row_tile <= 0:
        raise ValueError("row_tile must be positive")
    if out is None:
        out = D16ComplexTensor.zeros(w.shape, dtype=w.dtype, device=w.device)
    elif out.shape != w.shape:
        raise ValueError(f"Source shape {tuple(w.shape)} does not match destination {tuple(out.shape)}")
    if second_coord_mode != "d16_sat_rope_gate":
        partner_weight = None
    elif partner_weight is not None and partner_weight.shape != w.shape:
        raise ValueError("partner_weight must have the same shape as weight")

    sigma_w, sigma_partner = _compute_global_sigmas_tiled(
        w, second_coord_mode, partner_weight, dtype=out.dtype, c_sat=c_sat, row_tile=row_tile,
    )
    sigma_w, sigma_partner = sigma_w.to(out.device), sigma_partner.to(out.device)
    mask = None
    if is_im_mask is not None and bool(is_im_mask.any()):
        mask = is_im_mask.to(device=out.device, dtype=torch.bool)
        if w.ndim == 2 and mask.ndim == 1:
            mask = mask.unsqueeze(1 if im_dim == 0 else 0)
        mask = mask.expand(w.shape)

    n_rows = w.shape[0] if w.ndim else 1
    for start in range(0, n_rows, row_tile):
        rows = slice(start, min(start + row_tile, n_rows)) if w.ndim else Ellipsis
        w_sl = w[rows].to(device=out.device, dtype=out.dtype)
        p_sl = None if partner_weight is None else partner_weight[rows].to(device=out.device, dtype=out.dtype)
        w2_sl = _compute_w2_slice(
            w_sl,
            second_coord_mode,
            sigma_w=sigma_w,
            partner_slice=p_sl,
            sigma_partner=sigma_partner,
            row_offset=start if w.ndim == 2 else 0,
            c_sat=c_sat,
            c_w0=c_w0,
            sign=sign,
        )
        if mask is None:
            packed_sl = D16ComplexTensor.pack(w_sl, w2_sl)
        else:
            m_sl = mask[rows]
            packed_sl = D16ComplexTensor.pack(torch.where(m_sl, w2_sl, w_sl), torch.where(m_sl, w_sl, w2_sl))
        out.copy_slice_(rows, packed_sl)

    return out


@contextmanager
def _fast_empty_d16_init(dtype: torch.dtype, device: Union[str, torch.device]):
    """Skip expensive random polar sampling when constructing a D16 model for weight import or checkpoint load."""
    orig_random = D16Parameter.random_matrix
    orig_zeros = D16Parameter.zeros
    orig_ones = D16Parameter.ones

    @classmethod  # type: ignore[misc]
    def _fast_zeros(cls, shape: tuple[int, ...]) -> D16Parameter:
        return cls(D16ComplexTensor.zeros(shape, dtype=dtype, device=device))

    @classmethod  # type: ignore[misc]
    def _fast_ones(cls, shape: tuple[int, ...]) -> D16Parameter:
        return cls(D16ComplexTensor.ones(shape, dtype=dtype, device=device))

    @classmethod  # type: ignore[misc]
    def _fast_rand(cls, shape: tuple[int, int], std: float, *, normal: bool) -> D16Parameter:
        return cls(D16ComplexTensor.zeros(shape, dtype=dtype, device=device))

    D16Parameter.zeros = _fast_zeros  # type: ignore[assignment]
    D16Parameter.ones = _fast_ones  # type: ignore[assignment]
    D16Parameter.random_matrix = _fast_rand  # type: ignore[assignment]
    try:
        yield
    finally:
        D16Parameter.zeros = orig_zeros  # type: ignore[assignment]
        D16Parameter.ones = orig_ones  # type: ignore[assignment]
        D16Parameter.random_matrix = orig_random  # type: ignore[assignment]


def _get_tensor(
    state_dict: Mapping[str, torch.Tensor], *keys: str
) -> Optional[torch.Tensor]:
    for k in keys:
        if k in state_dict:
            return state_dict[k]
    return None


def _require_tensor(
    state_dict: Mapping[str, torch.Tensor], *keys: str
) -> torch.Tensor:
    t = _get_tensor(state_dict, *keys)
    if t is None:
        raise KeyError(f"Missing required Qwen checkpoint tensor (tried keys: {keys})")
    return t


def _validate_qwen_state_dict(state_dict: Mapping[str, torch.Tensor], config: D16QwenConfig) -> None:
    """Check the complete shapes and bias layout of the requested layers before allocating the model."""
    errors: list[str] = []

    def check(name: str, shape: Optional[tuple[int, ...]], *, required: bool = True) -> None:
        tensor = _get_tensor(state_dict, name, name.removeprefix("model."))
        if tensor is None:
            if required and shape is not None:
                errors.append(f"missing {name}")
        elif shape is None:
            errors.append(f"unexpected {name} for this bias configuration")
        elif not isinstance(tensor, torch.Tensor):
            errors.append(f"{name} must be a tensor")
        elif tensor.shape != shape:
            errors.append(f"{name}: expected {shape}, got {tuple(tensor.shape)}")

    hidden = config.complex_hidden_size
    intermediate = config.intermediate_size
    if config.num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be positive")
    kv_width = config.head_dim * config.num_key_value_heads
    embedding_shape = (config.vocab_size, hidden)
    check("model.embed_tokens.weight", embedding_shape)
    check("model.norm.weight", (hidden,))
    check("lm_head.weight", embedding_shape, required=not config.tie_word_embeddings)

    for idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{idx}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            check(f"{prefix}.{norm}.weight", (hidden,))
        for proj, width in (("q_proj", hidden), ("k_proj", kv_width), ("v_proj", kv_width), ("o_proj", hidden)):
            name = f"{prefix}.self_attn.{proj}"
            check(f"{name}.weight", (width, hidden))
            check(f"{name}.bias", (width,) if config.attention_bias and proj != "o_proj" else None)
        for proj in ("up_proj", "gate_proj", "down_proj"):
            name = f"{prefix}.mlp.{proj}"
            required = proj != "gate_proj" or config.mlp_activation in ("bilinear", "gate")
            shape = (hidden, intermediate) if proj == "down_proj" else (intermediate, hidden)
            check(f"{name}.weight", shape, required=required)
            check(f"{name}.bias", (shape[0],) if config.mlp_bias else None, required=required)

    if errors:
        raise ValueError("Invalid Qwen checkpoint: " + "; ".join(errors))


@torch.no_grad()
def import_qwen_weights(
    source: Union[str, Path, nn.Module, Mapping[str, torch.Tensor]],
    config: Optional[D16QwenImportConfig] = None,
    *,
    num_hidden_layers: Optional[int] = None,
    min_orbit_score: float = 0.25,
    indiff_eps: IndiffThreshold = 0.0,
    indiff_rel: float = 0.25,
    mlp_activation: MLPActivation = "gate",
    second_coord_mode: str = "zero",
    c_sat: float = WEDGE_SAT_SLOPE,
    c_w0: float = WEDGE_HALF_SAT,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
) -> Tuple[D16QwenForCausalLM, ConversionReport]:
    """Import weights 1-to-1 from a Qwen file/folder/model/state_dict into `d16_qwen_model2.D16QwenForCausalLM`."""
    # Load raw checkpoint onto CPU first to avoid doubling GPU VRAM during import:
    state_dict, cfg_dict = load_qwen_checkpoint(source, device="cpu")

    if config is None:
        if cfg_dict is not None:
            config = D16QwenImportConfig.from_qwen_config(
                cfg_dict,
                num_hidden_layers=num_hidden_layers,
                indiff_eps=indiff_eps,
                indiff_rel=indiff_rel,
                mlp_activation=mlp_activation,
            )
        else:
            config = D16QwenImportConfig.from_state_dict(
                state_dict,
                num_hidden_layers=num_hidden_layers,
                indiff_eps=indiff_eps,
                indiff_rel=indiff_rel,
                mlp_activation=mlp_activation,
            )

    # Validate checkpoint completeness BEFORE allocating or populating the D16 model:
    _validate_qwen_state_dict(state_dict, config)

    emb_w = _require_tensor(state_dict, "model.embed_tokens.weight", "embed_tokens.weight")
    target_dtype = emb_w.dtype if dtype is None else dtype
    target_device = emb_w.device if device is None else torch.device(device)

    with _fast_empty_d16_init(dtype=target_dtype, device=target_device):
        d16_model = D16QwenForCausalLM(config)
    d16_model = d16_model.to(device=target_device)
    if config.complex_hidden_size == config.hidden_size and (config.head_dim % 2) == 0:
        even_idx = torch.arange(0, config.head_dim, 2, device=target_device)
        d16_model.model.rotary_emb.frequency_indices.copy_(torch.cat((even_idx, even_idx)))

    report = ConversionReport(min_orbit_score=min_orbit_score)

    def _load_param(
        param: D16Parameter,
        w: torch.Tensor,
        is_im_mask: Optional[torch.Tensor] = None,
        im_dim: int = 0,
        *,
        mode: str = "zero",
        partner_w: Optional[torch.Tensor] = None,
        sign: float = 1.0,
    ) -> None:
        n_el = w.numel()
        pack_weights_1to1(
            w, is_im_mask, im_dim=im_dim, second_coord_mode=mode, partner_weight=partner_w,
            c_sat=c_sat, c_w0=c_w0, sign=sign, row_tile=_PACK_ROW_TILE, out=param.value(),
        )

        report.total_imported_tensors += 1
        report.total_weights += n_el
        if is_im_mask is not None and bool(is_im_mask.any()):
            if w.ndim == 2 and is_im_mask.ndim == 1:
                im_cnt = int(is_im_mask.sum().item()) * (w.shape[1] if im_dim == 0 else w.shape[0])
            else:
                im_cnt = int(is_im_mask.sum().item())
        else:
            im_cnt = 0
        report.total_im_weights += im_cnt
        report.total_re_weights += n_el - im_cnt

    # 1. Embeddings, Final Norm, LM Head
    _load_param(d16_model.model.embed_tokens.weight, emb_w, mode=second_coord_mode, sign=1.0)

    norm_w = _require_tensor(state_dict, "model.norm.weight", "norm.weight")
    _load_param(d16_model.model.norm.gain, norm_w, mode="zero")

    if not config.tie_word_embeddings:
        lm_w = _require_tensor(state_dict, "lm_head.weight")
        _load_param(d16_model.lm_head.weight, lm_w, mode=second_coord_mode, sign=1.0)

    # 2. Decoder Layers
    for idx, layer in enumerate(d16_model.model.layers):
        p_m = f"model.layers.{idx}"
        p_s = f"layers.{idx}"

        in_ln = _require_tensor(state_dict, f"{p_m}.input_layernorm.weight", f"{p_s}.input_layernorm.weight")
        _load_param(layer.input_layernorm.gain, in_ln, mode="zero")

        post_ln = _require_tensor(state_dict, f"{p_m}.post_attention_layernorm.weight", f"{p_s}.post_attention_layernorm.weight")
        _load_param(layer.post_attention_layernorm.gain, post_ln, mode="zero")

        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            proj_mod = getattr(layer.self_attn, proj_name)
            pw = _require_tensor(state_dict, f"{p_m}.self_attn.{proj_name}.weight", f"{p_s}.self_attn.{proj_name}.weight")
            is_qk = proj_name in ("q_proj", "k_proj") and second_coord_mode == "d16_sat_rope_gate"
            pw_partner = _rope_head_half_partner(pw, config.head_dim) if is_qk else None
            _load_param(proj_mod.weight, pw, mode=second_coord_mode, partner_w=pw_partner, sign=1.0)
            if proj_mod.bias is not None:
                pb = _require_tensor(state_dict, f"{p_m}.self_attn.{proj_name}.bias", f"{p_s}.self_attn.{proj_name}.bias")
                pb_partner = _rope_head_half_partner(pb, config.head_dim) if is_qk else None
                _load_param(proj_mod.bias, pb, mode=(second_coord_mode if is_qk else "zero"), partner_w=pb_partner, sign=1.0)

        up_w = _require_tensor(state_dict, f"{p_m}.mlp.up_proj.weight", f"{p_s}.mlp.up_proj.weight")
        gate_w = _get_tensor(state_dict, f"{p_m}.mlp.gate_proj.weight", f"{p_s}.mlp.gate_proj.weight")
        down_w = _require_tensor(state_dict, f"{p_m}.mlp.down_proj.weight", f"{p_s}.mlp.down_proj.weight")

        layer_rep = find_mlp_quartets(
            up_w,
            gate_w,
            down_w,
            layer_idx=idx,
            min_orbit_score=min_orbit_score,
        )
        report.layer_reports[idx] = layer_rep
        im_mask = layer_rep.is_im_neuron if second_coord_mode == "zero" else None

        _load_param(layer.mlp.up_proj.weight, up_w, is_im_mask=im_mask, im_dim=0, mode=second_coord_mode, partner_w=gate_w, sign=1.0)
        if layer.mlp.up_proj.bias is not None:
            up_b = _require_tensor(state_dict, f"{p_m}.mlp.up_proj.bias", f"{p_s}.mlp.up_proj.bias")
            _load_param(layer.mlp.up_proj.bias, up_b, is_im_mask=im_mask, im_dim=0, mode="zero")

        if layer.mlp.gate_proj is not None:
            gate_req = _require_tensor(state_dict, f"{p_m}.mlp.gate_proj.weight", f"{p_s}.mlp.gate_proj.weight")
            _load_param(layer.mlp.gate_proj.weight, gate_req, is_im_mask=im_mask, im_dim=0, mode=second_coord_mode, partner_w=up_w, sign=1.0)
            if layer.mlp.gate_proj.bias is not None:
                gate_b = _require_tensor(state_dict, f"{p_m}.mlp.gate_proj.bias", f"{p_s}.mlp.gate_proj.bias")
                _load_param(layer.mlp.gate_proj.bias, gate_b, is_im_mask=im_mask, im_dim=0, mode="zero")

        _load_param(layer.mlp.down_proj.weight, down_w, is_im_mask=im_mask, im_dim=1, mode=second_coord_mode, sign=1.0)
        if layer.mlp.down_proj.bias is not None:
            down_b = _require_tensor(state_dict, f"{p_m}.mlp.down_proj.bias", f"{p_s}.mlp.down_proj.bias")
            _load_param(layer.mlp.down_proj.bias, down_b, mode="zero")

        if verbose:
            print(
                f"[Слой {idx:2d}/{config.num_hidden_layers - 1:2d}] "
                f"Четверок >= {min_orbit_score:.2f}: {layer_rep.active_quartets_count:4d}/{layer_rep.quadruplets.shape[0]:<4d} | "
                f"Re: {layer_rep.re_neurons_count:4d}, Im: {layer_rep.im_neurons_count:4d} | "
                f"Ср. Score: {layer_rep.mean_orbit_score:.4f}"
            )

    return d16_model, report


def print_import_summary(report: ConversionReport) -> None:
    """Print a concise summary table of the 1-to-1 Qwen -> D16 import."""
    print("=" * 116)
    print(
        f"ОТЧЕТ ИМПОРТА ВЕСОВ 1-В-1 В D16QwenForCausalLM (d16_qwen_model2) | Порог четверок min_orbit_score = {report.min_orbit_score:.2f}"
    )
    print("=" * 116)
    print(
        f"{'Слой':>4} | {'Всего нейронов':>14} | {'Четверок (>=порога)':>19} | "
        f"{'В четверках':>12} | {'Без пары (в x)':>14} | {'Роль Re (в x)':>13} | {'Роль Im (в y)':>13} | {'Ср. Score':>9}"
    )
    print("-" * 116)
    for idx in sorted(report.layer_reports.keys()):
        lr = report.layer_reports[idx]
        tot_q = lr.quadruplets.shape[0]
        act_q = lr.active_quartets_count
        print(
            f"{idx:4d} | {lr.orig_intermediate_size:14d} | {act_q:8d} / {tot_q:<8d} | "
            f"{lr.paired_neurons_count:12d} | {lr.unpaired_neurons_count:14d} | "
            f"{lr.re_neurons_count:13d} | {lr.im_neurons_count:13d} | {lr.mean_orbit_score:9.4f}"
        )
    print("=" * 116)
    im_pct = 100.0 * report.total_im_weights / max(report.total_weights, 1)
    re_pct = 100.0 * report.total_re_weights / max(report.total_weights, 1)
    print(
        f"ИТОГО ИМПОРТИРОВАНО ТЕНЗОРОВ: {report.total_imported_tensors} | "
        f"ВСЕГО ВЕСОВ D16: {report.total_weights} | "
        f"В Re (x, y=0): {report.total_re_weights} ({re_pct:.1f}%) | "
        f"В Im (x=0, y): {report.total_im_weights} ({im_pct:.1f}%)"
    )
    print("=" * 116)


# =====================================================================
# 6. FAST SAVE & LOAD OF PACKED D16 CHECKPOINTS (.safetensors / .pt)
# =====================================================================

def _serialize_config(model: D16QwenForCausalLM) -> Dict[str, Any]:
    cfg = model.config
    cfg_dict = asdict(cfg)
    cfg_dict["is_1to1_import"] = bool(cfg.complex_hidden_size == cfg.hidden_size)
    cfg_dict["dtype"] = str(model.model.embed_tokens.weight.x.dtype).replace("torch.", "")
    return cfg_dict


def _deserialize_config(cfg_dict: Mapping[str, Any]) -> Union[D16QwenImportConfig, D16QwenConfig]:
    is_1to1 = bool(cfg_dict.get("is_1to1_import", True))
    target_cls = D16QwenImportConfig if is_1to1 else D16QwenConfig
    valid_keys = {f.name for f in fields(target_cls)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    return target_cls(**filtered)


def _deduplicate_tied_state_dict(model: D16QwenForCausalLM) -> Dict[str, torch.Tensor]:
    """Return a contiguous CPU state_dict omitting duplicate tied `lm_head.weight.*` tensors."""
    raw_sd = model.state_dict()
    clean_sd: Dict[str, torch.Tensor] = {}
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    for k, v in raw_sd.items():
        if tied and k.startswith("lm_head.weight."):
            continue
        clean_sd[k] = v.detach().cpu().contiguous()
    return clean_sd


def _serialize_report(report: ConversionReport) -> Dict[str, Any]:
    return {
        "min_orbit_score": report.min_orbit_score,
        "total_imported_tensors": report.total_imported_tensors,
        "total_weights": report.total_weights,
        "total_re_weights": report.total_re_weights,
        "total_im_weights": report.total_im_weights,
        "layer_reports": {
            idx: {
                "layer_idx": lr.layer_idx,
                "mode": lr.mode,
                "orig_intermediate_size": lr.orig_intermediate_size,
                "d16_intermediate_size": lr.d16_intermediate_size,
                "quadruplets": lr.quadruplets.cpu(),
                "group_ops": lr.group_ops.cpu(),
                "orbit_scores": lr.orbit_scores.cpu(),
                "active_quartet_mask": lr.active_quartet_mask.cpu(),
                "is_im_neuron": lr.is_im_neuron.cpu(),
                "paired_neuron_mask": lr.paired_neuron_mask.cpu(),
                "op_histogram": lr.op_histogram.cpu(),
                "local_2x2": lr.local_2x2.cpu(),
            }
            for idx, lr in report.layer_reports.items()
        },
    }


def _deserialize_report(data: Mapping[str, Any]) -> ConversionReport:
    rep = ConversionReport(
        min_orbit_score=float(data["min_orbit_score"]),
        total_imported_tensors=int(data.get("total_imported_tensors", 0)),
        total_weights=int(data.get("total_weights", 0)),
        total_re_weights=int(data.get("total_re_weights", 0)),
        total_im_weights=int(data.get("total_im_weights", 0)),
    )
    for idx_k, lr_d in data.get("layer_reports", {}).items():
        rep.layer_reports[int(idx_k)] = LayerOrbitReport(
            layer_idx=int(lr_d["layer_idx"]),
            mode=str(lr_d["mode"]),
            orig_intermediate_size=int(lr_d["orig_intermediate_size"]),
            d16_intermediate_size=int(lr_d["d16_intermediate_size"]),
            quadruplets=lr_d["quadruplets"],
            group_ops=lr_d["group_ops"],
            orbit_scores=lr_d["orbit_scores"],
            active_quartet_mask=lr_d["active_quartet_mask"],
            is_im_neuron=lr_d["is_im_neuron"],
            paired_neuron_mask=lr_d["paired_neuron_mask"],
            op_histogram=lr_d["op_histogram"],
            local_2x2=lr_d["local_2x2"],
        )
    return rep


@torch.no_grad()
def save_d16_checkpoint(
    model: D16QwenForCausalLM,
    save_path: Union[str, Path],
    *,
    report: Optional[ConversionReport] = None,
    use_safetensors: bool = True,
) -> Path:
    """Save a packed/fine-tuned `D16QwenForCausalLM` model (and optional ConversionReport) to disk.

    If `save_path` has a `.pt` or `.pth` suffix, saves a single self-contained PyTorch archive.
    Otherwise, treats `save_path` as a directory and writes:
      - `d16_config.json`
      - `d16_model.safetensors` (or `d16_model.pt` if `safetensors` is unavailable)
      - `d16_orbit_report.pt` (if `report` is provided)
    """
    target = Path(save_path).expanduser()
    cfg_dict = _serialize_config(model)
    state_dict = _deduplicate_tied_state_dict(model)

    if target.suffix in (".pt", ".pth", ".bin"):
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "d16_config": cfg_dict,
            "state_dict": state_dict,
        }
        if report is not None:
            payload["d16_orbit_report"] = _serialize_report(report)
        torch.save(payload, str(target))
        return target

    target.mkdir(parents=True, exist_ok=True)
    cfg_file = target / "d16_config.json"
    cfg_file.write_text(json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    saved_weights = False
    if use_safetensors:
        try:
            from safetensors.torch import save_file
            save_file(state_dict, str(target / "d16_model.safetensors"))
            saved_weights = True
        except ImportError:
            saved_weights = False

    if not saved_weights:
        torch.save(state_dict, str(target / "d16_model.pt"))

    if report is not None:
        torch.save(_serialize_report(report), str(target / "d16_orbit_report.pt"))

    return target


@torch.no_grad()
def load_d16_checkpoint(
    load_path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
    return_report: bool = False,
) -> Union[D16QwenForCausalLM, Tuple[D16QwenForCausalLM, Optional[ConversionReport]]]:
    """Instantly load a packed `D16QwenForCausalLM` checkpoint without quartet search or random init."""
    target = Path(load_path).expanduser()
    target_device = torch.device(device)
    report: Optional[ConversionReport] = None

    if target.is_file() and target.suffix in (".pt", ".pth", ".bin"):
        raw = torch.load(str(target), map_location=target_device, weights_only=False)
        cfg_dict = raw["d16_config"]
        state_dict = raw["state_dict"]
        if "d16_orbit_report" in raw and raw["d16_orbit_report"] is not None:
            report = _deserialize_report(raw["d16_orbit_report"])
    elif target.is_dir():
        cfg_file = target / "d16_config.json"
        if not cfg_file.is_file():
            raise FileNotFoundError(f"Missing 'd16_config.json' in checkpoint directory: {target}")
        cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8"))

        st_file = target / "d16_model.safetensors"
        pt_file = target / "d16_model.pt"
        if st_file.is_file():
            from safetensors.torch import load_file
            st_dev = f"cuda:{torch.cuda.current_device()}" if target_device.type == "cuda" and target_device.index is None else str(target_device)
            state_dict = load_file(str(st_file), device=st_dev)
        elif pt_file.is_file():
            state_dict = torch.load(str(pt_file), map_location=target_device, weights_only=True)
        else:
            raise FileNotFoundError(f"No 'd16_model.safetensors' or 'd16_model.pt' found in: {target}")

        rep_file = target / "d16_orbit_report.pt"
        if rep_file.is_file():
            rep_data = torch.load(str(rep_file), map_location="cpu", weights_only=False)
            report = _deserialize_report(rep_data)
    else:
        raise FileNotFoundError(f"D16 checkpoint path not found: {target}")

    config = _deserialize_config(cfg_dict)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    saved_dtype = dtype_map.get(str(cfg_dict.get("dtype", "float32")), torch.float32)
    target_dtype = saved_dtype if dtype is None else dtype

    with _fast_empty_d16_init(dtype=target_dtype, device=target_device):
        d16_model = D16QwenForCausalLM(config)
    d16_model = d16_model.to(device=target_device)
    if config.complex_hidden_size == config.hidden_size and (config.head_dim % 2) == 0:
        even_idx = torch.arange(0, config.head_dim, 2, device=target_device)
        d16_model.model.rotary_emb.frequency_indices.copy_(torch.cat((even_idx, even_idx)))

    # Cast floating-point coordinates (x, y) if an explicit dtype override was requested
    if dtype is not None:
        for k, v in list(state_dict.items()):
            if v.is_floating_point() and not k.endswith("inv_freq"):
                state_dict[k] = v.to(dtype=target_dtype)

    missing, unexpected = d16_model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"lm_head.weight.x", "lm_head.weight.y", "lm_head.weight.rotation"} if config.tie_word_embeddings else set()
    real_missing = [k for k in missing if k not in allowed_missing]
    if real_missing or unexpected:
        raise RuntimeError(f"D16 checkpoint mismatch! Missing: {real_missing}, Unexpected: {unexpected}")

    if config.tie_word_embeddings:
        d16_model.lm_head.weight = d16_model.model.embed_tokens.weight
    d16_model = d16_model.to(device=target_device)

    if return_report:
        return d16_model, report
    return d16_model


__all__ = [
    "WEDGE_SAT_SLOPE",
    "WEDGE_HALF_SAT",
    "D16QwenImportConfig",
    "LayerOrbitReport",
    "ConversionReport",
    "load_qwen_checkpoint",
    "find_mlp_quartets",
    "compute_d16_second_coord",
    "pack_weights_1to1",
    "import_qwen_weights",
    "print_import_summary",
    "save_d16_checkpoint",
    "load_d16_checkpoint",
]
