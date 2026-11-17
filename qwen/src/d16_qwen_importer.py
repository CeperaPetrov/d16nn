"""Standalone 1-to-1 Weight Importer from Reference Qwen Checkpoints into d16_qwen_model2."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from d16_complex import D16ComplexTensor, INV_SQRT2
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

    mlp_activation: MLPActivation = "bilinear"

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
        mlp_activation: MLPActivation = "bilinear",
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
        mlp_activation: MLPActivation = "bilinear",
        collect_d16_stats: bool = False,
    ) -> "D16QwenImportConfig":
        """Infer 1-to-1 D16QwenImportConfig dimensions directly from a Qwen state_dict."""
        embed_w = state_dict.get("model.embed_tokens.weight", state_dict.get("embed_tokens.weight"))
        if embed_w is None:
            raise ValueError("state_dict must contain 'model.embed_tokens.weight'")
        vocab_size, hidden_size = int(embed_w.shape[0]), int(embed_w.shape[1])

        up_w = state_dict.get("model.layers.0.mlp.up_proj.weight", state_dict.get("layers.0.mlp.up_proj.weight"))
        if up_w is None:
            raise ValueError("state_dict must contain 'model.layers.0.mlp.up_proj.weight'")
        intermediate_size = int(up_w.shape[0])

        layer_indices = set()
        for key in state_dict.keys():
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_indices.add(int(parts[i + 1]))
        detected_layers = max(layer_indices) + 1 if layer_indices else 1
        layers = detected_layers if num_hidden_layers is None else min(int(num_hidden_layers), detected_layers)

        k_w = state_dict.get("model.layers.0.self_attn.k_proj.weight", state_dict.get("layers.0.self_attn.k_proj.weight"))
        q_b = state_dict.get("model.layers.0.self_attn.q_proj.bias", state_dict.get("layers.0.self_attn.q_proj.bias"))
        up_b = state_dict.get("model.layers.0.mlp.up_proj.bias", state_dict.get("layers.0.mlp.up_proj.bias"))

        kv_dim = int(k_w.shape[0]) if k_w is not None else hidden_size
        if num_attention_heads is None:
            if hidden_size == 896:
                num_attention_heads = 14
            elif hidden_size % 64 == 0:
                num_attention_heads = hidden_size // 64
            else:
                num_attention_heads = max(1, hidden_size // max(kv_dim, 2))
        head_dim = hidden_size // num_attention_heads
        num_key_value_heads = max(1, kv_dim // head_dim)

        lm_w = state_dict.get("lm_head.weight")
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
            attention_bias=q_b is not None,
            mlp_bias=up_b is not None,
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
            return load_file(str(st_single), device=str(device)), cfg_dict
        if st_shards:
            from safetensors.torch import load_file
            merged: Dict[str, torch.Tensor] = {}
            for shard in st_shards:
                merged.update(load_file(str(shard), device=str(device)))
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
                return load_file(str(f0), device=str(device)), cfg_dict
            return torch.load(str(f0), map_location=device, weights_only=True), cfg_dict
        raise FileNotFoundError(f"No .safetensors or .bin checkpoint files found in directory: {path}")

    # 2. Direct path to a .safetensors, .bin, or .pt file
    if path.is_file():
        cfg_file = path.parent / "config.json"
        if cfg_file.is_file():
            cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8"))
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(path), device=str(device)), cfg_dict
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
# 5. STANDARD 1-TO-1 WEIGHT PACKING & MODEL IMPORT
# =====================================================================

@torch.no_grad()
def pack_weights_1to1(
    weight: torch.Tensor,
    is_im_mask: Optional[torch.Tensor] = None,
    *,
    im_dim: int = 0,
) -> D16ComplexTensor:
    """Convert a real weight tensor 1-to-1 into a `D16ComplexTensor` via `D16ComplexTensor.pack`."""
    w = weight.detach()
    zeros = torch.zeros_like(w)
    if is_im_mask is None or not bool(is_im_mask.any()):
        return D16ComplexTensor.pack(w, zeros)

    mask = is_im_mask.to(device=w.device, dtype=torch.bool)
    if w.ndim == 2 and mask.ndim == 1:
        mask = mask.unsqueeze(1) if im_dim == 0 else mask.unsqueeze(0)
    re = torch.where(mask, zeros, w)
    im = torch.where(mask, w, zeros)
    return D16ComplexTensor.pack(re, im)


@contextmanager
def _fast_empty_d16_init(dtype: torch.dtype, device: Union[str, torch.device]):
    """Skip expensive random polar sampling when constructing a D16 model for weight import."""
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


@torch.no_grad()
def import_qwen_weights(
    source: Union[str, Path, nn.Module, Mapping[str, torch.Tensor]],
    config: Optional[D16QwenImportConfig] = None,
    *,
    num_hidden_layers: Optional[int] = None,
    min_orbit_score: float = 0.25,
    indiff_eps: IndiffThreshold = 0.0,
    indiff_rel: float = 0.25,
    mlp_activation: MLPActivation = "bilinear",
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
) -> Tuple[D16QwenForCausalLM, ConversionReport]:
    """Import weights 1-to-1 from a Qwen file/folder/model/state_dict into `d16_qwen_model2.D16QwenForCausalLM`."""
    load_dev = "cpu" if device is None else device
    state_dict, cfg_dict = load_qwen_checkpoint(source, device=load_dev)

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

    sample_t = next(iter(state_dict.values()))
    target_dtype = sample_t.dtype if dtype is None else dtype
    target_device = sample_t.device if device is None else torch.device(device)

    with _fast_empty_d16_init(dtype=target_dtype, device=target_device):
        d16_model = D16QwenForCausalLM(config)

    report = ConversionReport(min_orbit_score=min_orbit_score)

    def _load_param(
        param: D16Parameter,
        w: torch.Tensor,
        is_im_mask: Optional[torch.Tensor] = None,
        im_dim: int = 0,
    ) -> None:
        target_w = w.to(device=target_device, dtype=target_dtype)
        d16_val = pack_weights_1to1(target_w, is_im_mask=is_im_mask, im_dim=im_dim)
        param.copy_(d16_val)

        n_el = target_w.numel()
        report.total_imported_tensors += 1
        report.total_weights += n_el
        if is_im_mask is not None and bool(is_im_mask.any()):
            if target_w.ndim == 2 and is_im_mask.ndim == 1:
                im_cnt = int(is_im_mask.sum().item()) * (
                    target_w.shape[1] if im_dim == 0 else target_w.shape[0]
                )
            else:
                im_cnt = int(is_im_mask.sum().item())
        else:
            im_cnt = 0
        report.total_im_weights += im_cnt
        report.total_re_weights += n_el - im_cnt

    # 1. Embeddings, Final Norm, LM Head
    emb_w = _get_tensor(state_dict, "model.embed_tokens.weight", "embed_tokens.weight")
    if emb_w is not None:
        _load_param(d16_model.model.embed_tokens.weight, emb_w)

    norm_w = _get_tensor(state_dict, "model.norm.weight", "norm.weight")
    if norm_w is not None:
        _load_param(d16_model.model.norm.gain, norm_w)

    if not config.tie_word_embeddings:
        lm_w = _get_tensor(state_dict, "lm_head.weight")
        if lm_w is not None:
            _load_param(d16_model.lm_head.weight, lm_w)

    # 2. Decoder Layers
    for idx, layer in enumerate(d16_model.model.layers):
        p_m = f"model.layers.{idx}"
        p_s = f"layers.{idx}"

        in_ln = _get_tensor(state_dict, f"{p_m}.input_layernorm.weight", f"{p_s}.input_layernorm.weight")
        if in_ln is not None:
            _load_param(layer.input_layernorm.gain, in_ln)

        post_ln = _get_tensor(state_dict, f"{p_m}.post_attention_layernorm.weight", f"{p_s}.post_attention_layernorm.weight")
        if post_ln is not None:
            _load_param(layer.post_attention_layernorm.gain, post_ln)

        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            proj_mod = getattr(layer.self_attn, proj_name)
            pw = _get_tensor(state_dict, f"{p_m}.self_attn.{proj_name}.weight", f"{p_s}.self_attn.{proj_name}.weight")
            if pw is not None:
                _load_param(proj_mod.weight, pw)
            if proj_mod.bias is not None:
                pb = _get_tensor(state_dict, f"{p_m}.self_attn.{proj_name}.bias", f"{p_s}.self_attn.{proj_name}.bias")
                if pb is not None:
                    _load_param(proj_mod.bias, pb)

        up_w = _get_tensor(state_dict, f"{p_m}.mlp.up_proj.weight", f"{p_s}.mlp.up_proj.weight")
        gate_w = _get_tensor(state_dict, f"{p_m}.mlp.gate_proj.weight", f"{p_s}.mlp.gate_proj.weight")
        down_w = _get_tensor(state_dict, f"{p_m}.mlp.down_proj.weight", f"{p_s}.mlp.down_proj.weight")

        if up_w is not None:
            layer_rep = find_mlp_quartets(
                up_w,
                gate_w,
                down_w,
                layer_idx=idx,
                min_orbit_score=min_orbit_score,
            )
            report.layer_reports[idx] = layer_rep
            im_mask = layer_rep.is_im_neuron

            _load_param(layer.mlp.up_proj.weight, up_w, is_im_mask=im_mask, im_dim=0)
            if layer.mlp.up_proj.bias is not None:
                up_b = _get_tensor(state_dict, f"{p_m}.mlp.up_proj.bias", f"{p_s}.mlp.up_proj.bias")
                if up_b is not None:
                    _load_param(layer.mlp.up_proj.bias, up_b, is_im_mask=im_mask, im_dim=0)

            if layer.mlp.gate_proj is not None and gate_w is not None:
                _load_param(layer.mlp.gate_proj.weight, gate_w, is_im_mask=im_mask, im_dim=0)
                if layer.mlp.gate_proj.bias is not None:
                    gate_b = _get_tensor(state_dict, f"{p_m}.mlp.gate_proj.bias", f"{p_s}.mlp.gate_proj.bias")
                    if gate_b is not None:
                        _load_param(layer.mlp.gate_proj.bias, gate_b, is_im_mask=im_mask, im_dim=0)

            if down_w is not None:
                _load_param(layer.mlp.down_proj.weight, down_w, is_im_mask=im_mask, im_dim=1)
                if layer.mlp.down_proj.bias is not None:
                    down_b = _get_tensor(state_dict, f"{p_m}.mlp.down_proj.bias", f"{p_s}.mlp.down_proj.bias")
                    if down_b is not None:
                        _load_param(layer.mlp.down_proj.bias, down_b)

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


__all__ = [
    "D16QwenImportConfig",
    "LayerOrbitReport",
    "ConversionReport",
    "load_qwen_checkpoint",
    "find_mlp_quartets",
    "pack_weights_1to1",
    "import_qwen_weights",
    "print_import_summary",
]