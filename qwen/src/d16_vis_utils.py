"""Visualization and Inspection Utilities for D16Complex Orbit Conversion & D16Parameter Weights.

Provides console tables and matplotlib polar/bar/heatmap plots to inspect:
1. Independent D16Complex objects (neurons) that lie on the same Z_8 |>< Z_2 orbit.
2. Polar 16-sector compass showing where each of the 4 orbit-related neurons points
   inside the anchor's complex plane and fundamental wedge [0°, 22.5°].
3. Histogram of all 16 dihedral operations of Z_8 |>< Z_2 (Axial Im vs 45° Hadamard Re).
4. Polar 16-sector response profiles of _eval_d16(z) for axial (x*y) vs diagonal 0.5*(x^2-y^2) branches.
5. Native D16Parameter weight sector distributions (polar 16-sector rose + [0°, 22.5°] wedge scatter),
   console migration tables, and 16x16 sector transition matrices during D16AdamW training.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from d16_complex import D16Complex, D16ComplexTensor, STATE_LABELS, TAN_PI_8
from d16_parameter import D16Parameter
from d16_qwen_converter import ConversionReport, LayerOrbitReport
from d16_qwen_model import D16QwenMLP


# =====================================================================
# 1. ORBIT CONVERSION INSPECTION & PLOTS
# =====================================================================

def print_orbit_table(layer_report: LayerOrbitReport, top_k: int = 10) -> None:
    """Print a detailed table of top-k Z_8 |>< Z_2 orbit groups of D16Complex objects."""
    print("=" * 116)
    print(
        f"СЛОЙ {layer_report.layer_idx:2d} | Режим: {layer_report.mode} | "
        f"Объектов D16Complex: {layer_report.orig_intermediate_size} -> {layer_report.d16_intermediate_size} (все сохранены!) | "
        f"Средний score орбит: {layer_report.mean_orbit_score:.4f} | Доля 45° Re(z^2): {layer_report.hadamard_45_pct:.1f}%"
    )
    print("=" * 116)
    print(
        f"{'Орбита':>6} | {'Score':>6} | {'Якорь D16 (m=0)':<15} | "
        f"{'Объект m=1 (op)':<26} | {'Объект m=2 (op)':<26} | {'Объект m=3 (op)':<26}"
    )
    print("-" * 116)

    for item in layer_report.top_orbits(k=top_k):
        q = item["unit_idx"]
        sc = item["score"]
        n = item["neurons"]
        ops = item["ops"]

        def _fmt_member(m_idx: int) -> str:
            nid = n[m_idx]
            op_id = ops[m_idx]
            branch = "Re 45°" if op_id >= 8 else "Im ax "
            return f"n={nid:<4d} [op={op_id:2d}, {branch}]"

        print(
            f"{q:6d} | {sc:6.4f} | n={n[0]:<4d} [op= 0]  | "
            f"{_fmt_member(1):<26} | {_fmt_member(2):<26} | {_fmt_member(3):<26}"
        )
    print("=" * 116)


def plot_single_orbit_compass(
    layer_report: LayerOrbitReport,
    unit_idx: int = 0,
    ax=None,
):
    """Plot a polar 16-sector compass showing 4 D16Complex objects lying on orbit unit_idx."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure

    theta_wedge = np.linspace(0.0, math.pi / 8.0, 50)
    ax.fill_between(theta_wedge, 0.0, 1.15, color="#ffe082", alpha=0.45, label="Клин [0°, 22.5°]")

    for s in range(16):
        ang = s * (math.pi / 8.0)
        ax.plot([ang, ang], [0.0, 1.15], color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        mid_ang = (s + 0.5) * (math.pi / 8.0)
        ax.text(mid_ang, 1.08, f"S{s}", ha="center", va="center", fontsize=8, color="#333333")

    quad = layer_report.quadruplets[unit_idx].tolist()
    ops = layer_report.group_ops[unit_idx].tolist()
    loc = layer_report.local_2x2[unit_idx]

    colors = ["#d32f2f", "#1976d2", "#388e3c", "#7b1fa2"]
    max_r = float(loc.norm(dim=-1).max().clamp_min(1e-6).item())

    for m in range(4):
        gx, gy = float(loc[m, 1, 0].item()) / max_r, float(loc[m, 1, 1].item()) / max_r
        z_m = D16Complex.from_xy(gx, gy)
        angle = math.atan2(gy, gx) % (2.0 * math.pi)
        radius = math.hypot(gx, gy)
        op_id = ops[m]
        branch_tag = "Re 45°" if op_id >= 8 else "Im 0°/90°"
        lbl = f"m={m}: D16 #{quad[m]} (op={op_id}, {branch_tag}, S{z_m.state()})"
        ax.annotate(
            "",
            xy=(angle, radius),
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color=colors[m], lw=2.2),
        )
        ax.plot([angle], [radius], marker="o", markersize=7, color=colors[m], label=lbl)

    ax.set_ylim(0.0, 1.18)
    ax.set_yticklabels([])
    score_val = float(layer_report.orbit_scores[unit_idx].item())
    ax.set_title(
        f"Слой {layer_report.layer_idx} | Орбита #{unit_idx} (Score = {score_val:.4f})\n"
        f"Точки на орбите (независимые объекты D16Complex): {quad}",
        pad=18,
        fontsize=10,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), fontsize=8)
    return fig, ax


def plot_layer_z8z2_spectrum(report: ConversionReport, layer_idx: int = 0):
    """Plot histogram of the 16 Z_8 |>< Z_2 operations and per-layer 45° Hadamard ratio."""
    import matplotlib.pyplot as plt
    import numpy as np

    layer_rep = report.layer_reports[layer_idx]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    counts = layer_rep.op_histogram.numpy()
    ops_x = np.arange(16)
    bar_colors = ["#1976d2" if op < 8 else "#e65100" for op in ops_x]
    ax1.bar(ops_x, counts, color=bar_colors, alpha=0.85, edgecolor="black")
    ax1.set_xticks(ops_x)
    ax1.set_xticklabels([f"{op}" for op in ops_x])
    ax1.set_xlabel("Индекс операции вращения масок rotation в Z_8 |>< Z_2")
    ax1.set_ylabel("Число пар на орбитах")
    ax1.set_title(
        f"Слой {layer_idx}: Спектр 16 операций Z_8 |>< Z_2\n"
        f"(Доля 45°-поворотов Адамара Re(z^2): {layer_rep.hadamard_45_pct:.1f}%)"
    )
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    layers = sorted(report.layer_reports.keys())
    had_pcts = [report.layer_reports[l].hadamard_45_pct for l in layers]
    scores = [report.layer_reports[l].mean_orbit_score * 100.0 for l in layers]

    ax2.plot(layers, had_pcts, marker="o", color="#e65100", lw=2, label="Доля 45° Re(z^2) связей (%)")
    ax2.plot(layers, scores, marker="s", color="#2e7d32", lw=2, label="Средний косинус орбиты (×100%)")
    ax2.set_xlabel("Номер слоя (Layer)")
    ax2.set_ylabel("Процент (%)")
    ax2.set_title("Послойная структура орбит Z_8 |>< Z_2 по всей сети")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_learned_sector_gains(d16_mlp: D16QwenMLP, unit_indices: Sequence[int] = (0, 1)):
    """Plot polar 16-sector response profiles of _eval_d16(z) for axial vs 45°-rotated masks."""
    import matplotlib.pyplot as plt
    import numpy as np

    angles = np.linspace(0.0, 2.0 * math.pi, 65)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), subplot_kw={"projection": "polar"})

    with torch.no_grad():
        u_norms = d16_mlp.up_proj.weight.detach().norm(dim=1).cpu()
        g_norms = d16_mlp.gate_proj.weight.detach().norm(dim=1).cpu()

        th_t = torch.tensor(angles, dtype=torch.float32, device=d16_mlp.up_proj.weight.device)
        for u_idx in unit_indices:
            r_scale = float((u_norms[u_idx] + g_norms[u_idx]).item() * 0.5)
            L = r_scale * th_t.cos()
            E = r_scale * th_t.sin()
            z = D16ComplexTensor.from_xy(L, E)
            act_base = d16_mlp._eval_d16(z).abs().cpu().numpy()
            # Поворот маски rotation на +45° через инкапсулированный метод apply_group_op(8)
            z_rot45 = z.apply_group_op(8)
            act_rot45 = d16_mlp._eval_d16(z_rot45).abs().cpu().numpy()

            axes[0].plot(angles, act_base, lw=1.8, label=f"D16 #{u_idx} (rot)")
            axes[1].plot(angles, act_rot45, lw=1.8, label=f"D16 #{u_idx} (apply_group_op(8))")

    axes[0].set_title("Отклик _eval_d16(z) по 16 секторам (исходный rotation)", pad=15)
    axes[1].set_title("Отклик _eval_d16(z) при повороте маски на +45° (op=8)", pad=15)
    for ax in axes:
        ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)

    plt.tight_layout()
    return fig


# =====================================================================
# 2. NATIVE D16Parameter WEIGHT SECTOR & MIGRATION VISUALIZATION
# =====================================================================

def _iter_complex_d16_parameters(
    model: nn.Module,
    module_name: Optional[str] = None,
    include_biases: bool = False,
) -> list[Tuple[str, D16Parameter]]:
    """Collect named complex D16Parameter modules from a model."""
    seen: set[int] = set()
    items: list[Tuple[str, D16Parameter]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, D16Parameter):
            continue
        if id(mod) in seen:
            continue
        if mod.real_only:
            continue
        if not include_biases and name.endswith("bias"):
            continue
        if module_name is not None and module_name not in name:
            continue
        seen.add(id(mod))
        items.append((name, mod))
    return items


@torch.no_grad()
def snapshot_d16_weight_states(
    model: nn.Module,
    include_biases: bool = False,
) -> Dict[str, D16ComplexTensor]:
    """Capture a detached CPU snapshot of all complex D16Parameter values before training."""
    snapshot: Dict[str, D16ComplexTensor] = {}
    for name, mod in _iter_complex_d16_parameters(model, include_biases=include_biases):
        snapshot[name] = mod.value().detach().to("cpu").clone()
    return snapshot


@torch.no_grad()
def print_d16_weight_migration_table(
    model: nn.Module,
    initial_snapshot: Optional[Dict[str, D16ComplexTensor]] = None,
    include_biases: bool = False,
) -> Dict[str, float]:
    """Print per-module D16Parameter statistics and sector migration summary."""
    params = _iter_complex_d16_parameters(model, include_biases=include_biases)
    print("=" * 116)
    print("КАРТА КОМПЛЕКСНЫХ ВЕСОВ D16Parameter (ТРОЙКИ x, y, rotation В КЛИНЕ [0°, 22.5°])")
    print("=" * 116)
    header = (
        f"{'Модуль D16Parameter':<38} | {'Форма':<12} | {'Средн. |z|':>10} | "
        f"{'Клин y/x':>8} | {'0xFFFF':>7} | {'Сменили сектор':<20}"
    )
    print(header)
    print("-" * 116)

    total_elems = 0
    total_changed = 0
    total_indiff = 0

    for name, mod in params:
        val = mod.value().detach().to("cpu")
        n_el = val.numel()
        radii = val.norm_squared().sqrt()
        mean_r = float(radii.mean().item())
        safe_x = val.x.clamp_min(1e-12)
        mean_ratio = float((val.y / safe_x).mean().item())
        indiff_cnt = int(val.is_zero().sum().item())
        indiff_pct = 100.0 * indiff_cnt / max(n_el, 1)

        total_elems += n_el
        total_indiff += indiff_cnt

        if initial_snapshot is not None and name in initial_snapshot:
            prev = initial_snapshot[name]
            changed = int((val.state() != prev.state()).sum().item())
            total_changed += changed
            ch_str = f"{changed:5d}/{n_el:<5d} ({100.0 * changed / max(n_el, 1):5.1f}%)"
        else:
            ch_str = "n/a (нет снимка)"

        print(
            f"{name:<38} | {str(tuple(val.shape)):<12} | {mean_r:10.4f} | "
            f"{mean_ratio:8.4f} | {indiff_pct:6.2f}% | {ch_str:<20}"
        )

    print("=" * 116)
    overall_mig = (100.0 * total_changed / max(total_elems, 1)) if initial_snapshot is not None else 0.0
    overall_indiff = 100.0 * total_indiff / max(total_elems, 1)
    if initial_snapshot is not None:
        print(
            f"ИТОГО ПО СЕТИ: {total_changed}/{total_elems} ({overall_mig:.2f}%) весов сменили сектор rotation | "
            f"Доля 0xFFFF: {overall_indiff:.2f}%"
        )
    else:
        print(f"ИТОГО ПО СЕТИ: {total_elems} комплексных весов D16Parameter | Доля 0xFFFF: {overall_indiff:.2f}%")
    print("=" * 116)

    return {
        "total_weights": float(total_elems),
        "changed_weights": float(total_changed),
        "migration_pct": overall_mig,
        "indifferent_pct": overall_indiff,
    }


@torch.no_grad()
def plot_d16_weight_sector_distribution(
    model: nn.Module,
    initial_snapshot: Optional[Dict[str, D16ComplexTensor]] = None,
    module_name: Optional[str] = None,
    max_scatter_points: int = 1500,
):
    """Plot a 16-sector polar rose of D16Parameter weights and their (x, y) positions in the [0°, 22.5°] wedge."""
    import matplotlib.pyplot as plt
    import numpy as np

    params = _iter_complex_d16_parameters(model, module_name=module_name)
    if not params:
        raise ValueError(f"No complex D16Parameter modules found matching {module_name!r}")

    after_counts = torch.zeros(18, dtype=torch.long)
    before_counts = torch.zeros(18, dtype=torch.long)
    xs_list, ys_list, st_list = [], [], []

    for name, mod in params:
        val = mod.value().detach().to("cpu")
        st = val.state().flatten()
        after_counts += torch.bincount(st, minlength=18)
        xs_list.append(val.x.flatten())
        ys_list.append(val.y.flatten())
        st_list.append(st)

        if initial_snapshot is not None and name in initial_snapshot:
            st_prev = initial_snapshot[name].state().flatten()
            before_counts += torch.bincount(st_prev, minlength=18)

    all_x = torch.cat(xs_list).numpy()
    all_y = torch.cat(ys_list).numpy()
    all_s = torch.cat(st_list).numpy()

    fig = plt.figure(figsize=(14, 5.8))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_wedge = fig.add_subplot(1, 2, 2)

    # 1. Left panel: Polar 16-sector rose diagram (S0..S15)
    sector_angles = np.array([(s + 0.5) * (math.pi / 8.0) for s in range(16)])
    width = (math.pi / 8.0) * 0.86
    tot_active = max(int(after_counts[:16].sum().item()), 1)
    after_pct = (after_counts[:16].numpy() / tot_active) * 100.0

    if initial_snapshot is not None and int(before_counts[:16].sum().item()) > 0:
        tot_before = max(int(before_counts[:16].sum().item()), 1)
        before_pct = (before_counts[:16].numpy() / tot_before) * 100.0
        ax_polar.bar(
            sector_angles,
            before_pct,
            width=width,
            color="#90caf9",
            alpha=0.55,
            edgecolor="#1565c0",
            linestyle="--",
            label="До обучения (%)",
        )

    bar_colors = ["#1976d2" if (s & 3) in (0, 3) else "#e65100" for s in range(16)]
    ax_polar.bar(
        sector_angles,
        after_pct,
        width=width * 0.72,
        color=bar_colors,
        alpha=0.85,
        edgecolor="black",
        label="Текущие веса (%)",
    )

    ring_theta = np.linspace(0.0, 2.0 * math.pi, 120)
    ax_polar.plot(
        ring_theta,
        np.full_like(ring_theta, 6.25),
        color="#2e7d32",
        linestyle=":",
        lw=1.8,
        label="Изотропный уровень (6.25%)",
    )
    ax_polar.set_xticks(sector_angles)
    ax_polar.set_xticklabels([f"S{s}" for s in range(16)], fontsize=8)
    title_scope = module_name if module_name is not None else "Вся модель"
    ax_polar.set_title(
        f"Распределение весов D16Parameter по 16 секторам ({title_scope})\n"
        f"[Синие = осевые секторы, Оранжевые = диагональные 45°]",
        pad=16,
        fontsize=10,
    )
    ax_polar.legend(loc="upper right", bbox_to_anchor=(1.32, 1.14), fontsize=8)

    # 2. Right panel: Coordinates (x, y) inside the closed [0°, 22.5°] fundamental wedge
    if len(all_x) > max_scatter_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(all_x), size=max_scatter_points, replace=False)
        plot_x, plot_y, plot_s = all_x[idx], all_y[idx], all_s[idx]
    else:
        plot_x, plot_y, plot_s = all_x, all_y, all_s

    max_x = float(np.percentile(all_x, 99.5)) if len(all_x) > 0 else 1.0
    max_x = max(max_x * 1.08, 1e-3)
    x_line = np.linspace(0.0, max_x, 100)
    y_bis = TAN_PI_8 * x_line

    ax_wedge.fill_between(
        x_line, 0.0, y_bis, color="#fff8e1", alpha=0.7, label="Фундаментальный клин [0°, 22.5°]"
    )
    ax_wedge.plot(x_line, np.zeros_like(x_line), color="#1565c0", lw=2.0, label="Луч 0° (y = 0)")
    ax_wedge.plot(
        x_line,
        y_bis,
        color="#c62828",
        lw=2.0,
        linestyle="--",
        label="Биссектриса 22.5° (y = (√2 - 1)x)",
    )

    sc = ax_wedge.scatter(
        plot_x,
        plot_y,
        c=np.clip(plot_s, 0, 15),
        cmap="twilight",
        vmin=0,
        vmax=15,
        s=14,
        alpha=0.75,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax_wedge, pad=0.02)
    cbar.set_label("Сектор rotation (S0..S15)")

    ax_wedge.set_xlim(0.0, max_x)
    ax_wedge.set_ylim(-0.02 * max_x * TAN_PI_8, max_x * TAN_PI_8 * 1.12)
    ax_wedge.set_xlabel("Катет клина x >= 0")
    ax_wedge.set_ylabel("Катет клина 0 <= y <= (√2 - 1)x")
    ax_wedge.set_title(
        f"Координаты весов (x, y) внутри клина [0°, 22.5°] ({title_scope})",
        fontsize=10,
    )
    ax_wedge.grid(True, linestyle=":", alpha=0.6)
    ax_wedge.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    return fig


@torch.no_grad()
def plot_d16_sector_transition_matrix(
    model: nn.Module,
    initial_snapshot: Dict[str, D16ComplexTensor],
    module_name: Optional[str] = None,
):
    """Plot the 16x16 sector transition matrix (S_init -> S_final) and per-module migration bar chart."""
    import matplotlib.pyplot as plt
    import numpy as np

    params = _iter_complex_d16_parameters(model, module_name=module_name)
    if not params:
        raise ValueError(f"No complex D16Parameter modules found matching {module_name!r}")

    trans_mat = torch.zeros((16, 16), dtype=torch.long)
    mod_labels: list[str] = []
    mod_pcts: list[float] = []

    for name, mod in params:
        if name not in initial_snapshot:
            continue
        s_init = initial_snapshot[name].state().flatten()
        s_curr = mod.value().detach().to("cpu").state().flatten()
        valid = (s_init < 16) & (s_curr < 16)
        pairs = s_init[valid] * 16 + s_curr[valid]
        trans_mat += torch.bincount(pairs, minlength=256).reshape(16, 16)

        changed_pct = float((s_init != s_curr).float().mean().item() * 100.0)
        short_name = name.replace("model.layers.", "L").replace(".weight", "").replace("self_attn.", "attn.")
        mod_labels.append(short_name)
        mod_pcts.append(changed_pct)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    mat_np = trans_mat.numpy().astype(float)
    row_sums = mat_np.sum(axis=1, keepdims=True)
    mat_pct = np.divide(mat_np, np.maximum(row_sums, 1.0)) * 100.0

    im = ax1.imshow(mat_pct, cmap="YlOrRd", vmin=0.0, vmax=max(float(mat_pct.max()), 10.0))
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Доля переходов из сектора (%)")
    ticks = np.arange(16)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_xticklabels([f"S{s}" for s in ticks], fontsize=8)
    ax1.set_yticklabels([f"S{s}" for s in ticks], fontsize=8)
    ax1.set_xlabel("Конечный сектор после D16AdamW (S_final)")
    ax1.set_ylabel("Начальный сектор до обучения (S_init)")
    ax1.set_title("Матрица миграции секторов rotation (16×16)")

    y_pos = np.arange(len(mod_labels))
    ax2.barh(y_pos, mod_pcts, color="#1976d2", alpha=0.85, edgecolor="black")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(mod_labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0.0, 105.0)
    ax2.set_xlabel("Доля весов, сменивших сектор (%)")
    ax2.set_title("Подвижность секторов D16Parameter по слоям сети")
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    return fig


# =====================================================================
# 3. STEP-BY-STEP TRAJECTORY TRACKING (ATTENTION vs MLP vs EMBEDDING)
# =====================================================================

class D16TrajectoryRecorder:
    """Record step-by-step D16Parameter sector trajectories for Attention, MLP, and Embedding layers."""

    def __init__(self, model: nn.Module, tracked_synapses_per_group: int = 4) -> None:
        self.initial_snapshot = snapshot_d16_weight_states(model)
        self.prev_snapshot = {k: v.clone() for k, v in self.initial_snapshot.items()}
        self.tracked_synapses_per_group = tracked_synapses_per_group
        self.steps: list[int] = []
        self.losses: list[float] = []

        # Group metrics history: keys are "attn", "mlp", "embed"
        self.cumul_migration_pct: Dict[str, list[float]] = {"attn": [], "mlp": [], "embed": []}
        self.step_flip_pct: Dict[str, list[float]] = {"attn": [], "mlp": [], "embed": []}
        self.diag45_pct: Dict[str, list[float]] = {"attn": [], "mlp": [], "embed": []}
        self.mean_radius: Dict[str, list[float]] = {"attn": [], "mlp": [], "embed": []}

        # Full snapshots of (re, im, state) for selected individual synapses in Attention and MLP
        self._synapse_coords: Dict[str, list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
            "attn": [],
            "mlp": [],
        }
        self._synapse_module_names: Dict[str, str] = {}
        self._synapse_flat_indices: Dict[str, torch.Tensor] = {}
        self._init_tracked_synapses(model)

    @staticmethod
    def _classify_module(name: str) -> str:
        if "self_attn" in name or "attn" in name:
            return "attn"
        if "mlp" in name:
            return "mlp"
        return "embed"

    @torch.no_grad()
    def _init_tracked_synapses(self, model: nn.Module) -> None:
        for grp in ("attn", "mlp"):
            for name, mod in _iter_complex_d16_parameters(model):
                if self._classify_module(name) == grp:
                    self._synapse_module_names[grp] = name
                    val = mod.value().detach().to("cpu")
                    n_el = val.numel()
                    k = min(self.tracked_synapses_per_group, n_el)
                    # Pick evenly spaced indices across the weight matrix
                    idx = torch.linspace(0, n_el - 1, steps=k).long()
                    self._synapse_flat_indices[grp] = idx
                    re, im = val.to_xy()
                    st = val.state()
                    self._synapse_coords[grp].append(
                        (re.flatten()[idx].clone(), im.flatten()[idx].clone(), st.flatten()[idx].clone())
                    )
                    break

    @torch.no_grad()
    def record_step(self, step: int, model: nn.Module, loss: Optional[float] = None) -> None:
        """Record sector migration, step flip rate, 45° ratio, and radius for the current training step."""
        self.steps.append(int(step))
        self.losses.append(float("nan") if loss is None else float(loss))

        counts_total = {"attn": 0, "mlp": 0, "embed": 0}
        counts_cumul = {"attn": 0, "mlp": 0, "embed": 0}
        counts_flip = {"attn": 0, "mlp": 0, "embed": 0}
        counts_diag = {"attn": 0, "mlp": 0, "embed": 0}
        sum_radius = {"attn": 0.0, "mlp": 0.0, "embed": 0.0}

        for name, mod in _iter_complex_d16_parameters(model):
            grp = self._classify_module(name)
            val = mod.value().detach().to("cpu")
            st_curr = val.state()
            st_init = self.initial_snapshot[name].state()
            st_prev = self.prev_snapshot[name].state()

            n_el = val.numel()
            counts_total[grp] += n_el
            counts_cumul[grp] += int((st_curr != st_init).sum().item())
            counts_flip[grp] += int((st_curr != st_prev).sum().item())
            counts_diag[grp] += int((((st_curr & 3) == 1) | ((st_curr & 3) == 2)).sum().item())
            sum_radius[grp] += float(val.norm_squared().sqrt().sum().item())

            self.prev_snapshot[name] = val.clone()

            if grp in self._synapse_module_names and name == self._synapse_module_names[grp]:
                idx = self._synapse_flat_indices[grp]
                re, im = val.to_xy()
                self._synapse_coords[grp].append(
                    (re.flatten()[idx].clone(), im.flatten()[idx].clone(), st_curr.flatten()[idx].clone())
                )

        for grp in ("attn", "mlp", "embed"):
            tot = max(counts_total[grp], 1)
            self.cumul_migration_pct[grp].append(100.0 * counts_cumul[grp] / tot)
            self.step_flip_pct[grp].append(100.0 * counts_flip[grp] / tot)
            self.diag45_pct[grp].append(100.0 * counts_diag[grp] / tot)
            self.mean_radius[grp].append(sum_radius[grp] / tot)


def print_d16_trajectory_summary(recorder: D16TrajectoryRecorder, every_n: int = 5) -> None:
    """Print a step-by-step console table comparing Attention and MLP weight sector trajectories."""
    print("=" * 118)
    print("ДИНАМИКА СЕКТОРОВ ВЕСОВ D16Parameter ПО ШАГАМ ОБУЧЕНИЯ: ATTENTION vs MLP vs EMBEDDING")
    print("=" * 118)
    print(
        f"{'Шаг':>4} | {'Loss':>7} | {'Attn Cumul%':>11} | {'MLP Cumul%':>10} | "
        f"{'Emb Cumul%':>10} | {'Attn Flip/шаг':>13} | {'MLP Flip/шаг':>12} | "
        f"{'Attn 45°%':>9} | {'MLP 45°%':>8}"
    )
    print("-" * 118)

    for i, step in enumerate(recorder.steps):
        if i != 0 and i != len(recorder.steps) - 1 and (step % max(every_n, 1) != 0):
            continue
        loss_v = recorder.losses[i]
        loss_s = f"{loss_v:7.4f}" if math.isfinite(loss_v) else "    n/a"
        print(
            f"{step:4d} | {loss_s} | {recorder.cumul_migration_pct['attn'][i]:10.2f}% | "
            f"{recorder.cumul_migration_pct['mlp'][i]:9.2f}% | "
            f"{recorder.cumul_migration_pct['embed'][i]:9.2f}% | "
            f"{recorder.step_flip_pct['attn'][i]:12.2f}% | "
            f"{recorder.step_flip_pct['mlp'][i]:11.2f}% | "
            f"{recorder.diag45_pct['attn'][i]:8.2f}% | "
            f"{recorder.diag45_pct['mlp'][i]:7.2f}%"
        )
    print("=" * 118)


def plot_d16_sector_trajectories(recorder: D16TrajectoryRecorder):
    """Plot 4-panel comparison of Attention vs MLP weight sector trajectories over training steps."""
    import matplotlib.pyplot as plt
    import numpy as np

    steps = np.array(recorder.steps)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
    ax_cum, ax_flip = axes[0, 0], axes[0, 1]
    ax_diag, ax_rad = axes[1, 0], axes[1, 1]

    styles = {
        "attn": ("#1976d2", "o", "Attention (q/k/v/o_proj)"),
        "mlp": ("#d32f2f", "s", "MLP (up/down_proj)"),
        "embed": ("#388e3c", "^", "Embeddings (embed_tokens)"),
    }

    # 1. Cumulative sector migration from Step 0 (%)
    for grp, (col, mk, lbl) in styles.items():
        ax_cum.plot(
            steps, recorder.cumul_migration_pct[grp], color=col, marker=mk, markevery=max(len(steps) // 8, 1), lw=2.0, label=lbl
        )
    ax_cum.set_xlabel("Шаг обучения (Step)")
    ax_cum.set_ylabel("Сменили начальный сектор (%)")
    ax_cum.set_title("1. Накопленная миграция весов из начального сектора S_init")
    ax_cum.grid(True, linestyle=":", alpha=0.6)
    ax_cum.legend(fontsize=8)

    # 2. Instantaneous step-to-step sector flip rate (%)
    for grp, (col, mk, lbl) in styles.items():
        ax_flip.plot(
            steps, recorder.step_flip_pct[grp], color=col, marker=mk, markevery=max(len(steps) // 8, 1), lw=2.0, label=lbl
        )
    ax_flip.set_xlabel("Шаг обучения (Step)")
    ax_flip.set_ylabel("Переключений за 1 шаг (%)")
    ax_flip.set_title("2. Мгновенная скорость смены секторов за 1 шаг D16AdamW")
    ax_flip.grid(True, linestyle=":", alpha=0.6)
    ax_flip.legend(fontsize=8)

    # 3. Share of 45-degree diagonal sectors (S1, S2, S5, S6, S9, S10, S13, S14)
    for grp, (col, mk, lbl) in styles.items():
        ax_diag.plot(
            steps, recorder.diag45_pct[grp], color=col, marker=mk, markevery=max(len(steps) // 8, 1), lw=2.0, label=lbl
        )
    ax_diag.axhline(50.0, color="gray", linestyle="--", lw=1.2, label="Равновесие осевых/диагональных (50%)")
    ax_diag.set_xlabel("Шаг обучения (Step)")
    ax_diag.set_ylabel("Доля диагональных 45°-секторов (%)")
    ax_diag.set_title("3. Баланс осевых и 45°-диагональных секторов по шагам")
    ax_diag.grid(True, linestyle=":", alpha=0.6)
    ax_diag.legend(fontsize=8)

    # 4. Mean wedge radius |z| = sqrt(x^2 + y^2) & Loss overlay
    for grp, (col, mk, lbl) in styles.items():
        ax_rad.plot(
            steps, recorder.mean_radius[grp], color=col, marker=mk, markevery=max(len(steps) // 8, 1), lw=2.0, label=f"|z| {lbl}"
        )
    ax_rad.set_xlabel("Шаг обучения (Step)")
    ax_rad.set_ylabel("Средний радиус весов |z|")
    ax_rad.set_title("4. Эволюция среднего радиуса весов |z| и Loss")
    ax_rad.grid(True, linestyle=":", alpha=0.6)

    losses_arr = np.array(recorder.losses, dtype=float)
    if np.isfinite(losses_arr).any():
        ax_loss = ax_rad.twinx()
        ax_loss.plot(steps, losses_arr, color="#6a1b9a", linestyle="--", lw=1.8, label="Loss (NLL)")
        ax_loss.set_ylabel("Loss (NLL)", color="#6a1b9a")
        lines1, labels1 = ax_rad.get_legend_handles_labels()
        lines2, labels2 = ax_loss.get_legend_handles_labels()
        ax_rad.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    else:
        ax_rad.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    return fig


def plot_d16_synapse_polar_trajectories(recorder: D16TrajectoryRecorder):
    """Plot polar 16-sector phase-space trajectories of individual Attention and MLP synapses across steps."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax_attn, ax_mlp) = plt.subplots(1, 2, figsize=(14, 6.4), subplot_kw={"projection": "polar"})
    colors = ["#d32f2f", "#1976d2", "#388e3c", "#7b1fa2", "#ef6c00", "#00838f"]

    for ax, grp, title_prefix in (
        (ax_attn, "attn", "Attention"),
        (ax_mlp, "mlp", "MLP"),
    ):
        theta_wedge = np.linspace(0.0, math.pi / 8.0, 50)
        history = recorder._synapse_coords.get(grp, [])
        if not history:
            continue

        # Stack [num_steps + 1, num_synapses]
        re_hist = torch.stack([item[0] for item in history], dim=0).numpy()
        im_hist = torch.stack([item[1] for item in history], dim=0).numpy()
        st_hist = torch.stack([item[2] for item in history], dim=0).numpy()

        radii = np.hypot(re_hist, im_hist)
        angles = np.mod(np.arctan2(im_hist, re_hist), 2.0 * math.pi)
        max_r = float(max(radii.max(), 1e-4)) * 1.15

        ax.fill_between(theta_wedge, 0.0, max_r, color="#ffe082", alpha=0.35, label="Клин [0°, 22.5°]")
        for s in range(16):
            ang = s * (math.pi / 8.0)
            ax.plot([ang, ang], [0.0, max_r], color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
            mid_ang = (s + 0.5) * (math.pi / 8.0)
            ax.text(mid_ang, max_r * 0.93, f"S{s}", ha="center", va="center", fontsize=8, color="#333333")

        num_syn = re_hist.shape[1]
        flat_idx = recorder._synapse_flat_indices[grp].tolist()
        for k in range(num_syn):
            col = colors[k % len(colors)]
            th_k = angles[:, k]
            r_k = radii[:, k]
            s_start = int(st_hist[0, k])
            s_end = int(st_hist[-1, k])
            ax.plot(th_k, r_k, color=col, lw=1.8, alpha=0.85, label=f"w[{flat_idx[k]}]: S{s_start}→S{s_end}")
            ax.scatter([th_k[0]], [r_k[0]], color=col, marker="o", s=35, zorder=5)
            ax.scatter([th_k[-1]], [r_k[-1]], color=col, marker="*", s=95, edgecolor="black", linewidth=0.6, zorder=6)

        mod_name = recorder._synapse_module_names.get(grp, grp)
        ax.set_ylim(0.0, max_r)
        ax.set_title(
            f"Траектории синапсов {title_prefix} ({mod_name})\n"
            f"[Круг ○ = старт (шаг 0), Звезда ★ = финал]",
            pad=18,
            fontsize=10,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.34, 1.12), fontsize=8)

    plt.tight_layout()
    return fig


__all__ = [
    "print_orbit_table",
    "plot_single_orbit_compass",
    "plot_layer_z8z2_spectrum",
    "plot_learned_sector_gains",
    "snapshot_d16_weight_states",
    "print_d16_weight_migration_table",
    "plot_d16_weight_sector_distribution",
    "plot_d16_sector_transition_matrix",
    "D16TrajectoryRecorder",
    "print_d16_trajectory_summary",
    "plot_d16_sector_trajectories",
    "plot_d16_synapse_polar_trajectories",
]
