"""Visualization and Inspection Utilities for D16Complex Orbit Conversion.

Provides console tables and matplotlib polar/bar plots to inspect:
1. Linked real neurons (quadruplets) forming a single D16Complex unit.
2. Polar 16-sector compass showing where each of the 4 linked neurons points
   inside the canonical complex plane (U_q, G_q) and fundamental wedge [0°, 22.5°].
3. Histogram of all 16 dihedral operations of Z_8 |>< Z_2 (Axial Im vs 45° Hadamard Re).
4. Polar profiles of the 18-state sector_im_gain and sector_re_gain tables.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch

from d16_complex import D16Complex, STATE_LABELS
from d16_qwen_converter import ConversionReport, LayerOrbitReport, OP_NAMES
from d16_qwen_model import D16QwenMLP


def print_orbit_table(layer_report: LayerOrbitReport, top_k: int = 10) -> None:
    """Print a detailed table of top-k linked neuron quadruplets in a converted layer."""
    print("=" * 116)
    print(
        f"СЛОЙ {layer_report.layer_idx:2d} | Режим: {layer_report.mode} | "
        f"Нейронов: {layer_report.orig_intermediate_size} -> {layer_report.d16_intermediate_size} D16Complex | "
        f"Средний score: {layer_report.mean_orbit_score:.4f} | Доля 45° Re(z^2): {layer_report.hadamard_45_pct:.1f}%"
    )
    print("=" * 116)
    print(
        f"{'D16 #':>5} | {'Score':>6} | {'Якорь (m=0)':<14} | "
        f"{'Партнер m=1 (op)':<26} | {'Партнер m=2 (op)':<26} | {'Партнер m=3 (op)':<26}"
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
            f"{q:5d} | {sc:6.4f} | n={n[0]:<4d} [op= 0] | "
            f"{_fmt_member(1):<26} | {_fmt_member(2):<26} | {_fmt_member(3):<26}"
        )
    print("=" * 116)


def plot_single_orbit_compass(
    layer_report: LayerOrbitReport,
    unit_idx: int = 0,
    ax=None,
):
    """Plot a polar 16-sector compass showing the 4 linked neurons of D16Complex unit_idx."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure

    # 1. Shade the fundamental wedge [0°, 22.5°] and draw all 16 sector boundaries
    theta_wedge = np.linspace(0.0, math.pi / 8.0, 50)
    ax.fill_between(theta_wedge, 0.0, 1.15, color="#gold" if False else "#ffe082", alpha=0.45, label="Клин [0°, 22.5°]")

    for s in range(16):
        ang = s * (math.pi / 8.0)
        ax.plot([ang, ang], [0.0, 1.15], color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        mid_ang = (s + 0.5) * (math.pi / 8.0)
        ax.text(mid_ang, 1.08, f"S{s}", ha="center", va="center", fontsize=8, color="#333333")

    # 2. Plot local 2D vectors of the 4 linked neurons inside canonical basis (U_q, G_q)
    quad = layer_report.quadruplets[unit_idx].tolist()
    ops = layer_report.group_ops[unit_idx].tolist()
    loc = layer_report.local_2x2[unit_idx]  # [4, 2, 2]: row 0 is u_m, row 1 is g_m in (U_q, G_q)

    colors = ["#d32f2f", "#1976d2", "#388e3c", "#7b1fa2"]
    max_r = float(loc.norm(dim=-1).max().clamp_min(1e-6).item())

    for m in range(4):
        # Diagonal vector (u_m + g_m) or gate vector g_m in canonical plane (L, E)
        gx, gy = float(loc[m, 1, 0].item()) / max_r, float(loc[m, 1, 1].item()) / max_r
        z_m = D16Complex.from_xy(gx, gy)
        angle = math.atan2(gy, gx) % (2.0 * math.pi)
        radius = math.hypot(gx, gy)
        op_id = ops[m]
        branch_tag = "Re 45°" if op_id >= 8 else "Im 0°/90°"
        lbl = f"m={m}: нейрон #{quad[m]} (op={op_id}, {branch_tag}, S{z_m.state()})"
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
        f"Слой {layer_report.layer_idx} | Узел D16Complex #{unit_idx} (Score = {score_val:.4f})\n"
        f"Связанная четверка нейронов: {quad}",
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

    # Left: 16 operations histogram for selected layer
    counts = layer_rep.op_histogram.numpy()
    ops_x = np.arange(16)
    bar_colors = ["#1976d2" if op < 8 else "#e65100" for op in ops_x]
    ax1.bar(ops_x, counts, color=bar_colors, alpha=0.85, edgecolor="black")
    ax1.set_xticks(ops_x)
    ax1.set_xticklabels([f"{op}" for op in ops_x])
    ax1.set_xlabel("Индекс операции группы Z_8 |>< Z_2 (0..7: Im z^2, 8..15: 45° Re z^2)")
    ax1.set_ylabel("Число связей в четверках")
    ax1.set_title(
        f"Слой {layer_idx}: Спектр 16 операций Z_8 |>< Z_2\n"
        f"(Доля 45°-поворотов Адамара Re(z^2): {layer_rep.hadamard_45_pct:.1f}%)"
    )
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Right: Layer-by-layer curve of 45° Hadamard percentage & mean orbit score
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
    """Plot polar 16-sector gain profiles (sector_im_gain & sector_re_gain) of D16QwenMLP units."""
    import matplotlib.pyplot as plt
    import numpy as np

    angles = np.linspace(0.0, 2.0 * math.pi, 17)  # close the loop
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), subplot_kw={"projection": "polar"})

    with torch.no_grad():
        im_gains = d16_mlp.sector_im_gain.detach().cpu().numpy()
        re_gains = d16_mlp.sector_re_gain.detach().cpu().numpy()

    for u_idx in unit_indices:
        im_vals = np.append(im_gains[u_idx, :16], im_gains[u_idx, 0])
        re_vals = np.append(re_gains[u_idx, :16], re_gains[u_idx, 0])
        axes[0].plot(angles, im_vals, marker="o", lw=1.8, label=f"Unit #{u_idx}")
        axes[1].plot(angles, re_vals, marker="s", lw=1.8, label=f"Unit #{u_idx}")

    axes[0].set_title("Профиль 16 секторов: sector_im_gain (Im z^2)", pad=15)
    axes[1].set_title("Профиль 16 секторов: sector_re_gain (Re z^2)", pad=15)
    for ax in axes:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=8)

    plt.tight_layout()
    return fig


__all__ = [
    "print_orbit_table",
    "plot_single_orbit_compass",
    "plot_layer_z8z2_spectrum",
    "plot_learned_sector_gains",
]