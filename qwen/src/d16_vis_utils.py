"""Visualization and Inspection Utilities for D16Complex Orbit Conversion.

Provides console tables and matplotlib polar/bar plots to inspect:
1. Independent D16Complex objects (neurons) that lie on the same Z_8 |>< Z_2 orbit.
2. Polar 16-sector compass showing where each of the 4 orbit-related neurons points
   inside the anchor's complex plane and fundamental wedge [0°, 22.5°].
3. Histogram of all 16 dihedral operations of Z_8 |>< Z_2 (Axial Im vs 45° Hadamard Re).
4. Polar 16-sector response profiles of _eval_d16(z) for axial (x*y) vs diagonal 0.5*(x^2-y^2) branches.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from d16_complex import D16Complex, D16ComplexTensor
from d16_qwen_converter import ConversionReport, LayerOrbitReport
from d16_qwen_model import D16QwenMLP


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
        f"Слой {layer_report.layer_idx} | Орбита #{unit_idx} (Score = {score_val:.4f})"
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
        f"Слой {layer_idx}: Спектр 16 операций Z_8 |>< Z_2"
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
            # 1. Базовый отклик _eval_d16(z)
            act_base = d16_mlp._eval_d16(z).abs().cpu().numpy()
            # 2. Отклик при побитовом повороте маски rotation на 45° (op_id=8: rol на 2 бита)
            z_rot45 = D16ComplexTensor(z.x, z.y, (z.rotation << 2 | (z.rotation >> 14)) & 0xFFFF)
            act_rot45 = d16_mlp._eval_d16(z_rot45).abs().cpu().numpy()

            axes[0].plot(angles, act_base, lw=1.8, label=f"D16 #{u_idx} (rot)")
            axes[1].plot(angles, act_rot45, lw=1.8, label=f"D16 #{u_idx} (rol(rot, 2))")

    axes[0].set_title("Отклик _eval_d16(z) по 16 секторам (исходный rotation)", pad=15)
    axes[1].set_title("Отклик _eval_d16(z) при сдвиге маски rol(rotation, 2) (+45°)", pad=15)
    for ax in axes:
        ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)

    plt.tight_layout()
    return fig


__all__ = [
    "print_orbit_table",
    "plot_single_orbit_compass",
    "plot_layer_z8z2_spectrum",
    "plot_learned_sector_gains",
]
