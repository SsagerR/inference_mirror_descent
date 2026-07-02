#!/usr/bin/env python3
"""Plot denoising-schedule sweeps from an existing sweep_metrics.csv."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DENOISING_ORDER = [
    "all_DDPM_mean",
    "all_DDIM",
    "all_Identity",
    "DDPM_then_last1_DDIM",
    "DDPM_then_last2_DDIM",
    "Identity_then_last1_DDIM",
    "Identity_then_last2_DDIM",
]
MALA_SCHEDULE_ORDER = ["constant", "linear_low_noise", "quadratic_low_noise"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--stages", default="final_clean,t000")
    p.add_argument(
        "--metrics",
        default=(
            "kl_sample_target,js,ks,marginal_ks_x,marginal_ks_y,"
            "sample_mean_q,target_mean_q,acceptance_rate"
        ),
    )
    p.add_argument("--baseline-denoising-schedule", default="all_DDPM_mean")
    return p.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def as_float(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def row_stage_key(row: dict) -> str | None:
    stage = row.get("stage")
    if stage == "final_clean":
        return "final_clean"
    if stage == "intermediate" and str(row.get("timestep")) in {"0", "0.0"}:
        return "t000"
    return None


def sorted_values(values, preferred_order=None):
    vals = sorted(set(v for v in values if v not in {None, ""}))
    if preferred_order is None:
        return vals
    rank = {v: i for i, v in enumerate(preferred_order)}
    return sorted(vals, key=lambda v: (rank.get(v, 999), v))


def cell(rows, metric, denoise, budget, mala_schedule, eta):
    vals = []
    for row in rows:
        if (row.get("denoising_schedule") or "from_predictor") != denoise:
            continue
        if str(row.get("mala_budget", "")) != str(budget):
            continue
        if (row.get("mala_step_schedule") or "constant") != mala_schedule:
            continue
        if str(row.get("mala_eta", "")) != str(eta):
            continue
        value = as_float(row.get(metric))
        if np.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else math.nan


def matrix(rows, metric, denoises, budgets, mala_schedule, eta):
    mat = np.full((len(budgets), len(denoises)), np.nan, dtype=np.float64)
    for i, budget in enumerate(budgets):
        for j, denoise in enumerate(denoises):
            mat[i, j] = cell(rows, metric, denoise, budget, mala_schedule, eta)
    return mat


def add_text(ax, mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3g}", ha="center", va="center", fontsize=7)


def plot_metric(rows, metric, stage_key, out_path, delta=False, baseline="all_DDPM_mean") -> bool:
    stage_rows = [
        row
        for row in rows
        if row_stage_key(row) == stage_key
        and (row.get("sampler") or "mala") == "mala"
        and (row.get("guidance_gradient_space") or "xt") == "xt"
    ]
    if not stage_rows or all(metric not in row for row in stage_rows):
        return False

    denoises = sorted_values([row.get("denoising_schedule") for row in stage_rows], DENOISING_ORDER)
    if delta:
        denoises = [d for d in denoises if d != baseline]
    budgets = sorted_values([row.get("mala_budget") for row in stage_rows])
    budgets = sorted(budgets, key=lambda x: as_float(x, 0.0))
    mala_schedules = sorted_values([row.get("mala_step_schedule") for row in stage_rows], MALA_SCHEDULE_ORDER)
    etas = sorted_values([row.get("mala_eta") for row in stage_rows])
    etas = sorted(etas, key=lambda x: as_float(x, 0.0))
    panel_keys = [(s, e) for s in mala_schedules for e in etas]
    if not denoises or not budgets or not panel_keys:
        return False

    fig, axes = plt.subplots(
        len(panel_keys),
        1,
        figsize=(max(8.0, 1.45 * len(denoises)), 2.45 * len(panel_keys)),
        squeeze=False,
    )
    for ax, (mala_schedule, eta) in zip(axes.ravel(), panel_keys):
        mat = matrix(stage_rows, metric, denoises, budgets, mala_schedule, eta)
        if delta:
            base = matrix(stage_rows, metric, [baseline], budgets, mala_schedule, eta)
            mat = mat - base
            finite = mat[np.isfinite(mat)]
            vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
            im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
        add_text(ax, mat)
        ax.set_xticks(range(len(denoises)), denoises, rotation=25, ha="right")
        ax.set_yticks(range(len(budgets)), budgets)
        ax.set_ylabel("budget")
        ax.set_title(f"{mala_schedule}, eta={eta}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.82)

    kind = f"delta vs {baseline}" if delta else "raw"
    fig.suptitle(f"{stage_key}: {metric} ({kind})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    rows = read_rows(args.metrics_file)
    stages = [s.strip() for s in args.stages.replace(",", " ").split() if s.strip()]
    metrics = [m.strip() for m in args.metrics.replace(",", " ").split() if m.strip()]
    count = 0
    for stage in stages:
        for metric in metrics:
            if plot_metric(rows, metric, stage, args.out_dir / "raw" / stage / f"{metric}.png", delta=False):
                count += 1
            if plot_metric(
                rows,
                metric,
                stage,
                args.out_dir / "delta_vs_all_DDPM_mean" / stage / f"{metric}.png",
                delta=True,
                baseline=args.baseline_denoising_schedule,
            ):
                count += 1
    print(f"wrote {count} denoising-schedule plot(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
