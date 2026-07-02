#!/usr/bin/env python3
"""Plot MALA budget-schedule sweeps from an existing sweep_metrics.csv."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEDULE_ORDER = [
    "constant",
    "continuous_uniform_sqrt_alpha_bar_min1",
    "linear_low_noise",
    "quadratic_low_noise",
]
PREDICTOR_ORDER = ["DDPM_mean", "DDIM", "Identity"]
GRADIENT_ORDER = ["xt", "x0hat", "x0hatclipped"]


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
    p.add_argument("--guidance-schedule", default="constant")
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


def collect_cell(rows: list[dict], metric: str, schedule: str, budget: str, predictor: str, gradient: str, eta: str):
    vals = []
    for row in rows:
        if (row.get("mala_step_schedule") or "constant") != schedule:
            continue
        if str(row.get("mala_budget", "")) != str(budget):
            continue
        if row.get("denoising_predictor") != predictor:
            continue
        if row.get("guidance_gradient_space") != gradient:
            continue
        if str(row.get("mala_eta", "")) != str(eta):
            continue
        value = as_float(row.get(metric))
        if not math.isnan(value):
            vals.append(value)
    if not vals:
        return math.nan
    return float(np.mean(vals))


def heatmap_matrix(rows, metric, schedule, budgets, predictors, gradient, eta):
    mat = np.full((len(budgets), len(predictors)), np.nan, dtype=np.float64)
    for i, budget in enumerate(budgets):
        for j, predictor in enumerate(predictors):
            mat[i, j] = collect_cell(rows, metric, schedule, budget, predictor, gradient, eta)
    return mat


def add_text(ax, mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=8, color="black")


def plot_grid(rows, metric, stage_key, out_path: Path, delta: bool = False) -> bool:
    stage_rows = [
        row
        for row in rows
        if row_stage_key(row) == stage_key
        and (row.get("sampler") or "mala") == "mala"
    ]
    if not stage_rows or all(metric not in row for row in stage_rows):
        return False

    schedules = sorted_values(
        [row.get("mala_step_schedule") or "constant" for row in stage_rows],
        SCHEDULE_ORDER,
    )
    if delta:
        schedules = [s for s in schedules if s != "constant"]
    budgets = sorted_values([row.get("mala_budget") for row in stage_rows], None)
    budgets = sorted(budgets, key=lambda x: as_float(x, 0.0))
    predictors = sorted_values([row.get("denoising_predictor") for row in stage_rows], PREDICTOR_ORDER)
    gradients = sorted_values([row.get("guidance_gradient_space") for row in stage_rows], GRADIENT_ORDER)
    etas = sorted_values([row.get("mala_eta") for row in stage_rows], None)
    etas = sorted(etas, key=lambda x: as_float(x, 0.0))
    panel_keys = [(g, e) for g in gradients for e in etas]
    if not schedules or not budgets or not predictors or not panel_keys:
        return False

    fig, axes = plt.subplots(
        len(schedules),
        len(panel_keys),
        figsize=(3.5 * len(panel_keys), 2.8 * len(schedules)),
        squeeze=False,
    )
    for r, schedule in enumerate(schedules):
        for c, (gradient, eta) in enumerate(panel_keys):
            ax = axes[r, c]
            mat = heatmap_matrix(stage_rows, metric, schedule, budgets, predictors, gradient, eta)
            if delta:
                base = heatmap_matrix(stage_rows, metric, "constant", budgets, predictors, gradient, eta)
                mat = mat - base
                finite = mat[np.isfinite(mat)]
                vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
                im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(mat, aspect="auto", cmap="viridis")
            add_text(ax, mat)
            ax.set_xticks(range(len(predictors)), predictors, rotation=25, ha="right")
            ax.set_yticks(range(len(budgets)), budgets)
            if c == 0:
                ax.set_ylabel(f"{schedule}\nbudget")
            if r == 0:
                ax.set_title(f"{gradient}, eta={eta}", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.78)

    title_kind = "delta vs constant" if delta else "raw"
    fig.suptitle(f"{stage_key}: {metric} ({title_kind})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    rows = read_rows(args.metrics_file)
    rows = [row for row in rows if (row.get("guidance_schedule") or "constant") == args.guidance_schedule]
    stages = [s.strip() for s in args.stages.replace(",", " ").split() if s.strip()]
    metrics = [m.strip() for m in args.metrics.replace(",", " ").split() if m.strip()]
    count = 0
    for stage in stages:
        for metric in metrics:
            raw_path = args.out_dir / "raw" / stage / f"{metric}.png"
            if plot_grid(rows, metric, stage, raw_path, delta=False):
                count += 1
            delta_path = args.out_dir / "delta_vs_constant" / stage / f"{metric}.png"
            if plot_grid(rows, metric, stage, delta_path, delta=True):
                count += 1
    print(f"wrote {count} budget-schedule plot(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
