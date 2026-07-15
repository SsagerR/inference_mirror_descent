#!/usr/bin/env python3
"""Plot heatmaps for RSM-vs-MALA-distillation toy sweeps."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["pi_orig_distill", "rsm", "mala_distill"]
DEFAULT_METRICS = ["js", "kl_sample_target", "w1", "ks", "sliced_w1", "marginal_ks_x", "marginal_ks_y"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--metrics", default="auto")
    p.add_argument("--stage", default="final_clean")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def as_float(value, default=math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def beta_value(row: dict[str, str]) -> str:
    return row.get("beta_input") or row.get("beta") or ""


def budget_value(row: dict[str, str]) -> str:
    return row.get("compute_budget") or row.get("sample_budget") or ""


def numeric_sorted_unique(values) -> list[str]:
    vals = []
    seen = set()
    for value in values:
        if value is None or value == "":
            continue
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        vals.append(key)
    return sorted(vals, key=lambda x: as_float(x))


def metric_list(rows: list[dict[str, str]], requested: str) -> list[str]:
    available = set().union(*(row.keys() for row in rows)) if rows else set()
    if requested.strip().lower() == "auto":
        return [metric for metric in DEFAULT_METRICS if metric in available]
    return [metric for metric in requested.replace(",", " ").split() if metric and metric in available]


def mean_cell(rows: list[dict[str, str]], metric: str, method: str, beta: str, budget: str) -> tuple[float, int]:
    vals = [
        as_float(row.get(metric))
        for row in rows
        if row.get("method") == method and str(beta_value(row)) == str(beta) and str(budget_value(row)) == str(budget)
    ]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return math.nan, 0
    return float(np.mean(vals)), len(vals)


def collect_grid(rows: list[dict[str, str]], metric: str, method: str, betas: list[str], budgets: list[str]) -> np.ndarray:
    mat = np.full((len(betas), len(budgets)), np.nan, dtype=np.float64)
    for i, beta in enumerate(betas):
        for j, budget in enumerate(budgets):
            mat[i, j] = mean_cell(rows, metric, method, beta, budget)[0]
    return mat


def collect_delta_grid(
    rows: list[dict[str, str]],
    metric: str,
    left_method: str,
    right_method: str,
    betas: list[str],
    budgets: list[str],
) -> np.ndarray:
    mat = np.full((len(betas), len(budgets)), np.nan, dtype=np.float64)
    for i, beta in enumerate(betas):
        for j, budget in enumerate(budgets):
            left, _ = mean_cell(rows, metric, left_method, beta, budget)
            right, _ = mean_cell(rows, metric, right_method, beta, budget)
            if np.isfinite(left) and np.isfinite(right):
                mat[i, j] = left - right
    return mat


def finite_limits(mats: list[np.ndarray], *, symmetric: bool = False) -> tuple[float, float]:
    vals = np.concatenate([mat[np.isfinite(mat)] for mat in mats if np.isfinite(mat).any()]) if any(np.isfinite(mat).any() for mat in mats) else np.asarray([1.0])
    if symmetric:
        lim = float(np.max(np.abs(vals)))
        return (-lim if lim > 0 else -1.0, lim if lim > 0 else 1.0)
    return float(np.min(vals)), float(np.max(vals))


def add_cell_text(ax, mat: np.ndarray) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=7, color="black")


def save_raw(rows: list[dict[str, str]], metric: str, betas: list[str], budgets: list[str], out_path: Path, dpi: int) -> bool:
    mats = [collect_grid(rows, metric, method, betas, budgets) for method in METHODS]
    if not any(np.isfinite(mat).any() for mat in mats):
        return False
    vmin, vmax = finite_limits(mats)
    fig, axes = plt.subplots(1, len(METHODS), figsize=(4.0 * len(METHODS), 3.5), squeeze=False)
    for ax, method, mat in zip(axes.ravel(), METHODS, mats):
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        add_cell_text(ax, mat)
        ax.set_title(method, fontsize=9)
        ax.set_xticks(range(len(budgets)), budgets, rotation=30, ha="right")
        ax.set_yticks(range(len(betas)), betas)
        ax.set_xlabel("compute_budget")
        ax.set_ylabel("beta")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(f"{metric}: mean over dataset seeds", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def save_reference_delta(rows: list[dict[str, str]], metric: str, betas: list[str], budgets: list[str], out_path: Path, dpi: int) -> bool:
    panels = [("rsm", "pi_orig_distill"), ("mala_distill", "pi_orig_distill")]
    mats = [collect_delta_grid(rows, metric, left, right, betas, budgets) for left, right in panels]
    if not any(np.isfinite(mat).any() for mat in mats):
        return False
    vmin, vmax = finite_limits(mats, symmetric=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 3.5), squeeze=False)
    for ax, (left, right), mat in zip(axes.ravel(), panels, mats):
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        add_cell_text(ax, mat)
        ax.set_title(f"{left} - {right}", fontsize=9)
        ax.set_xticks(range(len(budgets)), budgets, rotation=30, ha="right")
        ax.set_yticks(range(len(betas)), betas)
        ax.set_xlabel("compute_budget")
        ax.set_ylabel("beta")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(f"{metric}: method - pi_orig_distill; negative is better", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def save_rsm_minus_mala(rows: list[dict[str, str]], metric: str, betas: list[str], budgets: list[str], out_path: Path, dpi: int) -> bool:
    mat = collect_delta_grid(rows, metric, "rsm", "mala_distill", betas, budgets)
    if not np.isfinite(mat).any():
        return False
    vmin, vmax = finite_limits([mat], symmetric=True)
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    add_cell_text(ax, mat)
    ax.set_title("rsm - mala_distill", fontsize=9)
    ax.set_xticks(range(len(budgets)), budgets, rotation=30, ha="right")
    ax.set_yticks(range(len(betas)), betas)
    ax.set_xlabel("compute_budget")
    ax.set_ylabel("beta")
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(f"{metric}: positive means MALA-distill is better", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    rows = [row for row in read_rows(args.metrics_file) if row.get("stage") == args.stage]
    if not rows:
        raise SystemExit(f"no rows found for stage={args.stage!r}")
    metrics = metric_list(rows, args.metrics)
    if not metrics:
        raise SystemExit("no requested metrics found")
    betas = numeric_sorted_unique(beta_value(row) for row in rows)
    budgets = numeric_sorted_unique(budget_value(row) for row in rows)
    count = 0
    for metric in metrics:
        if save_raw(rows, metric, betas, budgets, args.out_dir / "raw" / f"{metric}.png", args.dpi):
            count += 1
        if save_reference_delta(rows, metric, betas, budgets, args.out_dir / "method_minus_pi_orig_distill" / f"{metric}.png", args.dpi):
            count += 1
        if save_rsm_minus_mala(rows, metric, betas, budgets, args.out_dir / "rsm_minus_mala_distill" / f"{metric}.png", args.dpi):
            count += 1
    print(f"wrote {count} heatmap image(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
