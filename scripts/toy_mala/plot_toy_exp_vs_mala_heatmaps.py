#!/usr/bin/env python3
"""Plot heatmaps for toy exponential-energy diffusion vs MALA sweeps."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["exponential_energy", "pi_orig_diffusion", "orig_energy_mala"]
MALA_INDEPENDENT_METHODS = {"exponential_energy", "pi_orig_diffusion"}
DEFAULT_METRICS = [
    "js",
    "kl_sample_target",
    "w1",
    "ks",
    "sliced_w1",
    "marginal_ks_x",
    "marginal_ks_y",
    "sample_mean_q",
    "target_mean_q",
    "acceptance_rate",
    "train_loss_final",
    "weight_max",
    "dataset_ess_over_N",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--metrics",
        default="auto",
        help="Comma/space separated metrics, or 'auto' for a useful default set present in the CSV.",
    )
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


def numeric_sorted_unique(values) -> list[str]:
    vals = []
    seen = set()
    for value in values:
        if value is None or value == "":
            continue
        f = as_float(value)
        if math.isnan(f):
            continue
        key = str(value)
        if key not in seen:
            vals.append(key)
            seen.add(key)
    return sorted(vals, key=lambda x: as_float(x))


def beta_value(row: dict[str, str]) -> str:
    return row.get("beta_input") or row.get("beta") or ""


def stage_matches(row: dict[str, str], stage: str) -> bool:
    if not stage:
        return True
    return (row.get("stage") or "") == stage


def metric_list(rows: list[dict[str, str]], requested: str) -> list[str]:
    available = set().union(*(row.keys() for row in rows)) if rows else set()
    if requested.strip().lower() == "auto":
        return [m for m in DEFAULT_METRICS if m in available]
    metrics = [m.strip() for m in requested.replace(",", " ").split() if m.strip()]
    return [m for m in metrics if m in available]


def mean_for_cell(
    rows: list[dict[str, str]],
    metric: str,
    method: str,
    mala_steps: str,
    beta: str,
    budget: str,
) -> tuple[float, int]:
    vals = []
    for row in rows:
        if row.get("method") != method:
            continue
        if method not in MALA_INDEPENDENT_METHODS and str(row.get("mala_steps", "")) != str(mala_steps):
            continue
        if str(beta_value(row)) != str(beta):
            continue
        if str(row.get("sample_budget", "")) != str(budget):
            continue
        value = as_float(row.get(metric))
        if np.isfinite(value):
            vals.append(value)
    if not vals:
        return math.nan, 0
    return float(np.mean(vals)), len(vals)


def collect_mean_grid(
    rows: list[dict[str, str]],
    metric: str,
    method: str,
    mala_steps: str,
    betas: list[str],
    budgets: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    mat = np.full((len(betas), len(budgets)), np.nan, dtype=np.float64)
    counts = np.zeros((len(betas), len(budgets)), dtype=np.int64)
    for i, beta in enumerate(betas):
        for j, budget in enumerate(budgets):
            if method == "delta_exp_minus_mala":
                exp_value, exp_n = mean_for_cell(rows, metric, "exponential_energy", mala_steps, beta, budget)
                mala_value, mala_n = mean_for_cell(rows, metric, "orig_energy_mala", mala_steps, beta, budget)
                if np.isfinite(exp_value) and np.isfinite(mala_value):
                    mat[i, j] = exp_value - mala_value
                    counts[i, j] = min(exp_n, mala_n)
            else:
                mat[i, j], counts[i, j] = mean_for_cell(rows, metric, method, mala_steps, beta, budget)
    return mat, counts


def add_cell_text(ax, mat: np.ndarray) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=7, color="black")


def finite_limits(mats: list[np.ndarray], *, symmetric: bool = False) -> tuple[float | None, float | None]:
    vals = np.concatenate([m[np.isfinite(m)] for m in mats if np.isfinite(m).any()]) if any(np.isfinite(m).any() for m in mats) else np.array([])
    if vals.size == 0:
        return None, None
    if symmetric:
        vmax = float(np.max(np.abs(vals)))
        if vmax == 0.0:
            vmax = 1.0
        return -vmax, vmax
    return float(np.min(vals)), float(np.max(vals))


def plot_raw_metric(
    rows: list[dict[str, str]],
    metric: str,
    mala_steps_values: list[str],
    betas: list[str],
    budgets: list[str],
    out_path: Path,
    dpi: int,
) -> bool:
    mats = [
        collect_mean_grid(rows, metric, method, steps, betas, budgets)[0]
        for method in METHODS
        for steps in mala_steps_values
    ]
    if not any(np.isfinite(m).any() for m in mats):
        return False
    vmin, vmax = finite_limits(mats)
    fig, axes = plt.subplots(
        len(METHODS),
        len(mala_steps_values),
        figsize=(3.2 * len(mala_steps_values), 3.0 * len(METHODS)),
        squeeze=False,
    )
    for r, method in enumerate(METHODS):
        for c, steps in enumerate(mala_steps_values):
            ax = axes[r, c]
            mat, _counts = collect_mean_grid(rows, metric, method, steps, betas, budgets)
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            add_cell_text(ax, mat)
            if method in MALA_INDEPENDENT_METHODS:
                title = f"{method}\nMALA-independent"
            else:
                title = f"{method}\nmala_steps={steps}"
            ax.set_title(title, fontsize=9)
            ax.set_xticks(range(len(budgets)), budgets, rotation=30, ha="right")
            ax.set_yticks(range(len(betas)), betas)
            if c == 0:
                ax.set_ylabel("beta")
            if r == len(METHODS) - 1:
                ax.set_xlabel("sample_budget")
            fig.colorbar(im, ax=ax, shrink=0.78)
    fig.suptitle(f"{metric}: mean over dataset seeds", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def mala_comparison_steps(rows: list[dict[str, str]]) -> list[str]:
    """Return MALA-step values that belong to the MALA method.

    ``exponential_energy`` is independent of MALA steps.  Its rows may either
    be canonicalized to ``mala_steps=0`` or duplicated across historical
    MALA-step sweeps, so the comparison axis must be defined by
    ``orig_energy_mala`` only.
    """
    values = [row.get("mala_steps") for row in rows if row.get("method") == "orig_energy_mala"]
    return numeric_sorted_unique(values)


def plot_delta_metric(
    rows: list[dict[str, str]],
    metric: str,
    mala_steps_values: list[str],
    betas: list[str],
    budgets: list[str],
    out_path: Path,
    dpi: int,
) -> bool:
    mats = [
        collect_mean_grid(rows, metric, "delta_exp_minus_mala", steps, betas, budgets)[0]
        for steps in mala_steps_values
    ]
    if not any(np.isfinite(m).any() for m in mats):
        return False
    vmin, vmax = finite_limits(mats, symmetric=True)
    fig, axes = plt.subplots(1, len(mala_steps_values), figsize=(3.4 * len(mala_steps_values), 3.1), squeeze=False)
    for c, steps in enumerate(mala_steps_values):
        ax = axes[0, c]
        mat, _counts = collect_mean_grid(rows, metric, "delta_exp_minus_mala", steps, betas, budgets)
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        add_cell_text(ax, mat)
        ax.set_title(f"mala_steps={steps}", fontsize=9)
        ax.set_xticks(range(len(budgets)), budgets, rotation=30, ha="right")
        ax.set_yticks(range(len(betas)), betas)
        if c == 0:
            ax.set_ylabel("beta")
        ax.set_xlabel("sample_budget")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{metric}: exponential_energy - orig_energy_mala", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def write_metric_tables(
    rows: list[dict[str, str]],
    metric: str,
    mala_steps_values: list[str],
    betas: list[str],
    budgets: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_path = out_dir / f"{metric}_cells.csv"
    with cell_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "mala_steps",
                "beta",
                "sample_budget",
                "exponential_energy",
                "orig_energy_mala",
                "delta_exp_minus_mala",
                "paired_seed_count",
            ],
        )
        writer.writeheader()
        for steps in mala_steps_values:
            for beta in betas:
                for budget in budgets:
                    exp_value, exp_n = mean_for_cell(rows, metric, "exponential_energy", steps, beta, budget)
                    mala_value, mala_n = mean_for_cell(rows, metric, "orig_energy_mala", steps, beta, budget)
                    delta = exp_value - mala_value if np.isfinite(exp_value) and np.isfinite(mala_value) else math.nan
                    writer.writerow({
                        "metric": metric,
                        "mala_steps": steps,
                        "beta": beta,
                        "sample_budget": budget,
                        "exponential_energy": exp_value,
                        "orig_energy_mala": mala_value,
                        "delta_exp_minus_mala": delta,
                        "paired_seed_count": min(exp_n, mala_n),
                    })

    by_beta = {}
    for steps in mala_steps_values:
        mat, _counts = collect_mean_grid(rows, metric, "delta_exp_minus_mala", steps, betas, budgets)
        for i, beta in enumerate(betas):
            vals = mat[i, np.isfinite(mat[i])]
            if vals.size:
                by_beta.setdefault(beta, []).extend(float(v) for v in vals)
    beta_path = out_dir / f"{metric}_by_beta.csv"
    with beta_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "beta",
                "n_cells",
                "exp_wins_delta_lt_0",
                "mala_wins_delta_gt_0",
                "mean_delta_exp_minus_mala",
                "median_delta_exp_minus_mala",
            ],
        )
        writer.writeheader()
        for beta in betas:
            vals = np.asarray(by_beta.get(beta, []), dtype=np.float64)
            if vals.size == 0:
                continue
            writer.writerow({
                "metric": metric,
                "beta": beta,
                "n_cells": int(vals.size),
                "exp_wins_delta_lt_0": int(np.sum(vals < 0)),
                "mala_wins_delta_gt_0": int(np.sum(vals > 0)),
                "mean_delta_exp_minus_mala": float(np.mean(vals)),
                "median_delta_exp_minus_mala": float(np.median(vals)),
            })


def main() -> None:
    args = parse_args()
    rows = [row for row in read_rows(args.metrics_file) if stage_matches(row, args.stage)]
    if not rows:
        raise SystemExit(f"no rows found for stage={args.stage!r} in {args.metrics_file}")
    betas = numeric_sorted_unique(beta_value(row) for row in rows)
    budgets = numeric_sorted_unique(row.get("sample_budget") for row in rows)
    mala_steps_values = mala_comparison_steps(rows)
    if not mala_steps_values:
        mala_steps_values = numeric_sorted_unique(row.get("mala_steps") for row in rows)
    metrics = metric_list(rows, args.metrics)
    if not metrics:
        raise SystemExit("no requested metrics were present in the CSV")

    plot_count = 0
    for metric in metrics:
        if plot_raw_metric(rows, metric, mala_steps_values, betas, budgets, args.out_dir / "raw" / f"{metric}.png", args.dpi):
            plot_count += 1
        if plot_delta_metric(rows, metric, mala_steps_values, betas, budgets, args.out_dir / "delta_exp_minus_mala" / f"{metric}.png", args.dpi):
            plot_count += 1
        write_metric_tables(rows, metric, mala_steps_values, betas, budgets, args.out_dir / "tables")

    print(f"wrote {plot_count} heatmap image(s) to {args.out_dir}")
    print(f"wrote per-cell and by-beta CSV tables to {args.out_dir / 'tables'}")


if __name__ == "__main__":
    main()
