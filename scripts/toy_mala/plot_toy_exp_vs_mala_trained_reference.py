#!/usr/bin/env python3
"""Plot method-minus-pi_orig_diffusion reference heatmaps.

This is the fair finite-data reference for the exp-vs-MALA toy experiment:
``pi_orig_diffusion`` is trained from the same pi_orig dataset and sampled
without reward guidance.  For lower-is-better divergence metrics, negative
``method - pi_orig_diffusion`` means the method improves over the trained
pi_orig diffusion reference.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = [
    "js",
    "kl_sample_target",
    "w1",
    "ks",
    "sliced_w1",
    "marginal_ks_x",
    "marginal_ks_y",
]
REFERENCE_METHOD = "pi_orig_diffusion"
METHODS = ["exponential_energy", "orig_energy_mala"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--metrics", default="auto")
    p.add_argument("--stage", default="final_clean")
    p.add_argument("--mala-steps", default="2")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def as_float(value, default=math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def beta_value(row: dict[str, str]) -> str:
    return row.get("beta_input") or row.get("beta") or ""


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


def align_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (str(row.get("sample_budget", "")), str(beta_value(row)), str(row.get("dataset_seed", "")))


def method_steps(row: dict[str, str]) -> str:
    if row.get("method") in {"exponential_energy", REFERENCE_METHOD}:
        return "independent"
    return str(row.get("mala_steps", ""))


def collect_rows(rows: list[dict[str, str]], metrics: list[str], mala_steps: str) -> list[dict]:
    reference_by_key = {align_key(row): row for row in rows if row.get("method") == REFERENCE_METHOD}
    out = []
    for row in rows:
        method = row.get("method", "")
        if method not in METHODS:
            continue
        if method == "orig_energy_mala" and str(row.get("mala_steps", "")) != str(mala_steps):
            continue
        reference = reference_by_key.get(align_key(row))
        if reference is None:
            continue
        for metric in metrics:
            value = as_float(row.get(metric))
            ref_value = as_float(reference.get(metric))
            if not (np.isfinite(value) and np.isfinite(ref_value)):
                continue
            delta = value - ref_value
            out.append(
                {
                    "metric": metric,
                    "method": method,
                    "mala_steps": method_steps(row),
                    "sample_budget": row.get("sample_budget", ""),
                    "beta": beta_value(row),
                    "dataset_seed": row.get("dataset_seed", ""),
                    "method_value": value,
                    "pi_orig_diffusion_reference": ref_value,
                    "method_minus_reference": delta,
                    "interpretation": "method_better_than_pi_orig_diffusion"
                    if delta < 0
                    else "method_worse_than_pi_orig_diffusion",
                }
            )
    return out


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["metric"], row["method"], row["mala_steps"], row["beta"], row["sample_budget"])
        groups.setdefault(key, []).append(row)
    out = []
    for (metric, method, steps, beta, budget), group in sorted(
        groups.items(),
        key=lambda kv: (kv[0][0], kv[0][1], as_float(kv[0][3]), as_float(kv[0][4]), kv[0][2]),
    ):
        values = np.asarray([float(row["method_value"]) for row in group], dtype=np.float64)
        refs = np.asarray([float(row["pi_orig_diffusion_reference"]) for row in group], dtype=np.float64)
        deltas = np.asarray([float(row["method_minus_reference"]) for row in group], dtype=np.float64)
        out.append(
            {
                "metric": metric,
                "method": method,
                "mala_steps": steps,
                "beta": beta,
                "sample_budget": budget,
                "n": len(group),
                "method_mean": float(np.mean(values)),
                "method_median": float(np.median(values)),
                "pi_orig_diffusion_reference_mean": float(np.mean(refs)),
                "pi_orig_diffusion_reference_median": float(np.median(refs)),
                "method_minus_reference_mean": float(np.mean(deltas)),
                "method_minus_reference_median": float(np.median(deltas)),
                "method_better_fraction": float(np.mean(deltas < 0)),
                "interpretation": "method_better_than_pi_orig_diffusion"
                if float(np.mean(deltas)) < 0
                else "method_worse_than_pi_orig_diffusion",
            }
        )
    return out


def matrix_from_cells(rows: list[dict], metric: str, method: str, steps: str, betas: list[str], budgets: list[str]) -> np.ndarray:
    mat = np.full((len(betas), len(budgets)), np.nan, dtype=np.float64)
    for i, beta in enumerate(betas):
        for j, budget in enumerate(budgets):
            vals = [
                as_float(row.get("method_minus_reference_mean"))
                for row in rows
                if row.get("metric") == metric
                and row.get("method") == method
                and row.get("mala_steps") == steps
                and str(row.get("beta")) == str(beta)
                and str(row.get("sample_budget")) == str(budget)
            ]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                mat[i, j] = float(np.mean(vals))
    return mat


def finite_symmetric_limit(mats: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([mat[np.isfinite(mat)] for mat in mats if np.isfinite(mat).any()]) if any(np.isfinite(mat).any() for mat in mats) else np.asarray([1.0])
    lim = float(np.max(np.abs(vals)))
    return (-lim if lim > 0 else -1.0, lim if lim > 0 else 1.0)


def add_cell_text(ax, mat: np.ndarray) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=7, color="black")


def plot_heatmaps(cell_rows: list[dict], metrics: list[str], out_dir: Path, dpi: int, mala_steps: str) -> int:
    betas = numeric_sorted_unique(row.get("beta") for row in cell_rows)
    budgets = numeric_sorted_unique(row.get("sample_budget") for row in cell_rows)
    panels = [("exponential_energy", "independent"), ("orig_energy_mala", str(mala_steps))]
    count = 0
    for metric in metrics:
        mats = [matrix_from_cells(cell_rows, metric, method, steps, betas, budgets) for method, steps in panels]
        if not any(np.isfinite(mat).any() for mat in mats):
            continue
        vmin, vmax = finite_symmetric_limit(mats)
        fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.5), squeeze=False)
        for ax, (method, steps), mat in zip(axes.ravel(), panels, mats):
            im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
            add_cell_text(ax, mat)
            ax.set_title(method if method == "exponential_energy" else f"{method}\nmala_steps={steps}", fontsize=9)
            ax.set_xticks(range(len(budgets)), budgets, rotation=30, ha="right")
            ax.set_yticks(range(len(betas)), betas)
            ax.set_xlabel("sample_budget")
            ax.set_ylabel("beta")
            fig.colorbar(im, ax=ax, shrink=0.82)
        fig.suptitle(f"{metric}: method - pi_orig_diffusion; negative is better", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path = out_dir / "heatmaps" / "method_minus_pi_orig_diffusion" / f"{metric}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    rows = [row for row in read_rows(args.metrics_file) if row.get("stage") == args.stage]
    if not rows:
        raise SystemExit(f"no rows found for stage={args.stage!r}")
    metrics = metric_list(rows, args.metrics)
    if not metrics:
        raise SystemExit("no requested metrics were present in the CSV")
    if not any(row.get("method") == REFERENCE_METHOD for row in rows):
        raise SystemExit("metrics file does not contain pi_orig_diffusion rows")

    raw_rows = collect_rows(rows, metrics, args.mala_steps)
    cell_rows = aggregate_rows(raw_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "method_minus_pi_orig_diffusion_rows.csv", raw_rows)
    write_rows(args.out_dir / "method_minus_pi_orig_diffusion_cells.csv", cell_rows)

    plot_count = 0
    if not args.no_plots:
        plot_count = plot_heatmaps(cell_rows, metrics, args.out_dir, args.dpi, args.mala_steps)

    print(f"wrote {len(cell_rows)} method-minus-pi_orig_diffusion aggregate rows")
    if not args.no_plots:
        print(f"wrote {plot_count} trained-reference heatmap image(s) to {args.out_dir / 'heatmaps'}")


if __name__ == "__main__":
    main()
