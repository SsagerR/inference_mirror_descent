#!/usr/bin/env python3
"""Render fixed-axis panels for RSM-vs-MALA-distillation final samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["pi_orig_distill", "rsm", "mala_distill"]
METHOD_LABELS = {
    "pi_orig_distill": "pi_orig distill",
    "rsm": "RSM weighted DSM",
    "mala_distill": "MALA samples -> DSM",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--compute-budgets", default="auto")
    p.add_argument("--betas", default="auto")
    p.add_argument("--mala-steps", default="auto")
    p.add_argument("--dataset-seeds", default="auto")
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--hist-bins", type=int, default=100)
    p.add_argument("--max-points", type=int, default=1800)
    return p.parse_args()


def as_float(value, default=math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def numeric_sort_key(value: str) -> tuple[float, str]:
    f = as_float(value)
    return (f if math.isfinite(f) else math.inf, str(value))


def normalize_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_value_filter(spec: str, values: list[str]) -> list[str]:
    if spec.strip().lower() == "auto":
        return sorted(set(values), key=numeric_sort_key)
    requested = [s.strip() for s in spec.replace(",", " ").split() if s.strip()]
    return sorted(requested, key=numeric_sort_key)


def read_run_dirs(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def read_panel_rows(runs_file: Path) -> list[dict]:
    rows = []
    for run_dir in read_run_dirs(runs_file):
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        for method in METHODS:
            npz_path = run_dir / "panel_data" / f"{method}.npz"
            if not npz_path.exists():
                continue
            with np.load(npz_path) as data:
                row = {
                    "run_dir": str(run_dir),
                    "method": method,
                    "dim": normalize_value(cfg.get("dim")),
                    "compute_budget": normalize_value(cfg.get("compute_budget")),
                    "beta": normalize_value(cfg.get("beta")),
                    "mala_steps": normalize_value(cfg.get("mala_steps") if method == "mala_distill" else 0),
                    "dataset_seed": normalize_value(cfg.get("dataset_seed")),
                    "samples": np.asarray(data["samples"]),
                    "target": np.asarray(data["target"]),
                }
                if "grid" in data:
                    row["grid"] = np.asarray(data["grid"])
                if "grid_x" in data:
                    row["grid_x"] = np.asarray(data["grid_x"])
                if "grid_y" in data:
                    row["grid_y"] = np.asarray(data["grid_y"])
                rows.append(row)
    return rows


def comparison_mala_steps(rows: list[dict], spec: str) -> list[str]:
    if spec.strip().lower() != "auto":
        return parse_value_filter(spec, [row["mala_steps"] for row in rows])
    vals = [row["mala_steps"] for row in rows if row["method"] == "mala_distill"]
    return sorted(set(vals), key=numeric_sort_key)


def axis_limits_1d(rows: list[dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    grids = [np.asarray(row["grid"], dtype=np.float64) for row in rows if "grid" in row]
    targets = [np.asarray(row["target"], dtype=np.float64) for row in rows]
    if not grids or not targets:
        return (-1.0, 1.0), (0.0, 1.0)
    xlim = (min(float(np.min(g)) for g in grids), max(float(np.max(g)) for g in grids))
    ymax = max(float(np.nanmax(t)) for t in targets)
    return xlim, (0.0, max(1e-6, 1.12 * ymax))


def axis_limits_2d(rows: list[dict]) -> tuple[tuple[float, float], tuple[float, float], np.ndarray]:
    grid_xs = [np.asarray(row["grid_x"], dtype=np.float64) for row in rows if "grid_x" in row]
    grid_ys = [np.asarray(row["grid_y"], dtype=np.float64) for row in rows if "grid_y" in row]
    targets = [np.asarray(row["target"], dtype=np.float64) for row in rows]
    if not grid_xs or not grid_ys or not targets:
        return (-1.0, 1.0), (-1.0, 1.0), np.linspace(0.0, 1.0, 12)
    xlim = (min(float(np.min(g)) for g in grid_xs), max(float(np.max(g)) for g in grid_xs))
    ylim = (min(float(np.min(g)) for g in grid_ys), max(float(np.max(g)) for g in grid_ys))
    target_max = max(float(np.nanmax(t)) for t in targets)
    levels = np.linspace(target_max / 12.0, target_max, 12) if target_max > 0 else np.linspace(0.0, 1.0, 12)
    return xlim, ylim, levels


def lookup(rows: list[dict], *, beta: str, budget: str, mala_steps: str, dataset_seed: str, method: str) -> dict | None:
    for row in rows:
        if row["beta"] != beta or row["compute_budget"] != budget or row["dataset_seed"] != dataset_seed or row["method"] != method:
            continue
        if method == "mala_distill" and row["mala_steps"] != mala_steps:
            continue
        return row
    return None


def draw_1d(ax, row: dict, xlim: tuple[float, float], ylim: tuple[float, float], bins: int) -> None:
    ax.hist(row["samples"], bins=np.linspace(xlim[0], xlim[1], bins + 1), density=True, alpha=0.38, color="tab:blue")
    ax.plot(row["grid"], row["target"], lw=1.4, color="tab:orange")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def draw_2d(ax, row: dict, xlim: tuple[float, float], ylim: tuple[float, float], levels: np.ndarray, max_points: int) -> None:
    samples = row["samples"]
    if len(samples) > max_points:
        rng = np.random.default_rng(0)
        samples = samples[rng.choice(len(samples), size=max_points, replace=False)]
    ax.contour(row["grid_x"], row["grid_y"], row["target"], levels=levels, colors="tab:orange", linewidths=0.9)
    ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.20, color="tab:blue", linewidths=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")


def safe_beta(beta: str) -> str:
    return beta.replace(".", "p").replace("-", "m")


def render_beta(rows: list[dict], *, beta: str, out_path: Path, methods: list[str], budgets: list[str], mala_steps_values: list[str], dataset_seeds: list[str], dpi: int, hist_bins: int, max_points: int) -> bool:
    beta_rows = [row for row in rows if row["beta"] == beta]
    if not beta_rows:
        return False
    dim = beta_rows[0]["dim"]
    if dim == "1d":
        xlim, ylim = axis_limits_1d(beta_rows)
    else:
        xlim, ylim, levels = axis_limits_2d(beta_rows)

    row_specs = [(m, s, method) for m in mala_steps_values for s in dataset_seeds for method in methods]
    fig, axes = plt.subplots(
        max(1, len(row_specs)),
        max(1, len(budgets)),
        figsize=(max(8.0, 3.1 * len(budgets)), max(4.5, 1.9 * len(row_specs))),
        squeeze=False,
    )
    wrote_any = False
    for r, (mala_steps, dataset_seed, method) in enumerate(row_specs):
        for c, budget in enumerate(budgets):
            ax = axes[r, c]
            row = lookup(beta_rows, beta=beta, budget=budget, mala_steps=mala_steps, dataset_seed=dataset_seed, method=method)
            if row is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                continue
            wrote_any = True
            if dim == "1d":
                draw_1d(ax, row, xlim, ylim, hist_bins)
            else:
                draw_2d(ax, row, xlim, ylim, levels, max_points)
            if r == 0:
                ax.set_title(f"N={budget}", fontsize=9)
            if c == 0:
                label = METHOD_LABELS.get(method, method)
                ax.set_ylabel(f"m={mala_steps}, seed={dataset_seed}\n{label}", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"Final target diffusion samples vs target, beta={beta}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return wrote_any


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_dir", "dim", "method", "compute_budget", "beta", "mala_steps", "dataset_seed"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})


def main() -> None:
    args = parse_args()
    rows = read_panel_rows(args.runs_file)
    if not rows:
        raise SystemExit("no panel_data rows found; the runs were likely produced before final sample saving was added")
    methods = [m.strip() for m in args.methods.replace(",", " ").split() if m.strip()]
    budgets = parse_value_filter(args.compute_budgets, [row["compute_budget"] for row in rows])
    betas = parse_value_filter(args.betas, [row["beta"] for row in rows])
    mala_steps_values = comparison_mala_steps(rows, args.mala_steps)
    dataset_seeds = parse_value_filter(args.dataset_seeds, [row["dataset_seed"] for row in rows])
    count = 0
    for beta in betas:
        if render_beta(
            rows,
            beta=beta,
            out_path=args.out_dir / "by_beta" / f"beta_{safe_beta(beta)}.png",
            methods=methods,
            budgets=budgets,
            mala_steps_values=mala_steps_values,
            dataset_seeds=dataset_seeds,
            dpi=args.dpi,
            hist_bins=args.hist_bins,
            max_points=args.max_points,
        ):
            count += 1
    write_manifest(args.out_dir / "fixed_panel_manifest.csv", rows)
    print(f"wrote {count} beta panel figure(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
