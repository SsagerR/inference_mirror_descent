#!/usr/bin/env python3
"""Render toy exponential-vs-MALA panels from saved samples with fixed axes."""

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


METHODS = ["exponential_energy", "orig_energy_mala"]
METHOD_LABELS = {
    "exponential_energy": "Exponential energy diffusion",
    "orig_energy_mala": "Orig energy diffusion + MALA",
}


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


def panel_npz_name(method: str) -> str:
    return f"{method}.npz"


def read_panel_rows(runs_file: Path) -> list[dict]:
    rows = []
    for run_dir in read_run_dirs(runs_file):
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        for method in METHODS:
            npz_path = run_dir / "panel_data" / panel_npz_name(method)
            if not npz_path.exists():
                continue
            with np.load(npz_path) as data:
                row = {
                    "run_dir": str(run_dir),
                    "method": method,
                    "dim": normalize_value(cfg.get("dim")),
                    "sample_budget": normalize_value(cfg.get("sample_budget")),
                    "beta": normalize_value(cfg.get("beta")),
                    "mala_steps": normalize_value(cfg.get("mala_steps")),
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
    mala_values = [row["mala_steps"] for row in rows if row["method"] == "orig_energy_mala"]
    if mala_values:
        return sorted(set(mala_values), key=numeric_sort_key)
    return sorted(set(row["mala_steps"] for row in rows), key=numeric_sort_key)


def axis_limits_1d(rows: list[dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    grids = [np.asarray(row["grid"], dtype=np.float64) for row in rows if "grid" in row]
    targets = [np.asarray(row["target"], dtype=np.float64) for row in rows]
    if not grids or not targets:
        return (-1.0, 1.0), (0.0, 1.0)
    x_low = min(float(np.min(grid)) for grid in grids)
    x_high = max(float(np.max(grid)) for grid in grids)
    target_max = max(float(np.nanmax(target)) for target in targets)
    y_high = max(target_max * 1.12, 1e-6)
    return (x_low, x_high), (0.0, y_high)


def axis_limits_2d(rows: list[dict]) -> tuple[tuple[float, float], tuple[float, float], np.ndarray]:
    grid_xs = [np.asarray(row["grid_x"], dtype=np.float64) for row in rows if "grid_x" in row]
    grid_ys = [np.asarray(row["grid_y"], dtype=np.float64) for row in rows if "grid_y" in row]
    targets = [np.asarray(row["target"], dtype=np.float64) for row in rows]
    if not grid_xs or not grid_ys or not targets:
        return (-1.0, 1.0), (-1.0, 1.0), np.linspace(0.0, 1.0, 12)
    xlim = (min(float(np.min(g)) for g in grid_xs), max(float(np.max(g)) for g in grid_xs))
    ylim = (min(float(np.min(g)) for g in grid_ys), max(float(np.max(g)) for g in grid_ys))
    target_max = max(float(np.nanmax(t)) for t in targets)
    if target_max <= 0.0 or not math.isfinite(target_max):
        levels = np.linspace(0.0, 1.0, 12)
    else:
        levels = np.linspace(target_max / 12.0, target_max, 12)
    return xlim, ylim, levels


def subset_rows(
    rows: list[dict],
    *,
    beta: str,
    sample_budget: str,
    mala_steps: str,
    dataset_seed: str,
    method: str,
) -> dict | None:
    for row in rows:
        if (
            row["beta"] == beta
            and row["sample_budget"] == sample_budget
            and row["dataset_seed"] == dataset_seed
            and row["method"] == method
            and (row["mala_steps"] == mala_steps or (method == "exponential_energy" and row["mala_steps"] == "0"))
        ):
            return row
    return None


def draw_1d(ax, row: dict, xlim: tuple[float, float], ylim: tuple[float, float], bins: int) -> None:
    ax.hist(row["samples"], bins=np.linspace(xlim[0], xlim[1], bins + 1), density=True, alpha=0.38, color="tab:blue")
    ax.plot(row["grid"], row["target"], lw=1.4, color="tab:orange")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def draw_2d(
    ax,
    row: dict,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    levels: np.ndarray,
    max_points: int,
) -> None:
    samples = row["samples"]
    if len(samples) > max_points:
        rng = np.random.default_rng(0)
        samples = samples[rng.choice(len(samples), size=max_points, replace=False)]
    ax.contour(row["grid_x"], row["grid_y"], row["target"], levels=levels, colors="tab:orange", linewidths=0.9)
    ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.20, color="tab:blue", linewidths=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")


def render_one_beta_panel(
    rows: list[dict],
    *,
    beta: str,
    out_path: Path,
    dpi: int,
    methods: list[str] | None = None,
    sample_budgets: list[str] | None = None,
    mala_steps_values: list[str] | None = None,
    dataset_seeds: list[str] | None = None,
    hist_bins: int = 100,
    max_points: int = 1800,
) -> bool:
    beta_rows = [row for row in rows if row["beta"] == beta]
    if not beta_rows:
        return False
    dim = beta_rows[0]["dim"]
    methods = methods or METHODS
    sample_budgets = sample_budgets or parse_value_filter("auto", [row["sample_budget"] for row in beta_rows])
    mala_steps_values = mala_steps_values or comparison_mala_steps(beta_rows, "auto")
    dataset_seeds = dataset_seeds or parse_value_filter("auto", [row["dataset_seed"] for row in beta_rows])

    if dim == "1d":
        xlim, ylim = axis_limits_1d(beta_rows)
    else:
        xlim, ylim, levels = axis_limits_2d(beta_rows)

    row_specs = [
        (mala_steps, dataset_seed, method)
        for mala_steps in mala_steps_values
        for dataset_seed in dataset_seeds
        for method in methods
    ]
    n_rows = max(len(row_specs), 1)
    n_cols = max(len(sample_budgets), 1)
    fig_width = max(8.0, 3.1 * n_cols)
    fig_height = max(4.5, 1.9 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    wrote_any = False
    for r, (mala_steps, dataset_seed, method) in enumerate(row_specs):
        for c, sample_budget in enumerate(sample_budgets):
            ax = axes[r, c]
            row = subset_rows(
                beta_rows,
                beta=beta,
                sample_budget=sample_budget,
                mala_steps=mala_steps,
                dataset_seed=dataset_seed,
                method=method,
            )
            if row is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_facecolor("#f2f2f2")
            else:
                if dim == "1d":
                    draw_1d(ax, row, xlim, ylim, hist_bins)
                else:
                    draw_2d(ax, row, xlim, ylim, levels, max_points)
                wrote_any = True
            if r == 0:
                ax.set_title(f"N={sample_budget}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"K={mala_steps}, seed={dataset_seed}\n{METHOD_LABELS.get(method, method)}", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"fixed-axis samples vs target | beta={beta}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.45, w_pad=0.25)
    if wrote_any:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return wrote_any


def safe_beta(beta: str) -> str:
    return beta.replace(".", "p").replace("-", "m")


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_dir", "dim", "method", "sample_budget", "beta", "mala_steps", "dataset_seed"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--sample-budgets", default="auto")
    p.add_argument("--betas", default="auto")
    p.add_argument("--mala-steps", default="auto")
    p.add_argument("--dataset-seeds", default="auto")
    p.add_argument("--hist-bins", type=int, default=100)
    p.add_argument("--max-points", type=int, default=1800)
    p.add_argument("--dpi", type=int, default=170)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_panel_rows(args.runs_file)
    if not rows:
        raise SystemExit(
            "no saved panel_data/*.npz files found. Re-run toy_mala_exponential_vs_mala.py "
            "after the panel-data patch, then rerun this script."
        )

    methods = [m.strip() for m in args.methods.replace(",", " ").split() if m.strip()]
    sample_budgets = parse_value_filter(args.sample_budgets, [row["sample_budget"] for row in rows])
    betas = parse_value_filter(args.betas, [row["beta"] for row in rows])
    mala_steps_values = comparison_mala_steps(rows, args.mala_steps)
    dataset_seeds = parse_value_filter(args.dataset_seeds, [row["dataset_seed"] for row in rows])

    write_manifest(args.out_dir / "fixed_panel_manifest.csv", rows)
    count = 0
    for beta in betas:
        if render_one_beta_panel(
            rows,
            beta=beta,
            out_path=args.out_dir / "by_beta" / f"beta_{safe_beta(beta)}.png",
            dpi=args.dpi,
            methods=methods,
            sample_budgets=sample_budgets,
            mala_steps_values=mala_steps_values,
            dataset_seeds=dataset_seeds,
            hist_bins=args.hist_bins,
            max_points=args.max_points,
        ):
            count += 1

    print(f"scanned {len(rows)} panel-data rows")
    print(f"wrote {count} fixed-axis panel image(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
