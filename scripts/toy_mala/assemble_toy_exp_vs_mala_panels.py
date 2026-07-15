#!/usr/bin/env python3
"""Assemble per-run toy exponential-vs-MALA sample/target plots into comparison panels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


METHODS = ["exponential_energy", "orig_energy_mala"]
METHOD_LABELS = {
    "exponential_energy": "Exponential energy diffusion",
    "orig_energy_mala": "Orig energy diffusion + MALA",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--sample-budgets", default="auto")
    p.add_argument("--betas", default="auto")
    p.add_argument("--mala-steps", default="auto")
    p.add_argument("--dataset-seeds", default="auto")
    p.add_argument(
        "--layout",
        choices=["one_beta", "beta_mala_steps_seed"],
        default="one_beta",
        help=(
            "one_beta writes one large figure per beta with rows=mala_steps x seed x method; "
            "beta_mala_steps_seed writes one smaller figure per beta/mala_steps/seed."
        ),
    )
    p.add_argument("--dpi", type=int, default=170)
    return p.parse_args()


def method_plot_name(method: str) -> str:
    if method == "exponential_energy":
        return "final_clean_exponential.png"
    if method == "orig_energy_mala":
        return "final_clean_mala.png"
    raise ValueError(f"unknown method {method!r}")


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


def read_runs_file(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def scan_run_dirs(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        available_methods = []
        plot_paths = {}
        for method in METHODS:
            plot_path = run_dir / "plots" / method_plot_name(method)
            if plot_path.exists():
                available_methods.append(method)
                plot_paths[method] = plot_path
        rows.append({
            "run_dir": str(run_dir),
            "dim": normalize_value(cfg.get("dim")),
            "sample_budget": normalize_value(cfg.get("sample_budget")),
            "beta": normalize_value(cfg.get("beta")),
            "mala_steps": normalize_value(cfg.get("mala_steps")),
            "dataset_seed": normalize_value(cfg.get("dataset_seed")),
            "available_methods": available_methods,
            "plot_paths": plot_paths,
        })
    return rows


def row_key(row: dict) -> tuple[str, str, str]:
    return (row["beta"], row["mala_steps"], row["dataset_seed"])


def index_rows(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    out = {}
    for row in rows:
        out[(row["sample_budget"], row["beta"], row["mala_steps"], row["dataset_seed"])] = row
    return out


def comparison_mala_steps(rows: list[dict], spec: str) -> list[str]:
    if spec.strip().lower() != "auto":
        return parse_value_filter(spec, [row["mala_steps"] for row in rows])
    mala_values = [
        row["mala_steps"]
        for row in rows
        if "orig_energy_mala" in row["available_methods"]
    ]
    if mala_values:
        return sorted(set(mala_values), key=numeric_sort_key)
    return parse_value_filter(spec, [row["mala_steps"] for row in rows])


def lookup_panel_row(
    rows_by_budget: dict[tuple[str, str, str, str], dict],
    *,
    budget: str,
    beta: str,
    mala_steps: str,
    dataset_seed: str,
    method: str,
) -> dict | None:
    row = rows_by_budget.get((budget, beta, mala_steps, dataset_seed))
    if row is not None and method in row["plot_paths"]:
        return row
    if method == "exponential_energy":
        row = rows_by_budget.get((budget, beta, "0", dataset_seed))
        if row is not None and method in row["plot_paths"]:
            return row
    return None


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_dir",
                "dim",
                "sample_budget",
                "beta",
                "mala_steps",
                "dataset_seed",
                "has_exponential_energy_plot",
                "has_orig_energy_mala_plot",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "run_dir": row["run_dir"],
                "dim": row["dim"],
                "sample_budget": row["sample_budget"],
                "beta": row["beta"],
                "mala_steps": row["mala_steps"],
                "dataset_seed": row["dataset_seed"],
                "has_exponential_energy_plot": int("exponential_energy" in row["available_methods"]),
                "has_orig_energy_mala_plot": int("orig_energy_mala" in row["available_methods"]),
            })


def panel_filename(beta: str, mala_steps: str, dataset_seed: str) -> str:
    safe_beta = beta.replace(".", "p").replace("-", "m")
    return f"beta_{safe_beta}_mala_steps_{mala_steps}_seed_{dataset_seed}.png"


def beta_panel_filename(beta: str) -> str:
    safe_beta = beta.replace(".", "p").replace("-", "m")
    return f"beta_{safe_beta}.png"


def assemble_beta_mala_steps_seed_panel(
    *,
    rows_by_budget: dict[tuple[str, str, str, str], dict],
    methods: list[str],
    sample_budgets: list[str],
    beta: str,
    mala_steps: str,
    dataset_seed: str,
    out_path: Path,
    dpi: int,
) -> bool:
    fig, axes = plt.subplots(
        len(methods),
        len(sample_budgets),
        figsize=(4.2 * len(sample_budgets), 3.4 * len(methods)),
        squeeze=False,
    )
    wrote_any = False
    for r, method in enumerate(methods):
        for c, budget in enumerate(sample_budgets):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            row = lookup_panel_row(
                rows_by_budget,
                budget=budget,
                beta=beta,
                mala_steps=mala_steps,
                dataset_seed=dataset_seed,
                method=method,
            )
            if row is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            else:
                image = mpimg.imread(row["plot_paths"][method])
                ax.imshow(image)
                wrote_any = True
            if r == 0:
                ax.set_title(f"N={budget}", fontsize=11)
            if c == 0:
                ax.set_ylabel(METHOD_LABELS.get(method, method), fontsize=11)
    fig.suptitle(
        f"samples vs target | beta={beta}, mala_steps={mala_steps}, dataset_seed={dataset_seed}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if wrote_any:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return wrote_any


def assemble_one_beta_panel(
    *,
    rows_by_budget: dict[tuple[str, str, str, str], dict],
    methods: list[str],
    sample_budgets: list[str],
    beta: str,
    mala_steps_values: list[str],
    dataset_seeds: list[str],
    out_path: Path,
    dpi: int,
) -> bool:
    row_specs = [
        (mala_steps, dataset_seed, method)
        for mala_steps in mala_steps_values
        for dataset_seed in dataset_seeds
        for method in methods
    ]
    n_rows = len(row_specs)
    n_cols = len(sample_budgets)
    fig_width = max(8.0, 3.55 * n_cols)
    fig_height = max(5.0, 2.15 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    wrote_any = False
    for r, (mala_steps, dataset_seed, method) in enumerate(row_specs):
        for c, budget in enumerate(sample_budgets):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            row = lookup_panel_row(
                rows_by_budget,
                budget=budget,
                beta=beta,
                mala_steps=mala_steps,
                dataset_seed=dataset_seed,
                method=method,
            )
            if row is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_facecolor("#f2f2f2")
            else:
                image = mpimg.imread(row["plot_paths"][method])
                ax.imshow(image, interpolation="nearest")
                wrote_any = True
            if r == 0:
                ax.set_title(f"N={budget}", fontsize=10)
            if c == 0:
                label = f"K={mala_steps}, seed={dataset_seed}\n{METHOD_LABELS.get(method, method)}"
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(
        f"samples vs target | beta={beta}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.45, w_pad=0.25)
    if wrote_any:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return wrote_any


def main() -> None:
    args = parse_args()
    run_dirs = read_runs_file(args.runs_file)
    rows = scan_run_dirs(run_dirs)
    if not rows:
        raise SystemExit(f"no usable run dirs found in {args.runs_file}")

    methods = [m.strip() for m in args.methods.replace(",", " ").split() if m.strip()]
    sample_budgets = parse_value_filter(args.sample_budgets, [row["sample_budget"] for row in rows])
    betas = parse_value_filter(args.betas, [row["beta"] for row in rows])
    mala_steps_values = comparison_mala_steps(rows, args.mala_steps)
    dataset_seeds = parse_value_filter(args.dataset_seeds, [row["dataset_seed"] for row in rows])
    rows_by_budget = index_rows(rows)

    write_manifest(args.out_dir / "panel_manifest.csv", rows)

    count = 0
    for beta in betas:
        if args.layout == "one_beta":
            out_path = args.out_dir / "by_beta" / beta_panel_filename(beta)
            if assemble_one_beta_panel(
                rows_by_budget=rows_by_budget,
                methods=methods,
                sample_budgets=sample_budgets,
                beta=beta,
                mala_steps_values=mala_steps_values,
                dataset_seeds=dataset_seeds,
                out_path=out_path,
                dpi=args.dpi,
            ):
                count += 1
        else:
            for mala_steps in mala_steps_values:
                for dataset_seed in dataset_seeds:
                    out_path = args.out_dir / f"mala_steps_{mala_steps}" / f"seed_{dataset_seed}" / panel_filename(beta, mala_steps, dataset_seed)
                    if assemble_beta_mala_steps_seed_panel(
                        rows_by_budget=rows_by_budget,
                        methods=methods,
                        sample_budgets=sample_budgets,
                        beta=beta,
                        mala_steps=mala_steps,
                        dataset_seed=dataset_seed,
                        out_path=out_path,
                        dpi=args.dpi,
                    ):
                        count += 1
    print(f"scanned {len(rows)} run dirs")
    print(f"wrote {count} panel image(s) to {args.out_dir}")
    print(f"wrote manifest to {args.out_dir / 'panel_manifest.csv'}")


if __name__ == "__main__":
    main()
