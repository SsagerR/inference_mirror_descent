"""Aggregate 2D toy MALA sweep outputs into tables and heatmaps.

This script intentionally does not rerun sampling or recompute 2D grid metrics.
It reads each run's config.json and metrics.csv produced by toy_mala_2d.py,
then writes a sweep table, a run manifest, copied contour panels, and heatmaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    default_root = Path(os.environ.get("ZSCRATCH", ".")) / "runs" / "toy_mala"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, default=default_root)
    p.add_argument("--runs-file", type=Path, default=None)
    p.add_argument("--match", default="toy_mala_2d_*")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--expected-runs", type=int, default=None)
    p.add_argument(
        "--heatmap-metrics",
        default=(
            "sliced_w1,js,kl_sample_target,marginal_ks_x,marginal_ks_y,"
            "acceptance_rate,clip_fraction,sample_mean_q,target_mean_q"
        ),
    )
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_run_dirs(args) -> list[Path]:
    if args.runs_file is not None:
        lines = [
            line.strip()
            for line in args.runs_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        run_dirs = [Path(line) for line in lines]
    else:
        run_dirs = sorted(args.runs_root.glob(args.match))

    completed = []
    for run_dir in run_dirs:
        if (run_dir / "config.json").exists() and (run_dir / "metrics.csv").exists():
            completed.append(run_dir)
    return completed


def value_key(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def unique_values(values: list):
    seen = {}
    for value in values:
        seen.setdefault(value_key(value), value)
    return [seen[key] for key in sorted(seen)]


def split_fixed_and_swept(configs: list[dict]) -> tuple[dict, dict]:
    keys = sorted({key for cfg in configs for key in cfg})
    fixed = {}
    swept = {}
    for key in keys:
        values = [cfg.get(key) for cfg in configs]
        uniques = unique_values(values)
        if len(uniques) == 1:
            fixed[key] = uniques[0]
        else:
            swept[key] = uniques
    return fixed, swept


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def sorted_unique(values: list):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    try:
        return sorted(out)
    except TypeError:
        return out


def row_stage_key(row: dict) -> str:
    stage = row["stage"]
    if stage == "intermediate":
        return f"intermediate_t{int(float(row['timestep'])):03d}"
    return stage


def save_heatmap_grid(rows: list[dict], metric: str, stage_key: str, out_path: Path) -> bool:
    selected = [
        row
        for row in rows
        if row_stage_key(row) == stage_key and np.isfinite(as_float(row.get(metric)))
    ]
    if not selected:
        return False

    betas = sorted_unique([as_float(row["beta"]) for row in selected])
    gradients = sorted_unique([row["guidance_gradient_space"] for row in selected])
    mala_steps = sorted_unique([int(float(row["mala_steps"])) for row in selected])
    predictors = sorted_unique([row["denoising_predictor"] for row in selected])
    if not mala_steps or not predictors:
        return False

    fig, axes = plt.subplots(
        max(1, len(betas)),
        max(1, len(gradients)),
        figsize=(4.2 * max(1, len(gradients)), 3.4 * max(1, len(betas))),
        squeeze=False,
        constrained_layout=True,
    )

    finite_values = np.asarray([as_float(row[metric]) for row in selected], dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    vmin = float(np.min(finite_values)) if finite_values.size else None
    vmax = float(np.max(finite_values)) if finite_values.size else None
    image = None

    for i, beta in enumerate(betas):
        for j, gradient in enumerate(gradients):
            ax = axes[i][j]
            mat = np.full((len(mala_steps), len(predictors)), np.nan, dtype=np.float64)
            for row in selected:
                if as_float(row["beta"]) != beta or row["guidance_gradient_space"] != gradient:
                    continue
                y = mala_steps.index(int(float(row["mala_steps"])))
                x = predictors.index(row["denoising_predictor"])
                mat[y, x] = as_float(row[metric])
            image = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(predictors)), predictors, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(mala_steps)), mala_steps)
            ax.set_xlabel("denoising predictor")
            ax.set_ylabel("mala steps")
            ax.set_title(f"beta={beta:g}, gradient={gradient}")
            for y in range(len(mala_steps)):
                for x in range(len(predictors)):
                    val = mat[y, x]
                    if np.isfinite(val):
                        ax.text(x, y, f"{val:.3g}", ha="center", va="center", color="white", fontsize=8)

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label=metric)
    fig.suptitle(f"{stage_key}: {metric}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def save_all_heatmaps(rows: list[dict], out_dir: Path, metrics: list[str]) -> int:
    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    count = 0
    for stage_key in sorted_unique([row_stage_key(row) for row in rows]):
        for metric in metrics:
            if save_heatmap_grid(rows, metric, stage_key, heatmap_dir / f"{stage_key}_{metric}.png"):
                count += 1
    return count


def copy_panels(run_dirs: list[Path], configs: list[dict], out_dir: Path) -> int:
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(exist_ok=True)
    count = 0
    for run_dir, cfg in zip(run_dirs, configs):
        src = run_dir / "plots" / "contour_panel.png"
        if not src.exists():
            continue
        name = (
            f"mala{cfg.get('mala_steps')}_"
            f"{cfg.get('guidance_gradient_space')}_"
            f"{cfg.get('denoising_predictor')}_"
            f"beta{cfg.get('beta')}_"
            f"{run_dir.name}.png"
        )
        shutil.copy2(src, panel_dir / name)
        count += 1
    return count


def write_metadata(
    out_dir: Path,
    args,
    configs: list[dict],
    run_dirs: list[Path],
    heatmap_metrics: list[str],
    heatmap_count: int,
    panel_count: int,
) -> None:
    fixed, swept = split_fixed_and_swept(configs)
    metadata = {
        "num_runs": len(run_dirs),
        "runs_file": str(args.runs_file) if args.runs_file is not None else None,
        "runs_root": str(args.runs_root),
        "match": args.match,
        "expected_runs": args.expected_runs,
        "heatmap_metrics": heatmap_metrics,
        "fixed_parameters": fixed,
        "swept_parameters": swept,
        "outputs": {
            "run_manifest": str(out_dir / "run_manifest.csv"),
            "sweep_metrics": str(out_dir / "sweep_metrics.csv"),
            "panels": str(out_dir / "panels"),
            "panel_count": panel_count,
            "heatmaps": str(out_dir / "heatmaps"),
            "heatmap_count": heatmap_count,
        },
        "run_dirs": [str(path) for path in run_dirs],
    }
    (out_dir / "experiment_config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    lines = [
        "# 2D Toy MALA Summary",
        "",
        f"- Runs summarized: {len(run_dirs)}",
        f"- Runs file: `{metadata['runs_file']}`",
        f"- Sweep metrics: `sweep_metrics.csv`",
        f"- Run manifest: `run_manifest.csv`",
        f"- Panels: `panels/` ({panel_count} files)",
        f"- Heatmaps: `heatmaps/` ({heatmap_count} files)",
        "",
        "## Swept Parameters",
        "",
    ]
    if swept:
        for key, values in swept.items():
            lines.append(f"- `{key}`: `{json.dumps(values, sort_keys=True)}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Fixed Parameters", ""])
    for key, value in fixed.items():
        lines.append(f"- `{key}`: `{json.dumps(value, sort_keys=True)}`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args)
    if args.expected_runs is not None and len(run_dirs) != args.expected_runs:
        raise SystemExit(f"Expected {args.expected_runs} run dirs, found {len(run_dirs)}.")
    if not run_dirs:
        raise SystemExit("No completed 2D toy MALA run directories found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    manifest_rows = []
    configs = []
    for run_dir in run_dirs:
        cfg = load_json(run_dir / "config.json")
        configs.append(cfg)
        metrics = read_csv(run_dir / "metrics.csv")
        manifest_rows.append(
            {
                "run_dir": str(run_dir),
                "job_name": run_dir.name,
                "mala_steps": cfg.get("mala_steps"),
                "guidance_gradient_space": cfg.get("guidance_gradient_space"),
                "denoising_predictor": cfg.get("denoising_predictor"),
                "beta": cfg.get("beta"),
                "beta_schedule_type": cfg.get("beta_schedule_type"),
                "mala_adapt_rate": cfg.get("mala_adapt_rate"),
                "num_samples": cfg.get("num_samples"),
                "seed": cfg.get("seed"),
            }
        )
        for row in metrics:
            full_row = dict(row)
            full_row["run_dir"] = str(run_dir)
            for key, value in cfg.items():
                full_row.setdefault(key, value if not isinstance(value, (list, dict)) else json.dumps(value))
            all_rows.append(full_row)

    write_csv(args.out_dir / "run_manifest.csv", manifest_rows)
    write_csv(args.out_dir / "sweep_metrics.csv", all_rows)

    heatmap_metrics = [m.strip() for m in args.heatmap_metrics.replace(",", " ").split() if m.strip()]
    heatmap_count = save_all_heatmaps(all_rows, args.out_dir, heatmap_metrics)
    panel_count = copy_panels(run_dirs, configs, args.out_dir)
    write_metadata(args.out_dir, args, configs, run_dirs, heatmap_metrics, heatmap_count, panel_count)

    print(f"runs summarized: {len(run_dirs)}")
    print(f"run manifest: {args.out_dir / 'run_manifest.csv'}")
    print(f"summary table: {args.out_dir / 'sweep_metrics.csv'}")
    print(f"panel images: {args.out_dir / 'panels'} ({panel_count} files)")
    print(f"heatmap images: {args.out_dir / 'heatmaps'} ({heatmap_count} files)")


if __name__ == "__main__":
    main()
