#!/usr/bin/env python3
"""Summarize completed 1D toy MALA runs.

The script reads existing run directories produced by ``toy_mala_1d.py`` and
creates a sweep-level metrics table plus one compact density panel per run.
It does not resample, so it is safe to run after jobs finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KEY_COLUMNS = [
    "run_name",
    "run_dir",
    "mala_steps",
    "guidance_gradient_space",
    "denoising_predictor",
    "beta",
    "alpha",
    "num_samples",
    "diffusion_steps",
    "x0_hat_clip_radius",
    "seed",
]

METRIC_COLUMNS = [
    "stage",
    "timestep",
    "kl_sample_target",
    "js",
    "w1",
    "ks",
    "sample_mean_q",
    "target_mean_q",
    "sample_mean",
    "sample_std",
    "acceptance_rate",
    "clip_fraction",
]


@dataclass(frozen=True)
class NumpySchedule:
    betas: np.ndarray
    alphas: np.ndarray
    alphas_cumprod: np.ndarray
    alphas_cumprod_prev: np.ndarray
    sqrt_alphas_cumprod: np.ndarray
    sqrt_one_minus_alphas_cumprod: np.ndarray
    sqrt_recip_alphas_cumprod: np.ndarray
    sqrt_recipm1_alphas_cumprod: np.ndarray


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    default_root = Path(os.environ.get("ZSCRATCH", ".")) / "runs" / "toy_mala"
    p.add_argument("--runs-root", type=Path, default=default_root)
    p.add_argument("--runs-file", type=Path, default=None,
                   help="Optional newline-separated run directories. Overrides --match discovery.")
    p.add_argument("--match", default="toy_mala_*",
                   help="Directory glob under --runs-root when --runs-file is not used.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--overwrite-metrics", action="store_true",
                   help="Recompute metrics.csv from samples.npz before aggregating.")
    p.add_argument("--heatmap-metrics", default="w1,ks,js,acceptance_rate,clip_fraction",
                   help="Comma-separated metric names to visualize as sweep heatmaps.")
    p.add_argument("--expected-runs", type=int, default=None,
                   help="Fail if the number of discovered completed run directories differs.")
    return p.parse_args()


def load_cfg(path: Path):
    data = json.loads(path.read_text())
    return SimpleNamespace(**data)


def discover_runs(args) -> list[Path]:
    if args.runs_file is not None:
        runs = [Path(line.strip()) for line in args.runs_file.read_text().splitlines() if line.strip()]
    else:
        runs = sorted(p for p in args.runs_root.glob(args.match) if p.is_dir())
    return [p for p in runs if (p / "config.json").exists() and (p / "samples.npz").exists()]


def read_metrics(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_metrics(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in METRIC_COLUMNS})


def cosine_beta_schedule(timesteps: int):
    s = 0.008
    t = np.arange(0, timesteps + 1) / timesteps
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod /= alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.999):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


def constant_kl_beta_schedule(timesteps: int, snr_max=1000.0):
    r = 1.0 + snr_max
    k = np.arange(timesteps, dtype=np.float64)
    exponent = (timesteps - k) / timesteps
    alphas_cumprod = 1.0 - r ** (-exponent)
    alphas_cumprod_with_1 = np.concatenate([[1.0], alphas_cumprod])
    betas = 1.0 - alphas_cumprod_with_1[1:] / alphas_cumprod_with_1[:-1]
    return np.clip(betas, 1e-8, 0.999)


def build_beta_schedule_np(num_timesteps: int, beta_schedule_type: str, snr_max: float) -> NumpySchedule:
    target_abar_0 = snr_max / (1.0 + snr_max)
    if beta_schedule_type == "constant_kl":
        betas = constant_kl_beta_schedule(num_timesteps, snr_max=snr_max)
    elif beta_schedule_type == "cosine":
        raw_betas = cosine_beta_schedule(num_timesteps)
        scale = (1.0 - target_abar_0) / raw_betas[0]
        betas = np.clip(scale * raw_betas, 0, 0.999)
    elif beta_schedule_type == "linear":
        raw_betas = linear_beta_schedule(num_timesteps)
        scale = (1.0 - target_abar_0) / raw_betas[0]
        betas = np.clip(scale * raw_betas, 0, 0.999)
    else:
        raise ValueError(f"Unknown beta_schedule_type: {beta_schedule_type}")

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    return NumpySchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=np.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=np.sqrt(1.0 - alphas_cumprod),
        sqrt_recip_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod),
        sqrt_recipm1_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod - 1.0),
    )


def eval_timesteps(spec: str, timesteps: int) -> list[int]:
    if spec == "all":
        return list(range(timesteps))
    if spec == "auto":
        vals = [0, timesteps // 4, timesteps // 2, (3 * timesteps) // 4, timesteps - 1]
    else:
        vals = [int(x) for x in spec.replace(",", " ").split()]
    return sorted(set(v for v in vals if 0 <= v < timesteps))


def np_logsumexp(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def np_base_logpdf(x, weights, means, stds, schedule, t_idx: int | None):
    x = np.asarray(x, dtype=np.float64)
    if t_idx is None:
        loc = means
        var = stds ** 2
    else:
        sqrt_ab = float(schedule.sqrt_alphas_cumprod[t_idx])
        sigma = float(schedule.sqrt_one_minus_alphas_cumprod[t_idx])
        loc = sqrt_ab * means
        var = (sqrt_ab ** 2) * (stds ** 2) + sigma ** 2
    log_comp = (
        np.log(weights)[None, :]
        - 0.5 * (np.log(2.0 * np.pi * var)[None, :] + (x[:, None] - loc[None, :]) ** 2 / var[None, :])
    )
    return np_logsumexp(log_comp, axis=-1)


def np_base_score(x, weights, means, stds, schedule, t_idx: int):
    x = np.asarray(x, dtype=np.float64)
    sqrt_ab = float(schedule.sqrt_alphas_cumprod[t_idx])
    sigma = float(schedule.sqrt_one_minus_alphas_cumprod[t_idx])
    loc = sqrt_ab * means
    var = (sqrt_ab ** 2) * (stds ** 2) + sigma ** 2
    log_comp = (
        np.log(weights)[None, :]
        - 0.5 * (np.log(2.0 * np.pi * var)[None, :] + (x[:, None] - loc[None, :]) ** 2 / var[None, :])
    )
    resp = np.exp(log_comp - np_logsumexp(log_comp, axis=-1)[:, None])
    return np.sum(resp * (-(x[:, None] - loc[None, :]) / var[None, :]), axis=-1)


def np_reward(a, center, scale):
    return -scale * (np.asarray(a) - center) ** 2


def effective_beta(cfg) -> float:
    beta = float(cfg.beta)
    if getattr(cfg, "advantage_normalization", False):
        beta /= float(np.sqrt(max(float(cfg.initial_advantage_second_moment_ema), 1e-6)))
    return beta


def np_x0_hat(x, weights, means, stds, schedule, t_idx: int):
    x = np.asarray(x, dtype=np.float64)
    sqrt_ab = float(schedule.sqrt_alphas_cumprod[t_idx])
    sigma = float(schedule.sqrt_one_minus_alphas_cumprod[t_idx])
    loc = sqrt_ab * means
    var = (sqrt_ab ** 2) * (stds ** 2) + sigma ** 2
    log_comp = (
        np.log(weights)[None, :]
        - 0.5 * (np.log(2.0 * np.pi * var)[None, :] + (x[:, None] - loc[None, :]) ** 2 / var[None, :])
    )
    resp = np.exp(log_comp - np_logsumexp(log_comp, axis=-1)[:, None])
    posterior_mean = means[None, :] + (sqrt_ab * stds[None, :] ** 2 / var[None, :]) * (x[:, None] - loc[None, :])
    return np.sum(resp * posterior_mean, axis=-1)


def normalized_density_on_grid(log_unnorm, grid):
    shifted = np.asarray(log_unnorm, dtype=np.float64) - np.max(log_unnorm)
    dens = np.exp(shifted)
    z = integrate_trapezoid(dens, grid)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Invalid density normalization constant.")
    return dens / z


def target_density(grid, cfg, weights, means, stds, schedule, t_idx: int | None):
    if t_idx is None:
        base_log = np_base_logpdf(grid, weights, means, stds, schedule, None)
        q_arg = grid
    else:
        base_log = np_base_logpdf(grid, weights, means, stds, schedule, t_idx)
        x0_hat = np_x0_hat(grid, weights, means, stds, schedule, t_idx)
        r = cfg.x0_hat_clip_radius
        q_arg = np.clip(x0_hat, -r, r) if np.isfinite(r) else x0_hat
    log_unnorm = cfg.alpha * base_log + effective_beta(cfg) * np_reward(q_arg, cfg.reward_center, cfg.reward_scale)
    return normalized_density_on_grid(log_unnorm, grid)


def metrics_from_samples(samples, grid, target_dens, cfg, *, q_sample_arg=None, q_grid_arg=None):
    samples = np.asarray(samples, dtype=np.float64)
    edges = np.linspace(grid[0], grid[-1], len(grid))
    counts, edges = np.histogram(samples, bins=edges)
    p = counts.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    mids = 0.5 * (edges[:-1] + edges[1:])
    q_dens = np.interp(mids, grid, target_dens)
    q = q_dens * np.diff(edges)
    q = q / q.sum()
    eps = 1e-12
    kl_sample_target = float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))
    m = 0.5 * (p + q)
    js = float(0.5 * np.sum(p * (np.log(p + eps) - np.log(m + eps))) +
               0.5 * np.sum(q * (np.log(q + eps) - np.log(m + eps))))

    target_cdf = np.cumsum(target_dens)
    target_cdf = target_cdf / target_cdf[-1]
    emp_counts, _ = np.histogram(samples, bins=edges)
    emp_cdf = np.cumsum(emp_counts.astype(np.float64))
    emp_cdf = emp_cdf / max(emp_cdf[-1], 1.0)
    target_cdf_mid = np.interp(mids, grid, target_cdf)
    ks = float(np.max(np.abs(emp_cdf - target_cdf_mid)))
    w1 = float(integrate_trapezoid(np.abs(emp_cdf - target_cdf_mid), mids))

    if q_sample_arg is None:
        q_sample_arg = samples
    if q_grid_arg is None:
        q_grid_arg = grid
    q_sample = float(np.mean(np_reward(q_sample_arg, cfg.reward_center, cfg.reward_scale)))
    q_target = float(integrate_trapezoid(np_reward(q_grid_arg, cfg.reward_center, cfg.reward_scale) * target_dens, grid))
    return {
        "kl_sample_target": kl_sample_target,
        "js": js,
        "w1": w1,
        "ks": ks,
        "sample_mean_q": q_sample,
        "target_mean_q": q_target,
        "sample_mean": float(np.mean(samples)),
        "sample_std": float(np.std(samples)),
    }


def save_density_panel(path, panels, title):
    n = len(panels)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, panel in zip(axes.ravel(), panels):
        ax.hist(panel["samples"], bins=80, density=True, alpha=0.38, label="samples")
        ax.plot(panel["grid"], panel["target_dens"], lw=1.8, label="target")
        ax.set_title(
            f"{panel['label']}\nW1={panel['w1']:.4f}, KS={panel['ks']:.4f}, "
            f"JS={panel['js']:.4f}, acc={panel['acc']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("density")
    axes.ravel()[0].legend(loc="best", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def get_cfg_attr(cfg, name, default):
    return getattr(cfg, name, default)


def make_eval_grid(samples, cfg, weights, means, stds, schedule, t_idx: int | None):
    grid_mode = get_cfg_attr(cfg, "grid_mode", "auto")
    grid_min = float(get_cfg_attr(cfg, "grid_min", -6.0))
    grid_max = float(get_cfg_attr(cfg, "grid_max", 6.0))
    grid_points = int(get_cfg_attr(cfg, "grid_points", 2001))
    if grid_mode == "fixed":
        return np.linspace(grid_min, grid_max, grid_points, dtype=np.float64)

    samples = np.asarray(samples, dtype=np.float64)
    if t_idx is None:
        loc = means
        sd = stds
    else:
        sqrt_ab = float(schedule.sqrt_alphas_cumprod[t_idx])
        sigma = float(schedule.sqrt_one_minus_alphas_cumprod[t_idx])
        loc = sqrt_ab * means
        sd = np.sqrt((sqrt_ab ** 2) * (stds ** 2) + sigma ** 2)

    base_low = float(np.min(loc - 6.0 * sd))
    base_high = float(np.max(loc + 6.0 * sd))
    sample_low, sample_high = np.quantile(samples, [0.001, 0.999])
    low = min(base_low, float(sample_low), grid_min)
    high = max(base_high, float(sample_high), grid_max)
    margin = max(0.05 * (high - low), 1e-3)
    return np.linspace(low - margin, high + margin, grid_points, dtype=np.float64)


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def sorted_unique(values):
    vals = sorted(set(values), key=lambda x: (str(type(x)), x))
    return vals


def row_stage_key(row):
    stage = row["stage"]
    if stage == "intermediate":
        return f"intermediate_t{int(float(row['timestep'])):03d}"
    return stage


def save_heatmap_grid(rows: list[dict], metric: str, stage_key: str, out_path: Path) -> bool:
    selected = [
        row for row in rows
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

    nrows = max(1, len(betas))
    ncols = max(1, len(gradients))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    all_values = np.asarray([as_float(row[metric]) for row in selected], dtype=np.float64)
    finite_values = all_values[np.isfinite(all_values)]
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
    stage_keys = sorted_unique([row_stage_key(row) for row in rows])
    count = 0
    for stage_key in stage_keys:
        for metric in metrics:
            path = heatmap_dir / f"{stage_key}_{metric}.png"
            if save_heatmap_grid(rows, metric, stage_key, path):
                count += 1
    return count


def recompute_metrics_and_panels(run_dir: Path, cfg: ToyConfig) -> list[dict]:
    weights = np.asarray(cfg.gmm_weights, dtype=np.float64)
    means = np.asarray(cfg.gmm_means, dtype=np.float64)
    stds = np.asarray(cfg.gmm_stds, dtype=np.float64)
    schedule = build_beta_schedule_np(cfg.diffusion_steps, cfg.beta_schedule_type, cfg.snr_max)
    samples = np.load(run_dir / "samples.npz")
    raw_x0 = np.asarray(samples["raw_x0"])
    action_clipped = np.asarray(samples["action_clipped"])
    trace = np.asarray(samples["trace"])
    eval_ts = (
        np.asarray(samples["eval_timesteps"], dtype=np.int32).tolist()
        if "eval_timesteps" in samples
        else eval_timesteps(cfg.eval_timesteps, cfg.diffusion_steps)
    )
    per_level_acc = np.asarray(samples["per_level_acc"])
    per_level_clip = np.asarray(samples["per_level_clip"])

    rows = []
    panels = []
    grid = make_eval_grid(raw_x0, cfg, weights, means, stds, schedule, None)
    final_target = target_density(grid, cfg, weights, means, stds, schedule, None)
    final_metrics = metrics_from_samples(raw_x0, grid, final_target, cfg)
    final_metrics.update({
        "stage": "final_clean",
        "timestep": -1,
        "acceptance_rate": float(per_level_acc[0]),
        "clip_fraction": float(per_level_clip[0]),
    })
    rows.append(final_metrics)
    panels.append({
        "label": "final clean",
        "samples": raw_x0,
        "grid": grid,
        "target_dens": final_target,
        "w1": final_metrics["w1"],
        "ks": final_metrics["ks"],
        "js": final_metrics["js"],
        "acc": final_metrics["acceptance_rate"],
    })

    clipped_metrics = metrics_from_samples(action_clipped, grid, final_target, cfg)
    clipped_metrics.update({
        "stage": "final_clipped_action",
        "timestep": -1,
        "acceptance_rate": float("nan"),
        "clip_fraction": float(np.mean(np.abs(raw_x0) > 1.0)),
    })
    rows.append(clipped_metrics)

    for t in eval_ts:
        grid = make_eval_grid(trace[t], cfg, weights, means, stds, schedule, t)
        dens = target_density(grid, cfg, weights, means, stds, schedule, t)
        sample_x0_hat = np_x0_hat(trace[t], weights, means, stds, schedule, t)
        grid_x0_hat = np_x0_hat(grid, weights, means, stds, schedule, t)
        if np.isfinite(cfg.x0_hat_clip_radius):
            sample_x0_hat = np.clip(sample_x0_hat, -cfg.x0_hat_clip_radius, cfg.x0_hat_clip_radius)
            grid_x0_hat = np.clip(grid_x0_hat, -cfg.x0_hat_clip_radius, cfg.x0_hat_clip_radius)
        metrics = metrics_from_samples(
            trace[t],
            grid,
            dens,
            cfg,
            q_sample_arg=sample_x0_hat,
            q_grid_arg=grid_x0_hat,
        )
        metrics.update({
            "stage": "intermediate",
            "timestep": int(t),
            "acceptance_rate": float(per_level_acc[t]),
            "clip_fraction": float(per_level_clip[t]),
        })
        rows.append(metrics)
        panels.append({
            "label": f"intermediate t={t}",
            "samples": trace[t],
            "grid": grid,
            "target_dens": dens,
            "w1": metrics["w1"],
            "ks": metrics["ks"],
            "js": metrics["js"],
            "acc": metrics["acceptance_rate"],
        })

    panel_path = run_dir / "plots" / "density_panel.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    save_density_panel(
        panel_path,
        panels,
        (
            f"mala_steps={cfg.mala_steps}, gradient={cfg.guidance_gradient_space}, "
            f"predictor={cfg.denoising_predictor}, beta={cfg.beta}"
        ),
    )
    return rows


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = args.out_dir / "panels"
    panel_dir.mkdir(exist_ok=True)
    heatmap_metrics = [m.strip() for m in args.heatmap_metrics.replace(",", " ").split() if m.strip()]
    runs = discover_runs(args)
    if not runs:
        raise SystemExit("No completed toy MALA run directories found.")
    if args.expected_runs is not None and len(runs) != args.expected_runs:
        raise SystemExit(
            f"Expected {args.expected_runs} completed run directories, found {len(runs)}. "
            "Check the runs file, failed jobs, or result copy command before trusting plots."
        )

    all_rows = []
    manifest_rows = []
    for run_dir in runs:
        cfg = load_cfg(run_dir / "config.json")
        metrics_path = run_dir / "metrics.csv"
        if args.overwrite_metrics or not metrics_path.exists() or not (run_dir / "plots" / "density_panel.png").exists():
            rows = recompute_metrics_and_panels(run_dir, cfg)
            write_metrics(metrics_path, rows)
        else:
            rows = read_metrics(metrics_path)

        panel_src = run_dir / "plots" / "density_panel.png"
        if panel_src.exists():
            panel_dst = panel_dir / f"{run_dir.name}_density_panel.png"
            panel_dst.write_bytes(panel_src.read_bytes())

        prefix = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "mala_steps": cfg.mala_steps,
            "guidance_gradient_space": cfg.guidance_gradient_space,
            "denoising_predictor": cfg.denoising_predictor,
            "beta": cfg.beta,
            "alpha": cfg.alpha,
            "num_samples": cfg.num_samples,
            "diffusion_steps": cfg.diffusion_steps,
            "x0_hat_clip_radius": cfg.x0_hat_clip_radius,
            "seed": cfg.seed,
        }
        manifest_rows.append(prefix)
        for row in rows:
            all_rows.append({**prefix, **row})

    manifest_path = args.out_dir / "run_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_COLUMNS)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({k: row.get(k, "") for k in KEY_COLUMNS})

    summary_path = args.out_dir / "sweep_metrics.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_COLUMNS + METRIC_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in KEY_COLUMNS + METRIC_COLUMNS})

    heatmap_count = save_all_heatmaps(all_rows, args.out_dir, heatmap_metrics)
    print(f"runs summarized: {len(runs)}")
    print(f"run manifest: {manifest_path}")
    print(f"summary table: {summary_path}")
    print(f"panel images: {panel_dir}")
    print(f"heatmap images: {args.out_dir / 'heatmaps'} ({heatmap_count} files)")


if __name__ == "__main__":
    main()
