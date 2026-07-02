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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KEY_COLUMNS = [
    "run_name",
    "run_dir",
    "target_preset",
    "reward_type",
    "sampler",
    "mala_steps",
    "mala_budget",
    "mala_step_schedule",
    "mala_steps_per_level",
    "mala_eta",
    "langevin_steps",
    "langevin_eta",
    "guidance_gradient_space",
    "denoising_predictor",
    "x0_hat_method",
    "beta",
    "guidance_schedule",
    "alpha",
    "beta_schedule_type",
    "snr_max",
    "num_samples",
    "diffusion_steps",
    "x0_hat_clip_radius",
    "x_recon_clip_radius",
    "mala_adapt_rate",
    "train_steps",
    "policy_parameterization",
    "policy_final_layer",
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
    p.add_argument("--metrics-file", type=Path, default=None,
                   help="Replot from an existing sweep_metrics.csv without reading run directories.")
    p.add_argument("--overwrite-metrics", action="store_true",
                   help="Recompute metrics.csv from samples.npz before aggregating.")
    p.add_argument("--heatmap-metrics", default="kl_sample_target,js,w1,ks,sample_mean_q,target_mean_q,sample_mean,sample_std,acceptance_rate",
                   help="Comma-separated metric names to visualize as sweep heatmaps.")
    p.add_argument("--expected-runs", type=int, default=None,
                   help="Fail if the number of discovered completed run directories differs.")
    return p.parse_args()


def load_cfg(path: Path):
    data = json.loads(path.read_text())
    return SimpleNamespace(**data)


def cfg_to_dict(cfg) -> dict:
    return dict(vars(cfg))


def value_key(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def unique_values(values: list):
    seen = {}
    for value in values:
        seen.setdefault(value_key(value), value)
    return [seen[k] for k in sorted(seen)]


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


def write_experiment_metadata(
    out_dir: Path,
    *,
    args,
    configs: list[dict],
    run_dirs: list[Path],
    heatmap_metrics: list[str],
    manifest_path: Path,
    summary_path: Path,
    panel_dir: Path,
    heatmap_dir: Path,
    heatmap_count: int,
    curve_dir: Path,
    curve_count: int,
) -> tuple[Path, Path]:
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
            "run_manifest": str(manifest_path),
            "sweep_metrics": str(summary_path),
            "panels": str(panel_dir),
            "heatmaps": str(heatmap_dir),
            "heatmap_count": heatmap_count,
            "quality_curves": str(curve_dir),
            "quality_curve_count": curve_count,
        },
        "run_dirs": [str(p) for p in run_dirs],
    }

    json_path = out_dir / "experiment_config.json"
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    md_path = out_dir / "README.md"
    lines = [
        "# Toy MALA Summary",
        "",
        f"- Runs summarized: {len(run_dirs)}",
        f"- Runs file: `{metadata['runs_file']}`",
        f"- Sweep metrics: `{summary_path.name}`",
        f"- Run manifest: `{manifest_path.name}`",
        f"- Panels: `{panel_dir.name}/`",
        f"- Heatmaps: `{heatmap_dir.name}/` ({heatmap_count} files)",
        f"- Quality curves: `{curve_dir.name}/` ({curve_count} files)",
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
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def discover_runs(args) -> list[Path]:
    if args.runs_file is not None:
        runs = [Path(line.strip()) for line in args.runs_file.read_text().splitlines() if line.strip()]
    else:
        runs = sorted(p for p in args.runs_root.glob(args.match) if p.is_dir())
    return [p for p in runs if (p / "config.json").exists() and (p / "samples.npz").exists()]


def read_metrics(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_table(path: Path) -> list[dict]:
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


def np_reward(a, center, scale, cfg=None):
    a = np.asarray(a, dtype=np.float64)
    if cfg is None or getattr(cfg, "reward_type", "quadratic") == "quadratic":
        return -scale * (a - center) ** 2
    centers = np.asarray(getattr(cfg, "reward_bump_centers"), dtype=np.float64)
    widths = np.asarray(getattr(cfg, "reward_bump_widths"), dtype=np.float64)
    weights = np.asarray(getattr(cfg, "reward_bump_weights"), dtype=np.float64)
    bumps = np.sum(
        weights[None, :] * np.exp(-0.5 * ((a[..., None] - centers[None, :]) / widths[None, :]) ** 2),
        axis=-1,
    )
    penalty = float(getattr(cfg, "reward_l2", 0.0)) * a ** 2 + float(getattr(cfg, "reward_l4", 0.0)) * a ** 4
    if cfg.reward_type == "bumps":
        return bumps - penalty
    if cfg.reward_type == "rugged":
        return (
            bumps
            + float(getattr(cfg, "reward_sin_amp", 0.0))
            * np.sin(float(getattr(cfg, "reward_sin_freq", 0.0)) * a + float(getattr(cfg, "reward_sin_phase", 0.0)))
            - penalty
        )
    raise ValueError(f"Unknown reward_type: {cfg.reward_type}")


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
    log_unnorm = cfg.alpha * base_log + effective_beta(cfg) * np_reward(
        q_arg, cfg.reward_center, cfg.reward_scale, cfg
    )
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
    q_sample = float(np.mean(np_reward(q_sample_arg, cfg.reward_center, cfg.reward_scale, cfg)))
    q_target = float(integrate_trapezoid(
        np_reward(q_grid_arg, cfg.reward_center, cfg.reward_scale, cfg) * target_dens,
        grid,
    ))
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


def predictor_order(value):
    order = {"DDIM": 0, "DDPM_mean": 1, "Identity": 2}
    return (order.get(value, 99), value)


HEATMAP_STAGE_KEYS = [
    "final_clean",
    "intermediate_t000",
    "intermediate_t005",
    "intermediate_t010",
    "intermediate_t015",
    "intermediate_t019",
]


def row_stage_key(row):
    stage = row["stage"]
    if stage == "intermediate":
        return f"intermediate_t{int(float(row['timestep'])):03d}"
    return stage


def short_stage_label(stage_key: str) -> str:
    if stage_key == "final_clean":
        return "final_clean"
    return stage_key.replace("intermediate_", "")


def row_guidance_schedule(row: dict) -> str:
    return row.get("guidance_schedule") or "constant"


def panel_title(beta, sampler, eta, gradient, guidance_schedule="constant") -> str:
    eta_text = f"eta={eta:g}" if eta is not None and np.isfinite(eta) else "eta=?"
    if sampler == "mala":
        name = "MALA" if guidance_schedule == "constant" else f"MALA + {guidance_schedule}"
        return f"{name} | {gradient}\n{eta_text}, beta={beta:g}"
    return f"Langevin | {gradient}\n{eta_text}, beta={beta:g}"


def algorithm_panel_order(row_key):
    beta, sampler, eta, gradient, guidance_schedule = row_key
    sampler_rank = 0 if sampler == "mala" else 1
    schedule_rank = {"constant": 0, "alpha_bar": 1}.get(guidance_schedule, 9)
    eta_rank = -1.0 if eta is None or not np.isfinite(eta) else eta
    grad_rank = 0 if gradient == "xt" else 1
    return (sampler_rank, schedule_rank, eta_rank, grad_rank, beta)


def save_heatmap_grid(rows: list[dict], metric: str, stage_key: str, out_path: Path) -> bool:
    reference_rows = [
        row for row in rows
        if (
            row_stage_key(row) == stage_key
            and np.isfinite(as_float(row.get(metric)))
            and (row.get("sampler", "mala") or "mala") in {"dps", "mpgd", "unguided"}
        )
    ]
    selected = [
        row for row in rows
        if (
            row_stage_key(row) == stage_key
            and np.isfinite(as_float(row.get(metric)))
            and (row.get("sampler", "mala") or "mala") in {"mala", "langevin"}
        )
    ]
    if not selected:
        return False

    betas = sorted_unique([as_float(row["beta"]) for row in selected])
    algo_keys = []
    for row in selected:
        sampler = row.get("sampler", "mala") or "mala"
        eta = as_float(row.get("mala_eta", 1.0)) if sampler == "mala" else as_float(row.get("langevin_eta", ""))
        key = (sampler, eta, row_guidance_schedule(row))
        if key not in algo_keys:
            algo_keys.append(key)
    sampler_order = {"mala": 0, "langevin": 1}
    algo_keys = sorted(
        algo_keys,
        key=lambda x: (
            sampler_order.get(x[0], 99),
            {"constant": 0, "alpha_bar": 1}.get(x[2], 9),
            -1.0 if x[1] is None or not np.isfinite(x[1]) else x[1],
        ),
    )
    gradients = [g for g in ["xt", "x0hat"] if any(row["guidance_gradient_space"] == g for row in selected)]
    corrector_steps = sorted_unique([
        int(float(row["langevin_steps"] if (row.get("sampler", "mala") or "mala") == "langevin" else row["mala_steps"]))
        for row in selected
    ])
    predictors = sorted(set(row["denoising_predictor"] for row in selected), key=predictor_order)
    if not corrector_steps or not predictors:
        return False

    panel_keys = sorted(
        [
            (beta, sampler, eta, gradient, guidance_schedule)
            for beta in betas
            for sampler, eta, guidance_schedule in algo_keys
            for gradient in gradients
        ],
        key=algorithm_panel_order,
    )
    ncols = 2
    nrows = int(np.ceil(len(panel_keys) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.2 * ncols + 1.4, 5.7 * nrows + 0.8),
        squeeze=False,
    )
    for ax in axes.ravel()[len(panel_keys):]:
        ax.axis("off")

    all_values = np.asarray([as_float(row[metric]) for row in selected], dtype=np.float64)
    finite_values = all_values[np.isfinite(all_values)]
    vmin = float(np.min(finite_values)) if finite_values.size else None
    vmax = float(np.max(finite_values)) if finite_values.size else None
    image = None
    for ax, (beta, sampler, eta, gradient, guidance_schedule) in zip(axes.ravel(), panel_keys):
        mat = np.full((len(corrector_steps), len(predictors)), np.nan, dtype=np.float64)
        for row in selected:
            row_sampler = row.get("sampler", "mala") or "mala"
            row_eta = as_float(row.get("mala_eta", 1.0)) if row_sampler == "mala" else as_float(row.get("langevin_eta", ""))
            eta_matches = row_eta == eta
            if (
                as_float(row["beta"]) != beta
                or row_sampler != sampler
                or not eta_matches
                or row["guidance_gradient_space"] != gradient
                or row_guidance_schedule(row) != guidance_schedule
            ):
                continue
            step_value = row["langevin_steps"] if row_sampler == "langevin" else row["mala_steps"]
            y = corrector_steps.index(int(float(step_value)))
            x = predictors.index(row["denoising_predictor"])
            mat[y, x] = as_float(row[metric])
        image = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(predictors)), predictors, rotation=25, ha="right", fontsize=11)
        ax.set_yticks(np.arange(len(corrector_steps)), corrector_steps, fontsize=11)
        ax.set_xlabel("denoising predictor", fontsize=12)
        ax.set_ylabel("corrector steps", fontsize=12)
        ax.set_title(panel_title(beta, sampler, eta, gradient, guidance_schedule), fontsize=13, pad=10)
        for y in range(len(corrector_steps)):
            for x in range(len(predictors)):
                val = mat[y, x]
                if np.isfinite(val):
                    ax.text(x, y, f"{val:.3g}", ha="center", va="center", color="white", fontsize=10)

    if image is not None:
        cax = fig.add_axes([0.915, 0.14, 0.018, 0.74])
        fig.colorbar(image, cax=cax, label=metric)
    if reference_rows:
        ref_text = " | ".join(
            f"{row.get('sampler')}:{row.get('denoising_predictor')} "
            f"beta={as_float(row.get('beta')):g} {metric}={as_float(row.get(metric)):.4g}"
            for row in reference_rows[:8]
        )
        if len(reference_rows) > 8:
            ref_text += f" | +{len(reference_rows) - 8} more refs in sweep_metrics.csv"
        fig.text(0.5, 0.035, f"DDIM references: {ref_text}", ha="center", va="bottom", fontsize=10, wrap=True)
    fig.suptitle(f"{stage_key}: {metric}", fontsize=16)
    fig.subplots_adjust(
        left=0.06,
        right=0.885,
        bottom=0.13 if reference_rows else 0.065,
        top=0.93,
        hspace=0.62,
        wspace=0.28,
    )
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return True


def save_all_heatmaps(rows: list[dict], out_dir: Path, metrics: list[str]) -> int:
    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    present = set(row_stage_key(row) for row in rows)
    stage_keys = [stage_key for stage_key in HEATMAP_STAGE_KEYS if stage_key in present]
    count = 0
    for stage_key in stage_keys:
        stage_dir = heatmap_dir / short_stage_label(stage_key)
        stage_dir.mkdir(exist_ok=True)
        for metric in metrics:
            path = stage_dir / f"{metric}.png"
            if save_heatmap_grid(rows, metric, stage_key, path):
                count += 1
    return count


def save_quality_curve(rows: list[dict], metric: str, stage_key: str, out_path: Path) -> bool:
    selected = [
        row for row in rows
        if (
            row_stage_key(row) == stage_key
            and np.isfinite(as_float(row.get(metric)))
            and (row.get("sampler", "mala") or "mala") in {"mala", "langevin"}
        )
    ]
    if not selected:
        return False

    predictors = sorted_unique([row["denoising_predictor"] for row in selected])
    fig, axes = plt.subplots(
        1,
        len(predictors),
        figsize=(5.0 * len(predictors), 3.8),
        squeeze=False,
        constrained_layout=True,
    )
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    reference_rows = [
        row for row in rows
        if (
            row_stage_key(row) == stage_key
            and np.isfinite(as_float(row.get(metric)))
            and (row.get("sampler", "mala") or "mala") in {"dps", "mpgd", "unguided"}
        )
    ]

    any_line = False
    for ax, predictor in zip(axes.ravel(), predictors):
        pred_rows = [row for row in selected if row["denoising_predictor"] == predictor]
        groups = {}
        for row in pred_rows:
            sampler = row.get("sampler", "mala") or "mala"
            gradient = row["guidance_gradient_space"]
            eta = as_float(row.get("mala_eta", 1.0)) if sampler == "mala" else as_float(row.get("langevin_eta", ""))
            key = (sampler, eta, gradient, row_guidance_schedule(row))
            step_value = row["langevin_steps"] if sampler == "langevin" else row["mala_steps"]
            groups.setdefault(key, []).append((int(float(step_value)), as_float(row[metric])))

        for (sampler, eta, gradient, guidance_schedule), points in sorted(
            groups.items(),
            key=lambda item: (
                0 if item[0][0] == "mala" else 1,
                {"constant": 0, "alpha_bar": 1}.get(item[0][3], 9),
                -1 if item[0][1] is None else item[0][1],
                item[0][2],
            ),
        ):
            points = sorted(points)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if not xs:
                continue
            eta_label = f" eta={eta:g}" if eta is not None and np.isfinite(eta) else ""
            sampler_label = sampler if guidance_schedule == "constant" else f"{sampler}+{guidance_schedule}"
            label = f"{sampler_label}{eta_label} {gradient}"
            color_idx = abs(
                hash((sampler, guidance_schedule, round(float(eta), 8) if np.isfinite(eta) else None, gradient))
            ) % max(len(color_cycle), 1)
            color = color_cycle[color_idx] if color_cycle else None
            ax.plot(xs, ys, marker="o", label=label, color=color)
            any_line = True

        for ref in reference_rows:
            if ref.get("denoising_predictor") != predictor:
                continue
            sampler = ref.get("sampler", "")
            val = as_float(ref.get(metric))
            ax.axhline(val, linestyle="--", linewidth=1.0, alpha=0.75, label=f"{sampler} ref")

        ax.set_title(predictor)
        ax.set_xlabel("corrector steps")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    if not any_line:
        plt.close(fig)
        return False
    fig.suptitle(f"{stage_key}: {metric} vs corrector steps")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def save_quality_curves(rows: list[dict], out_dir: Path, metrics: list[str]) -> int:
    curve_dir = out_dir / "quality_curves"
    curve_dir.mkdir(exist_ok=True)
    present = set(row_stage_key(row) for row in rows)
    stage_keys = [stage_key for stage_key in HEATMAP_STAGE_KEYS if stage_key in present]
    curve_metrics = [m for m in metrics if m in {"kl_sample_target", "js", "w1", "ks", "sample_mean_q"}]
    count = 0
    for stage_key in stage_keys:
        stage_dir = curve_dir / short_stage_label(stage_key)
        stage_dir.mkdir(exist_ok=True)
        for metric in curve_metrics:
            if save_quality_curve(rows, metric, stage_key, stage_dir / f"{metric}.png"):
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
            f"sampler={get_cfg_attr(cfg, 'sampler', 'mala')}, mala_steps={cfg.mala_steps}, "
            f"gradient={cfg.guidance_gradient_space}, "
            f"predictor={cfg.denoising_predictor}, beta={cfg.beta}"
        ),
    )
    return rows


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_metrics = [m.strip() for m in args.heatmap_metrics.replace(",", " ").split() if m.strip()]

    if args.metrics_file is not None:
        all_rows = read_table(args.metrics_file)
        if not all_rows:
            raise SystemExit(f"No rows found in metrics file: {args.metrics_file}")
        heatmap_count = save_all_heatmaps(all_rows, args.out_dir, heatmap_metrics)
        curve_count = save_quality_curves(all_rows, args.out_dir, heatmap_metrics)
        print(f"replotted from: {args.metrics_file}")
        print(f"heatmap images: {args.out_dir / 'heatmaps'} ({heatmap_count} files)")
        print(f"quality curve images: {args.out_dir / 'quality_curves'} ({curve_count} files)")
        return

    panel_dir = args.out_dir / "panels"
    panel_dir.mkdir(exist_ok=True)
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
    configs = []
    for run_dir in runs:
        cfg = load_cfg(run_dir / "config.json")
        configs.append(cfg_to_dict(cfg))
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
            "target_preset": get_cfg_attr(cfg, "target_preset", "manual"),
            "reward_type": get_cfg_attr(cfg, "reward_type", "quadratic"),
            "sampler": get_cfg_attr(cfg, "sampler", "mala"),
            "mala_steps": cfg.mala_steps,
            "mala_budget": get_cfg_attr(cfg, "mala_budget", cfg.mala_steps * cfg.diffusion_steps),
            "mala_step_schedule": get_cfg_attr(cfg, "mala_step_schedule", "constant"),
            "mala_steps_per_level": json.dumps(
                get_cfg_attr(cfg, "mala_steps_per_level", [cfg.mala_steps] * cfg.diffusion_steps)
            ),
            "mala_eta": get_cfg_attr(cfg, "mala_eta", 1.0),
            "langevin_steps": get_cfg_attr(cfg, "langevin_steps", ""),
            "langevin_eta": get_cfg_attr(cfg, "langevin_eta", ""),
            "guidance_gradient_space": cfg.guidance_gradient_space,
            "denoising_predictor": cfg.denoising_predictor,
            "x0_hat_method": get_cfg_attr(cfg, "x0_hat_method", "posterior_mean"),
            "beta": cfg.beta,
            "guidance_schedule": get_cfg_attr(cfg, "guidance_schedule", "constant"),
            "alpha": cfg.alpha,
            "beta_schedule_type": cfg.beta_schedule_type,
            "snr_max": cfg.snr_max,
            "num_samples": cfg.num_samples,
            "diffusion_steps": cfg.diffusion_steps,
            "x0_hat_clip_radius": cfg.x0_hat_clip_radius,
            "x_recon_clip_radius": get_cfg_attr(cfg, "x_recon_clip_radius", 1.0),
            "mala_adapt_rate": cfg.mala_adapt_rate,
            "train_steps": get_cfg_attr(cfg, "train_steps", ""),
            "policy_parameterization": get_cfg_attr(cfg, "policy_parameterization", ""),
            "policy_final_layer": get_cfg_attr(cfg, "policy_final_layer", ""),
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
    curve_count = save_quality_curves(all_rows, args.out_dir, heatmap_metrics)
    heatmap_dir = args.out_dir / "heatmaps"
    curve_dir = args.out_dir / "quality_curves"
    config_path, readme_path = write_experiment_metadata(
        args.out_dir,
        args=args,
        configs=configs,
        run_dirs=runs,
        heatmap_metrics=heatmap_metrics,
        manifest_path=manifest_path,
        summary_path=summary_path,
        panel_dir=panel_dir,
        heatmap_dir=heatmap_dir,
        heatmap_count=heatmap_count,
        curve_dir=curve_dir,
        curve_count=curve_count,
    )
    print(f"runs summarized: {len(runs)}")
    print(f"experiment config: {config_path}")
    print(f"experiment readme: {readme_path}")
    print(f"run manifest: {manifest_path}")
    print(f"summary table: {summary_path}")
    print(f"panel images: {panel_dir}")
    print(f"heatmap images: {heatmap_dir} ({heatmap_count} files)")
    print(f"quality curve images: {curve_dir} ({curve_count} files)")


if __name__ == "__main__":
    main()
