#!/usr/bin/env python3
"""Visualize 1D oracle guided targets across diffusion levels.

This script does not sample. It evaluates the same normalized target densities
used by toy_mala_1d.py:

    clean: p0(x)^alpha exp(beta_eff Q(x))
    t:     pt(x_t)^alpha exp(beta_eff Q(clip(x0_hat(x_t))))

Outputs a static PNG panel and a standalone local HTML slider.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class VizConfig:
    target_preset: str
    gmm_weights: list[float]
    gmm_means: list[float]
    gmm_stds: list[float]
    reward_type: str
    reward_center: float
    reward_scale: float
    reward_bump_centers: list[float]
    reward_bump_widths: list[float]
    reward_bump_weights: list[float]
    reward_sin_amp: float
    reward_sin_freq: float
    reward_sin_phase: float
    reward_l2: float
    reward_l4: float
    diffusion_steps: int
    beta_schedule_type: str
    snr_max: float
    alpha: float
    beta: float
    advantage_normalization: bool
    initial_advantage_second_moment_ema: float
    x0_hat_method: str
    x0_hat_clip_radius: float
    grid_min: float
    grid_max: float
    grid_points: int
    output_dir: str


@dataclass(frozen=True)
class NumpySchedule:
    betas: np.ndarray
    alphas_cumprod: np.ndarray
    sqrt_alphas_cumprod: np.ndarray
    sqrt_one_minus_alphas_cumprod: np.ndarray
    sqrt_recip_alphas_cumprod: np.ndarray
    sqrt_recipm1_alphas_cumprod: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target_preset", choices=["manual", "complex_1d_v1", "complex_1d_v2"], default="manual")
    p.add_argument("--gmm_weights", default="0.35,0.30,0.35")
    p.add_argument("--gmm_means", default="-0.75,0.0,0.75")
    p.add_argument("--gmm_stds", default="0.10,0.16,0.10")
    p.add_argument("--reward_type", choices=["quadratic", "bumps", "rugged"], default="quadratic")
    p.add_argument("--reward_center", type=float, default=0.45)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--reward_bump_centers", default="-0.72,-0.28,0.18,0.64,0.92")
    p.add_argument("--reward_bump_widths", default="0.055,0.09,0.06,0.12,0.045")
    p.add_argument("--reward_bump_weights", default="0.85,-0.45,0.75,1.10,-0.35")
    p.add_argument("--reward_sin_amp", type=float, default=0.12)
    p.add_argument("--reward_sin_freq", type=float, default=18.0)
    p.add_argument("--reward_sin_phase", type=float, default=0.4)
    p.add_argument("--reward_l2", type=float, default=0.08)
    p.add_argument("--reward_l4", type=float, default=0.02)
    p.add_argument("--diffusion_steps", type=int, default=20)
    p.add_argument("--beta_schedule_type", choices=["linear", "cosine", "constant_kl"], default="cosine")
    p.add_argument("--snr_max", type=float, default=124.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--advantage_normalization", action="store_true")
    p.add_argument("--initial_advantage_second_moment_ema", type=float, default=1.0)
    p.add_argument("--x0_hat_method", choices=["posterior_mean", "tweedie"], default="tweedie")
    p.add_argument("--x0_hat_clip_radius", type=float, default=10.0)
    p.add_argument("--grid_min", type=float, default=-6.0)
    p.add_argument("--grid_max", type=float, default=6.0)
    p.add_argument("--grid_points", type=int, default=2001)
    p.add_argument("--output_dir", type=Path, default=Path("toy_mala_1d_target_viz"))
    args = p.parse_args()
    if args.target_preset in {"complex_1d_v1", "complex_1d_v2"}:
        args.gmm_weights = "0.07,0.18,0.11,0.24,0.08,0.19,0.13"
        args.gmm_means = "-1.25,-0.82,-0.46,-0.08,0.22,0.61,1.05"
        args.gmm_stds = "0.055,0.10,0.045,0.15,0.035,0.09,0.06"
        args.reward_type = "rugged"
        if args.target_preset == "complex_1d_v1":
            args.reward_bump_centers = "-1.04,-0.58,-0.19,0.36,0.78,1.12"
            args.reward_bump_widths = "0.06,0.08,0.05,0.10,0.055,0.08"
            args.reward_bump_weights = "0.75,-0.65,1.05,0.82,-0.55,0.45"
            args.reward_sin_amp = 0.16
            args.reward_sin_freq = 22.0
            args.reward_sin_phase = 0.35
            args.reward_l2 = 0.07
            args.reward_l4 = 0.018
        else:
            args.reward_bump_centers = "-1.04,-0.63,-0.08,0.39,0.61,0.88"
            args.reward_bump_widths = "0.055,0.055,0.12,0.06,0.10,0.055"
            args.reward_bump_weights = "3.2,2.4,-3.0,3.0,-2.6,2.8"
            args.reward_sin_amp = 0.10
            args.reward_sin_freq = 18.0
            args.reward_sin_phase = 0.20
            args.reward_l2 = 0.03
            args.reward_l4 = 0.01
    return args


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.replace(",", " ").split()]


def normalize_weights(weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("GMM weights must be finite nonnegative values with positive sum.")
    return w / w.sum()


def cosine_beta_schedule(timesteps: int) -> np.ndarray:
    s = 0.008
    t = np.arange(0, timesteps + 1, dtype=np.float64) / timesteps
    alphas_cumprod = np.cos((t + s) / (1.0 + s) * np.pi / 2.0) ** 2
    alphas_cumprod /= alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, 0.0, 0.999)


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.999) -> np.ndarray:
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


def constant_kl_beta_schedule(timesteps: int, snr_max=1000.0) -> np.ndarray:
    r = 1.0 + snr_max
    k = np.arange(timesteps, dtype=np.float64)
    exponent = (timesteps - k) / timesteps
    alphas_cumprod = 1.0 - r ** (-exponent)
    alphas_cumprod_with_1 = np.concatenate([[1.0], alphas_cumprod])
    betas = 1.0 - alphas_cumprod_with_1[1:] / alphas_cumprod_with_1[:-1]
    return np.clip(betas, 1e-8, 0.999)


def build_schedule(timesteps: int, schedule_type: str, snr_max: float) -> NumpySchedule:
    target_abar_0 = snr_max / (1.0 + snr_max)
    if schedule_type == "constant_kl":
        betas = constant_kl_beta_schedule(timesteps, snr_max)
    elif schedule_type == "cosine":
        raw_betas = cosine_beta_schedule(timesteps)
        scale = (1.0 - target_abar_0) / raw_betas[0]
        betas = np.clip(scale * raw_betas, 0.0, 0.999)
    elif schedule_type == "linear":
        raw_betas = linear_beta_schedule(timesteps)
        scale = (1.0 - target_abar_0) / raw_betas[0]
        betas = np.clip(scale * raw_betas, 0.0, 0.999)
    else:
        raise ValueError(f"Unknown beta_schedule_type: {schedule_type}")
    alphas_cumprod = np.cumprod(1.0 - betas)
    return NumpySchedule(
        betas=betas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=np.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=np.sqrt(1.0 - alphas_cumprod),
        sqrt_recip_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod),
        sqrt_recipm1_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod - 1.0),
    )


def logsumexp(a: np.ndarray, axis=-1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def base_logpdf(x, weights, means, stds, schedule: NumpySchedule, t_idx: int | None):
    x = np.asarray(x, dtype=np.float64)
    if t_idx is None:
        loc = means
        var = stds ** 2
    else:
        c = schedule.sqrt_alphas_cumprod[t_idx]
        sigma = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
        loc = c * means
        var = c * c * stds ** 2 + sigma * sigma
    log_comp = (
        np.log(weights)[None, :]
        - 0.5 * (np.log(2.0 * np.pi * var)[None, :] + (x[:, None] - loc[None, :]) ** 2 / var[None, :])
    )
    return logsumexp(log_comp, axis=-1)


def base_score(x, weights, means, stds, schedule: NumpySchedule, t_idx: int):
    x = np.asarray(x, dtype=np.float64)
    c = schedule.sqrt_alphas_cumprod[t_idx]
    sigma = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
    loc = c * means
    var = c * c * stds ** 2 + sigma * sigma
    log_comp = (
        np.log(weights)[None, :]
        - 0.5 * (np.log(2.0 * np.pi * var)[None, :] + (x[:, None] - loc[None, :]) ** 2 / var[None, :])
    )
    resp = np.exp(log_comp - logsumexp(log_comp, axis=-1)[:, None])
    return np.sum(resp * (-(x[:, None] - loc[None, :]) / var[None, :]), axis=-1)


def x0_hat(x, weights, means, stds, schedule: NumpySchedule, t_idx: int, method: str):
    x = np.asarray(x, dtype=np.float64)
    c = schedule.sqrt_alphas_cumprod[t_idx]
    sigma = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
    loc = c * means
    var = c * c * stds ** 2 + sigma * sigma
    log_comp = (
        np.log(weights)[None, :]
        - 0.5 * (np.log(2.0 * np.pi * var)[None, :] + (x[:, None] - loc[None, :]) ** 2 / var[None, :])
    )
    resp = np.exp(log_comp - logsumexp(log_comp, axis=-1)[:, None])
    if method == "tweedie":
        score = base_score(x, weights, means, stds, schedule, t_idx)
        return (x + sigma * sigma * score) / c
    if method != "posterior_mean":
        raise ValueError(f"Unknown x0_hat_method: {method}")
    posterior_mean = means[None, :] + (c * stds[None, :] ** 2 / var[None, :]) * (x[:, None] - loc[None, :])
    return np.sum(resp * posterior_mean, axis=-1)


def reward(a, cfg: VizConfig) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if cfg.reward_type == "quadratic":
        return -cfg.reward_scale * (a - cfg.reward_center) ** 2
    centers = np.asarray(cfg.reward_bump_centers, dtype=np.float64)
    widths = np.asarray(cfg.reward_bump_widths, dtype=np.float64)
    weights = np.asarray(cfg.reward_bump_weights, dtype=np.float64)
    bumps = np.sum(
        weights[None, :] * np.exp(-0.5 * ((a[..., None] - centers[None, :]) / widths[None, :]) ** 2),
        axis=-1,
    )
    penalty = cfg.reward_l2 * a ** 2 + cfg.reward_l4 * a ** 4
    if cfg.reward_type == "bumps":
        return bumps - penalty
    if cfg.reward_type == "rugged":
        return bumps + cfg.reward_sin_amp * np.sin(cfg.reward_sin_freq * a + cfg.reward_sin_phase) - penalty
    raise ValueError(f"Unknown reward_type: {cfg.reward_type}")


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def normalize_density(log_unnorm: np.ndarray, grid: np.ndarray) -> np.ndarray:
    dens = np.exp(log_unnorm - np.max(log_unnorm))
    z = integrate_trapezoid(dens, grid)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Invalid density normalization constant.")
    return dens / z


def effective_beta(cfg: VizConfig) -> float:
    beta = float(cfg.beta)
    if cfg.advantage_normalization:
        beta /= float(np.sqrt(max(float(cfg.initial_advantage_second_moment_ema), 1e-6)))
    return beta


def target_density(grid, cfg: VizConfig, weights, means, stds, schedule: NumpySchedule, t_idx: int | None):
    if t_idx is None:
        base_log = base_logpdf(grid, weights, means, stds, schedule, None)
        q_arg = grid
    else:
        base_log = base_logpdf(grid, weights, means, stds, schedule, t_idx)
        q_arg = x0_hat(grid, weights, means, stds, schedule, t_idx, cfg.x0_hat_method)
        if np.isfinite(cfg.x0_hat_clip_radius):
            q_arg = np.clip(q_arg, -cfg.x0_hat_clip_radius, cfg.x0_hat_clip_radius)
    log_unnorm = cfg.alpha * base_log + effective_beta(cfg) * reward(q_arg, cfg)
    return normalize_density(log_unnorm, grid)


def original_density(grid, weights, means, stds, schedule: NumpySchedule, t_idx: int | None):
    return normalize_density(base_logpdf(grid, weights, means, stds, schedule, t_idx), grid)


def density_distances(grid, p, q) -> dict[str, float]:
    dx = np.diff(grid)
    mids = 0.5 * (grid[:-1] + grid[1:])
    p_mass = 0.5 * (p[:-1] + p[1:]) * dx
    q_mass = 0.5 * (q[:-1] + q[1:]) * dx
    p_mass = p_mass / p_mass.sum()
    q_mass = q_mass / q_mass.sum()
    eps = 1e-12
    m = 0.5 * (p_mass + q_mass)
    js = 0.5 * np.sum(p_mass * (np.log(p_mass + eps) - np.log(m + eps)))
    js += 0.5 * np.sum(q_mass * (np.log(q_mass + eps) - np.log(m + eps)))
    cdf_p = np.cumsum(p_mass)
    cdf_q = np.cumsum(q_mass)
    return {
        "js_orig_target": float(js),
        "w1_orig_target": float(integrate_trapezoid(np.abs(cdf_p - cdf_q), mids)),
        "ks_orig_target": float(np.max(np.abs(cdf_p - cdf_q))),
    }


def make_config(args: argparse.Namespace, weights, means, stds) -> VizConfig:
    reward_bump_centers = parse_float_list(args.reward_bump_centers)
    reward_bump_widths = parse_float_list(args.reward_bump_widths)
    reward_bump_weights = parse_float_list(args.reward_bump_weights)
    if args.reward_type != "quadratic":
        if not (len(reward_bump_centers) == len(reward_bump_widths) == len(reward_bump_weights)):
            raise ValueError("reward bump parameter lengths must match.")
        if np.any(np.asarray(reward_bump_widths) <= 0):
            raise ValueError("reward_bump_widths must be positive.")
    return VizConfig(
        target_preset=args.target_preset,
        gmm_weights=weights.tolist(),
        gmm_means=means.tolist(),
        gmm_stds=stds.tolist(),
        reward_type=args.reward_type,
        reward_center=args.reward_center,
        reward_scale=args.reward_scale,
        reward_bump_centers=reward_bump_centers,
        reward_bump_widths=reward_bump_widths,
        reward_bump_weights=reward_bump_weights,
        reward_sin_amp=args.reward_sin_amp,
        reward_sin_freq=args.reward_sin_freq,
        reward_sin_phase=args.reward_sin_phase,
        reward_l2=args.reward_l2,
        reward_l4=args.reward_l4,
        diffusion_steps=args.diffusion_steps,
        beta_schedule_type=args.beta_schedule_type,
        snr_max=args.snr_max,
        alpha=args.alpha,
        beta=args.beta,
        advantage_normalization=args.advantage_normalization,
        initial_advantage_second_moment_ema=args.initial_advantage_second_moment_ema,
        x0_hat_method=args.x0_hat_method,
        x0_hat_clip_radius=args.x0_hat_clip_radius,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        grid_points=args.grid_points,
        output_dir=str(args.output_dir),
    )


def compute_targets(cfg: VizConfig, weights, means, stds):
    schedule = build_schedule(cfg.diffusion_steps, cfg.beta_schedule_type, cfg.snr_max)
    grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_points, dtype=np.float64)
    labels = ["pi_orig", "final_clean pi_target"]
    densities = [
        original_density(grid, weights, means, stds, schedule, None),
        target_density(grid, cfg, weights, means, stds, schedule, None),
    ]
    for t in range(cfg.diffusion_steps):
        labels.append(f"t={t}")
        densities.append(target_density(grid, cfg, weights, means, stds, schedule, t))
    return grid, labels, np.asarray(densities, dtype=np.float64)


def save_orig_target_comparison(path: Path, grid, pi_orig, pi_target, cfg: VizConfig, distances) -> None:
    q = reward(grid, cfg)
    fig, ax1 = plt.subplots(figsize=(9.0, 4.8))
    ax1.plot(grid, pi_orig, color="0.35", lw=2.0, ls="--", label=r"$\pi_{\mathrm{orig}}$")
    ax1.plot(grid, pi_target, color="tab:orange", lw=2.3, label=r"$\pi_{\mathrm{target}}$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("density")
    ax1.set_xlim(float(grid[0]), float(grid[-1]))
    ax2 = ax1.twinx()
    ax2.plot(grid, q, color="tab:blue", lw=1.0, alpha=0.42, label="Q(x)")
    ax2.set_ylabel("Q(x)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    ax1.set_title(
        "clean target comparison: "
        f"JS={distances['js_orig_target']:.4f}, "
        f"W1={distances['w1_orig_target']:.4f}, "
        f"KS={distances['ks_orig_target']:.4f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_panel(path: Path, grid, labels, densities) -> None:
    n = len(labels)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.5 * rows), squeeze=False)
    ymax = float(np.max(densities) * 1.08)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, label, dens in zip(axes.ravel(), labels, densities):
        ax.plot(grid, dens, color="tab:blue", lw=1.8)
        ax.set_title(label, fontsize=9)
        ax.set_xlim(float(grid[0]), float(grid[-1]))
        ax.set_ylim(0.0, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("density")
    fig.suptitle("1D guided target densities")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_slider(path: Path, grid, labels, densities, cfg: VizConfig) -> None:
    payload = {
        "grid": np.round(grid, 8).tolist(),
        "labels": labels,
        "densities": np.round(densities, 8).tolist(),
        "config": asdict(cfg),
    }
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>1D Guided Target Densities</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
    canvas {{ width: 100%; max-width: 980px; height: 460px; border: 1px solid #ddd; }}
    .row {{ max-width: 980px; display: flex; gap: 12px; align-items: center; margin: 12px 0; }}
    input[type=range] {{ flex: 1; }}
    pre {{ max-width: 980px; overflow: auto; background: #f7f7f7; padding: 12px; }}
  </style>
</head>
<body>
  <h2>1D Guided Target Densities</h2>
  <div class="row">
    <strong id="label"></strong>
    <input id="slider" type="range" min="0" max="{len(labels) - 1}" value="0" step="1">
  </div>
  <canvas id="plot" width="980" height="460"></canvas>
  <pre id="cfg"></pre>
  <script>
    const data = {json.dumps(payload)};
    const canvas = document.getElementById("plot");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("slider");
    const label = document.getElementById("label");
    document.getElementById("cfg").textContent = JSON.stringify(data.config, null, 2);
    const margin = {{left: 58, right: 18, top: 24, bottom: 48}};
    const xmin = data.grid[0];
    const xmax = data.grid[data.grid.length - 1];
    const ymax = Math.max(...data.densities.flat()) * 1.08;
    function sx(x) {{
      return margin.left + (x - xmin) / (xmax - xmin) * (canvas.width - margin.left - margin.right);
    }}
    function sy(y) {{
      return canvas.height - margin.bottom - y / ymax * (canvas.height - margin.top - margin.bottom);
    }}
    function drawAxes() {{
      ctx.strokeStyle = "#444";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, canvas.height - margin.bottom);
      ctx.lineTo(canvas.width - margin.right, canvas.height - margin.bottom);
      ctx.stroke();
      ctx.fillStyle = "#333";
      ctx.font = "13px system-ui";
      ctx.fillText(xmin.toFixed(2), margin.left - 10, canvas.height - 18);
      ctx.fillText(xmax.toFixed(2), canvas.width - margin.right - 38, canvas.height - 18);
      ctx.fillText("density", 10, 20);
      ctx.fillText("x", canvas.width - 28, canvas.height - 18);
    }}
    function draw(i) {{
      label.textContent = data.labels[i];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawAxes();
      const ys = data.densities[i];
      ctx.strokeStyle = "#1f77b4";
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      for (let k = 0; k < data.grid.length; k++) {{
        const x = sx(data.grid[k]);
        const y = sy(ys[k]);
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.stroke();
    }}
    slider.addEventListener("input", () => draw(Number(slider.value)));
    draw(0);
  </script>
</body>
</html>
"""
    path.write_text(html)


def main() -> None:
    args = parse_args()
    weights = normalize_weights(parse_float_list(args.gmm_weights))
    means = np.asarray(parse_float_list(args.gmm_means), dtype=np.float64)
    stds = np.asarray(parse_float_list(args.gmm_stds), dtype=np.float64)
    if not (len(weights) == len(means) == len(stds)):
        raise ValueError("GMM weights, means, and stds must have the same length.")
    if np.any(stds <= 0):
        raise ValueError("GMM stds must be positive.")

    cfg = make_config(args, weights, means, stds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid, labels, densities = compute_targets(cfg, weights, means, stds)
    distances = density_distances(grid, densities[0], densities[1])
    q_grid = reward(grid, cfg)
    diagnostics = {
        **distances,
        "orig_mean_q": float(integrate_trapezoid(q_grid * densities[0], grid)),
        "target_mean_q": float(integrate_trapezoid(q_grid * densities[1], grid)),
        "reward_grid_min": float(np.min(q_grid)),
        "reward_grid_max": float(np.max(q_grid)),
    }
    save_orig_target_comparison(
        args.output_dir / "orig_vs_target.png",
        grid,
        densities[0],
        densities[1],
        cfg,
        distances,
    )
    save_panel(args.output_dir / "target_panel.png", grid, labels, densities)
    save_slider(args.output_dir / "target_slider.html", grid, labels, densities, cfg)
    np.savez_compressed(
        args.output_dir / "target_densities.npz",
        grid=grid,
        labels=np.asarray(labels),
        densities=densities,
    )
    (args.output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    (args.output_dir / "target_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(
        "pi_orig vs pi_target: "
        f"JS={distances['js_orig_target']:.6f}, "
        f"W1={distances['w1_orig_target']:.6f}, "
        f"KS={distances['ks_orig_target']:.6f}"
    )
    print(f"wrote {args.output_dir / 'orig_vs_target.png'}")
    print(f"wrote {args.output_dir / 'target_panel.png'}")
    print(f"wrote {args.output_dir / 'target_slider.html'}")


if __name__ == "__main__":
    main()
