#!/usr/bin/env python3
"""Measure adjacent target-density gaps for toy diffusion schedules.

This is a lightweight local diagnostic. It does not sample and does not use
JAX. For each requested schedule, it builds the same beta schedule used by the
repo, constructs the 1D GMM forward marginals, then measures how far adjacent
target densities are from each other:

    clean -> t0, t0 -> t1, ..., t(T-2) -> t(T-1).

It reports both base marginals and guided targets so schedule deformation can
be separated from reward guidance deformation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.replace(",", " ").split()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedules", default="linear,cosine,constant_kl")
    p.add_argument("--diffusion_steps", type=int, default=20)
    p.add_argument("--snr_max", type=float, default=124.0)
    p.add_argument("--gmm_weights", default="0.18,0.37,0.25,0.20")
    p.add_argument("--gmm_means", default="-0.85,-0.15,0.10,0.82")
    p.add_argument("--gmm_stds", default="0.09,0.06,0.16,0.08")
    p.add_argument("--reward_center", type=float, default=0.65)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--x0_hat_clip_radius", type=float, default=10.0)
    p.add_argument("--grid_min", type=float, default=-6.0)
    p.add_argument("--grid_max", type=float, default=6.0)
    p.add_argument("--grid_points", type=int, default=5001)
    p.add_argument("--out_dir", type=Path, default=Path("toy_schedule_gap_analysis"))
    return p.parse_args()


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("GMM weights must be finite nonnegative values with positive sum.")
    return weights / weights.sum()


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


def build_schedule(timesteps: int, schedule_type: str, snr_max: float) -> dict[str, np.ndarray]:
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
        raise ValueError(f"Unknown schedule: {schedule_type}")

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": np.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": np.sqrt(1.0 - alphas_cumprod),
    }


def logsumexp(a: np.ndarray, axis=-1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def base_logpdf(grid, weights, means, stds, schedule: dict[str, np.ndarray], t_idx: int | None):
    if t_idx is None:
        loc = means
        var = stds ** 2
    else:
        c = schedule["sqrt_alphas_cumprod"][t_idx]
        sigma = schedule["sqrt_one_minus_alphas_cumprod"][t_idx]
        loc = c * means
        var = c * c * stds ** 2 + sigma * sigma
    log_comp = (
        np.log(weights)[None, :]
        - 0.5
        * (np.log(2.0 * np.pi * var)[None, :] + (grid[:, None] - loc[None, :]) ** 2 / var[None, :])
    )
    return logsumexp(log_comp, axis=-1)


def x0_hat(grid, weights, means, stds, schedule: dict[str, np.ndarray], t_idx: int) -> np.ndarray:
    c = schedule["sqrt_alphas_cumprod"][t_idx]
    sigma = schedule["sqrt_one_minus_alphas_cumprod"][t_idx]
    loc = c * means
    var = c * c * stds ** 2 + sigma * sigma
    log_comp = (
        np.log(weights)[None, :]
        - 0.5
        * (np.log(2.0 * np.pi * var)[None, :] + (grid[:, None] - loc[None, :]) ** 2 / var[None, :])
    )
    resp = np.exp(log_comp - logsumexp(log_comp, axis=-1)[:, None])
    posterior_mean = means[None, :] + (c * stds[None, :] ** 2 / var[None, :]) * (
        grid[:, None] - loc[None, :]
    )
    return np.sum(resp * posterior_mean, axis=-1)


def reward(a, center: float, scale: float) -> np.ndarray:
    return -scale * (a - center) ** 2


def normalize_density(log_unnorm: np.ndarray, grid: np.ndarray) -> np.ndarray:
    dens = np.exp(log_unnorm - np.max(log_unnorm))
    z = np.trapezoid(dens, grid) if hasattr(np, "trapezoid") else np.trapz(dens, grid)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Invalid density normalization.")
    return dens / z


def target_density(
    grid,
    weights,
    means,
    stds,
    schedule,
    t_idx: int | None,
    *,
    guided: bool,
    alpha: float,
    beta: float,
    reward_center: float,
    reward_scale: float,
    x0_hat_clip_radius: float,
):
    base_log = base_logpdf(grid, weights, means, stds, schedule, t_idx)
    if not guided:
        return normalize_density(alpha * base_log, grid)
    if t_idx is None:
        q_arg = grid
    else:
        q_arg = x0_hat(grid, weights, means, stds, schedule, t_idx)
        if np.isfinite(x0_hat_clip_radius):
            q_arg = np.clip(q_arg, -x0_hat_clip_radius, x0_hat_clip_radius)
    return normalize_density(alpha * base_log + beta * reward(q_arg, reward_center, reward_scale), grid)


def metrics_between(p: np.ndarray, q: np.ndarray, grid: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    dx = np.diff(grid)
    mids = 0.5 * (grid[:-1] + grid[1:])
    p_mid = np.interp(mids, grid, p)
    q_mid = np.interp(mids, grid, q)
    p_mass = p_mid * dx
    q_mass = q_mid * dx
    p_mass /= p_mass.sum()
    q_mass /= q_mass.sum()
    m = 0.5 * (p_mass + q_mass)
    js = 0.5 * np.sum(p_mass * (np.log(p_mass + eps) - np.log(m + eps)))
    js += 0.5 * np.sum(q_mass * (np.log(q_mass + eps) - np.log(m + eps)))
    kl_pq = np.sum(p_mass * (np.log(p_mass + eps) - np.log(q_mass + eps)))

    p_cdf = np.cumsum(p_mass)
    q_cdf = np.cumsum(q_mass)
    ks = np.max(np.abs(p_cdf - q_cdf))
    w1 = np.trapezoid(np.abs(p_cdf - q_cdf), mids) if hasattr(np, "trapezoid") else np.trapz(np.abs(p_cdf - q_cdf), mids)
    return {
        "w1": float(w1),
        "ks": float(ks),
        "js": float(js),
        "kl_left_right": float(kl_pq),
    }


def density_stats(dens: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    mean = np.trapezoid(grid * dens, grid) if hasattr(np, "trapezoid") else np.trapz(grid * dens, grid)
    second = np.trapezoid(grid * grid * dens, grid) if hasattr(np, "trapezoid") else np.trapz(grid * grid * dens, grid)
    return float(mean), float(np.sqrt(max(second - mean * mean, 0.0)))


def save_metric_plot(rows: list[dict], out_dir: Path, metric: str, target_kind: str):
    schedules = sorted({row["schedule"] for row in rows})
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for schedule in schedules:
        selected = [row for row in rows if row["schedule"] == schedule and row["target_kind"] == target_kind]
        selected = sorted(selected, key=lambda r: r["pair_index"])
        ax.plot([row["pair_label"] for row in selected], [row[metric] for row in selected], marker="o", label=schedule)
    ax.set_xlabel("adjacent pair")
    ax.set_ylabel(metric)
    ax.set_title(f"{target_kind} adjacent schedule gaps: {metric}")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{target_kind}_adjacent_{metric}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    weights = normalize_weights(np.asarray(parse_float_list(args.gmm_weights), dtype=np.float64))
    means = np.asarray(parse_float_list(args.gmm_means), dtype=np.float64)
    stds = np.asarray(parse_float_list(args.gmm_stds), dtype=np.float64)
    if not (len(weights) == len(means) == len(stds)):
        raise ValueError("GMM weights, means, and stds must have the same length.")

    grid = np.linspace(args.grid_min, args.grid_max, args.grid_points, dtype=np.float64)
    schedules = [x.strip() for x in args.schedules.replace(",", " ").split() if x.strip()]
    rows = []

    for schedule_name in schedules:
        schedule = build_schedule(args.diffusion_steps, schedule_name, args.snr_max)
        for target_kind, guided in [("base", False), ("guided", True)]:
            densities = [
                target_density(
                    grid,
                    weights,
                    means,
                    stds,
                    schedule,
                    None,
                    guided=guided,
                    alpha=args.alpha,
                    beta=args.beta,
                    reward_center=args.reward_center,
                    reward_scale=args.reward_scale,
                    x0_hat_clip_radius=args.x0_hat_clip_radius,
                )
            ]
            densities.extend(
                target_density(
                    grid,
                    weights,
                    means,
                    stds,
                    schedule,
                    t,
                    guided=guided,
                    alpha=args.alpha,
                    beta=args.beta,
                    reward_center=args.reward_center,
                    reward_scale=args.reward_scale,
                    x0_hat_clip_radius=args.x0_hat_clip_radius,
                )
                for t in range(args.diffusion_steps)
            )

            for idx in range(len(densities) - 1):
                left_stage = "clean" if idx == 0 else f"t{idx - 1:03d}"
                right_stage = f"t{idx:03d}"
                metrics = metrics_between(densities[idx], densities[idx + 1], grid)
                left_mean, left_std = density_stats(densities[idx], grid)
                right_mean, right_std = density_stats(densities[idx + 1], grid)
                rows.append(
                    {
                        "schedule": schedule_name,
                        "target_kind": target_kind,
                        "pair_index": idx,
                        "pair_label": f"{left_stage}->{right_stage}",
                        "left_stage": left_stage,
                        "right_stage": right_stage,
                        "left_mean": left_mean,
                        "left_std": left_std,
                        "right_mean": right_mean,
                        "right_std": right_std,
                        "sqrt_alpha_bar_right": schedule["sqrt_alphas_cumprod"][idx]
                        if idx < args.diffusion_steps
                        else "",
                        "sigma_right": schedule["sqrt_one_minus_alphas_cumprod"][idx]
                        if idx < args.diffusion_steps
                        else "",
                        **metrics,
                    }
                )

    csv_path = args.out_dir / "adjacent_schedule_gaps.csv"
    fieldnames = [
        "schedule",
        "target_kind",
        "pair_index",
        "pair_label",
        "left_stage",
        "right_stage",
        "sqrt_alpha_bar_right",
        "sigma_right",
        "w1",
        "ks",
        "js",
        "kl_left_right",
        "left_mean",
        "left_std",
        "right_mean",
        "right_std",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    config_path = args.out_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str))

    for target_kind in ["base", "guided"]:
        for metric in ["w1", "ks", "js"]:
            save_metric_plot(rows, args.out_dir, metric, target_kind)

    print(f"wrote {csv_path}")
    print(f"wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
