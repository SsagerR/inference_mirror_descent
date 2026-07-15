#!/usr/bin/env python3
"""Compare exponential energy diffusion, pi_orig diffusion, and MALA.

This script is intentionally narrow: one job corresponds to one dimension,
one dataset seed, one sample budget, one beta, and one MALA-step setting.  The
job trains two energy-based diffusion policies on the same fixed pi_orig
dataset:

* ``exponential_energy`` uses the vanilla unnormalized
  ``exp(beta * normalized_reward)`` denoising-score-matching loss.
* ``pi_orig_diffusion`` uses the unweighted pi_orig loss, then samples without
  reward guidance or MALA.
* ``orig_energy_mala`` uses the unweighted pi_orig loss, then samples with
  reward-guided MALA.

Both are evaluated against the same toy target
``pi_orig * exp(beta * normalized_reward)``.  The exact toy density is used
only for evaluation, not for training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_RUNTIME_LOADED = False


def load_runtime() -> None:
    global _RUNTIME_LOADED
    if _RUNTIME_LOADED:
        return
    global jax, jnp, optax
    global allocate_mala_steps
    global apply_1d_target_preset, normalize_1d_weights, parse_1d_float_list
    global validate_1d_reward_params, validate_1d_sampler_params
    global make_eval_grid_1d, metrics_from_samples_1d, np_reward_1d
    global run_toy_mala_sampler_1d, save_density_plot_1d, target_density_1d
    global LearnedToyConfig, LearnedToyModel, make_actor_1d, sample_gmm_1d
    global apply_2d_target_preset, make_covariances, make_eval_grid_2d
    global metrics_from_samples_2d, np_reward_2d, parse_2d_float_list
    global parse_matrix_rows, parse_vector, run_toy_mala_sampler_2d
    global save_contour_plot_2d, target_density_grid_2d
    global validate_2d_reward_params, validate_2d_sampler_params
    global LearnedToy2DConfig, LearnedToy2DModel, make_actor_2d, sample_gmm_2d

    import jax
    import jax.numpy as jnp
    import matplotlib
    import optax

    matplotlib.use("Agg")

    from scripts.toy_mala.mala_step_schedule import allocate_mala_steps
    from scripts.toy_mala.toy_mala_1d import (
        _apply_target_preset as apply_1d_target_preset,
        _normalize_weights as normalize_1d_weights,
        _parse_float_list as parse_1d_float_list,
        _validate_reward_params as validate_1d_reward_params,
        _validate_sampler_params as validate_1d_sampler_params,
        make_eval_grid as make_eval_grid_1d,
        metrics_from_samples as metrics_from_samples_1d,
        np_reward as np_reward_1d,
        run_toy_mala_sampler as run_toy_mala_sampler_1d,
        save_density_plot as save_density_plot_1d,
        target_density as target_density_1d,
    )
    from scripts.toy_mala.toy_mala_1d_learned_diffusion import (
        LearnedToyConfig,
        LearnedToyModel,
        make_actor as make_actor_1d,
        sample_gmm as sample_gmm_1d,
    )
    from scripts.toy_mala.toy_mala_2d import (
        apply_target_preset as apply_2d_target_preset,
        make_covariances,
        make_eval_grid as make_eval_grid_2d,
        metrics_from_samples as metrics_from_samples_2d,
        np_reward as np_reward_2d,
        parse_float_list as parse_2d_float_list,
        parse_matrix_rows,
        parse_vector,
        run_toy_mala_sampler as run_toy_mala_sampler_2d,
        save_contour_plot as save_contour_plot_2d,
        target_density_grid as target_density_grid_2d,
        validate_reward_params as validate_2d_reward_params,
        validate_sampler_params as validate_2d_sampler_params,
    )
    from scripts.toy_mala.toy_mala_2d_learned_diffusion import (
        LearnedToy2DConfig,
        LearnedToy2DModel,
        make_actor as make_actor_2d,
        sample_gmm as sample_gmm_2d,
    )

    _RUNTIME_LOADED = True


def dataset_standardize_rewards(rewards: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, dict[str, float]]:
    rewards = np.asarray(rewards, dtype=np.float64)
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    normalized = (rewards - mean) / (std + eps)
    return normalized, {
        "reward_mean_raw": mean,
        "reward_std_raw": std,
        "reward_norm_mean": float(np.mean(normalized)),
        "reward_norm_std": float(np.std(normalized)),
    }


def vanilla_exponential_weights(normalized_rewards: np.ndarray, beta: float) -> np.ndarray:
    return np.exp(float(beta) * np.asarray(normalized_rewards, dtype=np.float64))


def fixed_target_grid_1d(cfg, weights, means, stds, schedule) -> np.ndarray:
    if cfg.grid_mode == "fixed":
        return np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_points, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    low = min(float(np.min(means - 6.0 * stds)), float(cfg.grid_min))
    high = max(float(np.max(means + 6.0 * stds)), float(cfg.grid_max))
    margin = max(0.05 * (high - low), 1e-3)
    return np.linspace(low - margin, high + margin, cfg.grid_points, dtype=np.float64)


def fixed_target_grid_2d(cfg, means, covs, schedule) -> tuple[np.ndarray, np.ndarray]:
    if cfg.grid_mode == "fixed":
        grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_points, dtype=np.float64)
        return grid, grid
    means = np.asarray(means, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)
    std = np.sqrt(np.maximum(np.stack([covs[:, 0, 0], covs[:, 1, 1]], axis=1), 1e-12))
    low = np.minimum(np.min(means - 5.0 * std, axis=0), np.array([cfg.grid_min, cfg.grid_min], dtype=np.float64))
    high = np.maximum(np.max(means + 5.0 * std, axis=0), np.array([cfg.grid_max, cfg.grid_max], dtype=np.float64))
    margin = np.maximum(0.05 * (high - low), 1e-3)
    grid_x = np.linspace(low[0] - margin[0], high[0] + margin[0], cfg.grid_points, dtype=np.float64)
    grid_y = np.linspace(low[1] - margin[1], high[1] + margin[1], cfg.grid_points, dtype=np.float64)
    return grid_x, grid_y


def save_panel_data_1d(path: Path, *, method: str, samples: np.ndarray, grid: np.ndarray, target: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dim=np.asarray(1, dtype=np.int32),
        method=np.asarray(method),
        samples=np.asarray(samples, dtype=np.float32),
        grid=np.asarray(grid, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
    )


def save_panel_data_2d(
    path: Path,
    *,
    method: str,
    samples: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    target: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dim=np.asarray(2, dtype=np.int32),
        method=np.asarray(method),
        samples=np.asarray(samples, dtype=np.float32),
        grid_x=np.asarray(grid_x, dtype=np.float64),
        grid_y=np.asarray(grid_y, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
    )


def weight_diagnostics(normalized_rewards: np.ndarray, beta: float) -> dict[str, float]:
    beta_rewards = float(beta) * np.asarray(normalized_rewards, dtype=np.float64)
    weights = vanilla_exponential_weights(normalized_rewards, beta)
    weight_sum = float(np.sum(weights))
    ess = (weight_sum * weight_sum) / float(np.sum(weights * weights)) if weight_sum > 0 else float("nan")
    n = float(len(weights))
    return {
        "beta_reward_min": float(np.min(beta_rewards)),
        "beta_reward_max": float(np.max(beta_rewards)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "dataset_ess": float(ess),
        "dataset_ess_over_N": float(ess / n) if n > 0 else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", choices=["1d", "2d"], required=True)
    p.add_argument(
        "--methods",
        choices=["both", "exponential_energy", "pi_orig_diffusion", "orig_energy_mala"],
        default="both",
    )
    p.add_argument("--reward_normalization", choices=["dataset_standardize"], default="dataset_standardize")
    p.add_argument("--exp_weight_mode", choices=["vanilla"], default="vanilla")
    p.add_argument("--sample_budget", type=int, required=True)
    p.add_argument("--dataset_seed", type=int, default=0)

    p.add_argument("--target_preset", default=None)
    p.add_argument("--gmm_weights", default=None)
    p.add_argument("--gmm_means", default=None)
    p.add_argument("--gmm_stds", default=None)
    p.add_argument("--gmm_corrs", default=None)
    p.add_argument("--reward_type", choices=["quadratic", "bumps", "rugged"], default=None)
    p.add_argument("--reward_center", default=None)
    p.add_argument("--reward_scale", type=float, default=None)
    p.add_argument("--reward_scales", default=None)
    p.add_argument("--reward_bump_centers", default=None)
    p.add_argument("--reward_bump_widths", default=None)
    p.add_argument("--reward_bump_weights", default=None)
    p.add_argument("--reward_sin_amp", type=float, default=None)
    p.add_argument("--reward_sin_freq", type=float, default=None)
    p.add_argument("--reward_sin_freqs", default=None)
    p.add_argument("--reward_sin_phase", type=float, default=None)
    p.add_argument("--reward_l2", type=float, default=None)
    p.add_argument("--reward_l4", type=float, default=None)

    p.add_argument("--num_samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diffusion_steps", type=int, default=20)
    p.add_argument("--beta_schedule_type", choices=["linear", "cosine", "constant_kl"], default="cosine")
    p.add_argument("--snr_max", type=float, default=124.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--guidance_schedule", default="constant")
    p.add_argument("--sampler", default="mala")
    p.add_argument("--mala_steps", type=int, default=4)
    p.add_argument("--mala_budget", type=int, default=None)
    p.add_argument("--mala_step_schedule", default="constant")
    p.add_argument("--mala_eta", type=float, default=1.0)
    p.add_argument("--langevin_steps", type=int, default=4)
    p.add_argument("--langevin_eta", type=float, default=1.0)
    p.add_argument("--denoising_predictor", choices=["Identity", "DDPM_mean", "DDIM"], default="DDPM_mean")
    p.add_argument("--denoising_schedule", default="from_predictor")
    p.add_argument("--guidance_gradient_space", choices=["xt", "x0hat", "x0hatclipped"], default="x0hat")
    p.add_argument("--x0_hat_method", choices=["tweedie"], default="tweedie")
    p.add_argument("--x0_hat_clip_radius", type=float, default=1_000_000.0)
    p.add_argument("--x_recon_clip_radius", type=float, default=1_000_000.0)
    p.add_argument("--action_clip_radius", type=float, default=1_000_000.0)
    p.add_argument("--mala_adapt_rate", type=float, default=0.0)
    p.add_argument("--guidance_strength_multiplier", type=float, default=1.0)
    p.add_argument("--batch_independent_guidance", action="store_true")
    p.add_argument("--advantage_normalization", action="store_true")
    p.add_argument("--initial_advantage_second_moment_ema", type=float, default=1.0)

    p.add_argument("--grid_mode", choices=["auto", "fixed"], default="auto")
    p.add_argument("--grid_min", type=float, default=None)
    p.add_argument("--grid_max", type=float, default=None)
    p.add_argument("--grid_points", type=int, default=None)
    p.add_argument("--metric_target_samples", type=int, default=20000)
    p.add_argument("--metric_slices", type=int, default=64)
    p.add_argument("--eval_timesteps", default="auto")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--hidden_num", type=int, default=3)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--diffusion_hidden_dim", type=int, default=256)
    p.add_argument("--policy_parameterization", choices=["E", "f"], default="E")
    p.add_argument("--policy_final_layer", choices=["default", "ff", "L2", "IP"], default="ff")
    p.add_argument("--train_steps", type=int, default=20000)
    p.add_argument("--train_batch_size", type=int, default=4096)
    p.add_argument("--train_lr", type=float, default=3e-4)
    p.add_argument("--train_log_every", type=int, default=500)
    p.add_argument("--diagnostic_batch_size", type=int, default=20000)
    return p.parse_args()


def _default(value, fallback):
    return fallback if value is None else value


def make_1d_cfg(args: argparse.Namespace, *, beta_for_sampler: float, sampler: str, output_dir: str) -> tuple[LearnedToyConfig, np.ndarray, np.ndarray, np.ndarray]:
    load_runtime()
    ns = argparse.Namespace(
        target_preset=_default(args.target_preset, "complex_1d_v2"),
        gmm_weights=_default(args.gmm_weights, "0.35,0.30,0.35"),
        gmm_means=_default(args.gmm_means, "-0.75,0.0,0.75"),
        gmm_stds=_default(args.gmm_stds, "0.10,0.16,0.10"),
        reward_type=_default(args.reward_type, "quadratic"),
        reward_center=float(_default(args.reward_center, 0.45)),
        reward_scale=float(_default(args.reward_scale, 1.0)),
        reward_bump_centers=_default(args.reward_bump_centers, "-0.72,-0.28,0.18,0.64,0.92"),
        reward_bump_widths=_default(args.reward_bump_widths, "0.055,0.09,0.06,0.12,0.045"),
        reward_bump_weights=_default(args.reward_bump_weights, "0.85,-0.45,0.75,1.10,-0.35"),
        reward_sin_amp=float(_default(args.reward_sin_amp, 0.12)),
        reward_sin_freq=float(_default(args.reward_sin_freq, 18.0)),
        reward_sin_phase=float(_default(args.reward_sin_phase, 0.4)),
        reward_l2=float(_default(args.reward_l2, 0.08)),
        reward_l4=float(_default(args.reward_l4, 0.02)),
    )
    ns = apply_1d_target_preset(ns)
    weights = normalize_1d_weights(parse_1d_float_list(ns.gmm_weights))
    means = np.asarray(parse_1d_float_list(ns.gmm_means), dtype=np.float64)
    stds = np.asarray(parse_1d_float_list(ns.gmm_stds), dtype=np.float64)
    reward_bump_centers = parse_1d_float_list(ns.reward_bump_centers)
    reward_bump_widths = parse_1d_float_list(ns.reward_bump_widths)
    reward_bump_weights = parse_1d_float_list(ns.reward_bump_weights)
    cfg = LearnedToyConfig(
        score_source="learned",
        target_preset=ns.target_preset,
        gmm_weights=weights.tolist(),
        gmm_means=means.tolist(),
        gmm_stds=stds.tolist(),
        reward_type=ns.reward_type,
        reward_center=float(ns.reward_center),
        reward_scale=float(ns.reward_scale),
        reward_bump_centers=reward_bump_centers,
        reward_bump_widths=reward_bump_widths,
        reward_bump_weights=reward_bump_weights,
        reward_sin_amp=float(ns.reward_sin_amp),
        reward_sin_freq=float(ns.reward_sin_freq),
        reward_sin_phase=float(ns.reward_sin_phase),
        reward_l2=float(ns.reward_l2),
        reward_l4=float(ns.reward_l4),
        num_samples=args.num_samples,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        beta_schedule_type=args.beta_schedule_type,
        snr_max=args.snr_max,
        alpha=args.alpha,
        beta=float(beta_for_sampler),
        guidance_schedule=args.guidance_schedule,
        sampler=sampler,
        mala_steps=args.mala_steps,
        mala_budget=args.mala_budget if args.mala_budget is not None else args.mala_steps * args.diffusion_steps,
        mala_step_schedule=args.mala_step_schedule,
        mala_steps_per_level=[],
        mala_eta=args.mala_eta,
        langevin_steps=args.langevin_steps,
        langevin_eta=args.langevin_eta,
        denoising_predictor=args.denoising_predictor,
        denoising_schedule=args.denoising_schedule,
        guidance_gradient_space=args.guidance_gradient_space,
        x0_hat_method=args.x0_hat_method,
        x0_hat_clip_radius=args.x0_hat_clip_radius,
        x_recon_clip_radius=args.x_recon_clip_radius,
        action_clip_radius=args.action_clip_radius,
        mala_adapt_rate=args.mala_adapt_rate,
        guidance_strength_multiplier=args.guidance_strength_multiplier,
        batch_independent_guidance=args.batch_independent_guidance,
        advantage_normalization=args.advantage_normalization,
        initial_advantage_second_moment_ema=args.initial_advantage_second_moment_ema,
        grid_mode=args.grid_mode,
        grid_min=float(_default(args.grid_min, -6.0)),
        grid_max=float(_default(args.grid_max, 6.0)),
        grid_points=int(_default(args.grid_points, 2001)),
        eval_timesteps=args.eval_timesteps,
        output_dir=output_dir,
        hidden_num=args.hidden_num,
        hidden_dim=args.hidden_dim,
        diffusion_hidden_dim=args.diffusion_hidden_dim,
        policy_parameterization=args.policy_parameterization,
        policy_final_layer=args.policy_final_layer,
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_log_every=args.train_log_every,
        diagnostic_batch_size=args.diagnostic_batch_size,
    )
    validate_1d_reward_params(cfg)
    validate_1d_sampler_params(cfg)
    return cfg, weights, means, stds


def make_2d_cfg(args: argparse.Namespace, *, beta_for_sampler: float, sampler: str, output_dir: str) -> tuple[LearnedToy2DConfig, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    load_runtime()
    ns = argparse.Namespace(
        target_preset=_default(args.target_preset, "complex_2d_v2"),
        gmm_weights=_default(args.gmm_weights, "0.18,0.37,0.25,0.20"),
        gmm_means=_default(args.gmm_means, "-0.85,-0.45;-0.15,0.18;0.10,-0.05;0.82,0.52"),
        gmm_stds=_default(args.gmm_stds, "0.09,0.14;0.06,0.10;0.16,0.07;0.08,0.11"),
        gmm_corrs=_default(args.gmm_corrs, "0.20,-0.35,0.40,-0.20"),
        reward_type=_default(args.reward_type, "quadratic"),
        reward_center=_default(args.reward_center, "0.65,0.35"),
        reward_scales=_default(args.reward_scales, "1.0,1.0"),
        reward_bump_centers=_default(args.reward_bump_centers, "0.55,0.28;-0.25,0.20;0.82,0.58"),
        reward_bump_widths=_default(args.reward_bump_widths, "0.16,0.14;0.20,0.12;0.12,0.16"),
        reward_bump_weights=_default(args.reward_bump_weights, "1.2,-0.7,0.8"),
        reward_sin_amp=float(_default(args.reward_sin_amp, 0.10)),
        reward_sin_freqs=_default(args.reward_sin_freqs, "8.0,11.0"),
        reward_sin_phase=float(_default(args.reward_sin_phase, 0.2)),
        reward_l2=float(_default(args.reward_l2, 0.05)),
        reward_l4=float(_default(args.reward_l4, 0.006)),
    )
    ns = apply_2d_target_preset(ns)
    weights = np.asarray(parse_2d_float_list(ns.gmm_weights), dtype=np.float64)
    weights = weights / weights.sum()
    means = np.asarray(parse_matrix_rows(ns.gmm_means), dtype=np.float64)
    stds = np.asarray(parse_matrix_rows(ns.gmm_stds), dtype=np.float64)
    corrs = np.asarray(parse_2d_float_list(ns.gmm_corrs), dtype=np.float64)
    covs = make_covariances(stds, corrs)
    reward_center = parse_vector(ns.reward_center)
    reward_scales = parse_vector(ns.reward_scales)
    reward_bump_centers = parse_matrix_rows(ns.reward_bump_centers)
    reward_bump_widths = parse_matrix_rows(ns.reward_bump_widths)
    reward_bump_weights = parse_2d_float_list(ns.reward_bump_weights)
    reward_sin_freqs = parse_vector(ns.reward_sin_freqs)
    cfg = LearnedToy2DConfig(
        score_source="learned",
        target_preset=ns.target_preset,
        gmm_weights=weights.tolist(),
        gmm_means=means.tolist(),
        gmm_stds=stds.tolist(),
        gmm_corrs=corrs.tolist(),
        reward_type=ns.reward_type,
        reward_center=reward_center,
        reward_scales=reward_scales,
        reward_bump_centers=reward_bump_centers,
        reward_bump_widths=reward_bump_widths,
        reward_bump_weights=reward_bump_weights,
        reward_sin_amp=float(ns.reward_sin_amp),
        reward_sin_freqs=reward_sin_freqs,
        reward_sin_phase=float(ns.reward_sin_phase),
        reward_l2=float(ns.reward_l2),
        reward_l4=float(ns.reward_l4),
        num_samples=args.num_samples,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        beta_schedule_type=args.beta_schedule_type,
        snr_max=args.snr_max,
        alpha=args.alpha,
        beta=float(beta_for_sampler),
        guidance_schedule=args.guidance_schedule,
        sampler=sampler,
        mala_steps=args.mala_steps,
        mala_budget=args.mala_budget if args.mala_budget is not None else args.mala_steps * args.diffusion_steps,
        mala_step_schedule=args.mala_step_schedule,
        mala_steps_per_level=[],
        mala_eta=args.mala_eta,
        langevin_steps=args.langevin_steps,
        langevin_eta=args.langevin_eta,
        denoising_predictor=args.denoising_predictor,
        denoising_schedule=args.denoising_schedule,
        guidance_gradient_space=args.guidance_gradient_space,
        x0_hat_method=args.x0_hat_method,
        x0_hat_clip_radius=args.x0_hat_clip_radius,
        x_recon_clip_radius=args.x_recon_clip_radius,
        action_clip_radius=args.action_clip_radius,
        mala_adapt_rate=args.mala_adapt_rate,
        guidance_strength_multiplier=args.guidance_strength_multiplier,
        batch_independent_guidance=args.batch_independent_guidance,
        advantage_normalization=args.advantage_normalization,
        initial_advantage_second_moment_ema=args.initial_advantage_second_moment_ema,
        grid_mode=args.grid_mode,
        grid_min=float(_default(args.grid_min, -3.0)),
        grid_max=float(_default(args.grid_max, 3.0)),
        grid_points=int(_default(args.grid_points, 181)),
        metric_target_samples=args.metric_target_samples,
        metric_slices=args.metric_slices,
        eval_timesteps=args.eval_timesteps,
        output_dir=output_dir,
        hidden_num=args.hidden_num,
        hidden_dim=args.hidden_dim,
        diffusion_hidden_dim=args.diffusion_hidden_dim,
        policy_parameterization=args.policy_parameterization,
        policy_final_layer=args.policy_final_layer,
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_log_every=args.train_log_every,
        diagnostic_batch_size=args.diagnostic_batch_size,
    )
    validate_2d_reward_params(cfg)
    validate_2d_sampler_params(cfg)
    return cfg, weights, means, covs, corrs


def with_mala_steps_per_level(actor: ActorCritic, cfg):
    load_runtime()
    steps = allocate_mala_steps(
        cfg.mala_step_schedule,
        diffusion_steps=cfg.diffusion_steps,
        mala_budget=cfg.mala_budget,
        sqrt_alphas_cumprod=np.asarray(actor.schedule.sqrt_alphas_cumprod),
        mala_steps=cfg.mala_steps,
    )
    return replace(cfg, mala_budget=int(sum(steps)), mala_steps_per_level=list(steps))


def canonical_exponential_sampling_cfg(cfg):
    """Return the reward-free denoising-only cfg used by exponential_energy.

    The exponential baseline puts reward into the diffusion training loss, so
    sampling must not depend on any MALA/guidance controls.  Keeping a canonical
    cfg also prevents downstream summaries from treating exponential rows as
    different ``mala_steps`` settings.
    """
    return replace(
        cfg,
        sampler="unguided",
        beta=0.0,
        mala_steps=0,
        mala_budget=0,
        mala_steps_per_level=[0] * int(cfg.diffusion_steps),
        denoising_predictor="DDIM",
        denoising_schedule="from_predictor",
    )


def canonical_pi_orig_diffusion_sampling_cfg(cfg):
    """Return the reward-free denoising-only cfg for pi_orig diffusion.

    This is intentionally identical to the exponential method's sampling
    configuration.  The difference between the two methods is only the training
    loss: exponential uses reward weights; pi_orig_diffusion does not.
    """
    return canonical_exponential_sampling_cfg(cfg)


def train_policy_on_dataset(
    actor: ActorCritic,
    cfg,
    dataset: np.ndarray,
    *,
    normalized_rewards: np.ndarray | None,
    beta: float,
    seed_offset: int,
) -> tuple[object, np.ndarray, np.ndarray]:
    load_runtime()
    x0_data = jnp.asarray(dataset, dtype=jnp.float32)
    if normalized_rewards is None:
        weights = None
    else:
        weights = jnp.asarray(vanilla_exponential_weights(normalized_rewards, beta), dtype=jnp.float32)

    opt = optax.adam(cfg.train_lr)
    init_key, loop_key = jax.random.split(jax.random.key(cfg.seed + seed_offset))
    params = actor.init_params(init_key).policy
    opt_state = opt.init(params)
    n = int(dataset.shape[0])

    def loss_fn(policy_params, key):
        key_idx, key_t, key_noise = jax.random.split(key, 3)
        idx = jax.random.randint(key_idx, (cfg.train_batch_size,), 0, n)
        x0 = x0_data[idx]
        t = jax.random.randint(key_t, (cfg.train_batch_size,), 0, cfg.diffusion_steps)
        noise = jax.random.normal(key_noise, x0.shape)
        x_t = actor.q_sample(t, x0, noise)
        obs = jnp.zeros((cfg.train_batch_size, 1), dtype=x0.dtype)
        pred = actor.eps_pred(policy_params, obs, x_t, t)
        per_item = jnp.mean(optax.squared_error(pred, noise), axis=-1)
        if weights is not None:
            per_item = weights[idx] * per_item
        return jnp.mean(per_item)

    @jax.jit
    def train_step(policy_params, opt_state, key):
        loss, grads = jax.value_and_grad(loss_fn)(policy_params, key)
        updates, opt_state = opt.update(grads, opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, updates)
        return policy_params, opt_state, loss

    steps = []
    losses = []
    for step in range(1, cfg.train_steps + 1):
        loop_key, step_key = jax.random.split(loop_key)
        params, opt_state, loss = train_step(params, opt_state, step_key)
        if step == 1 or step == cfg.train_steps or step % cfg.train_log_every == 0:
            steps.append(step)
            losses.append(float(loss))
            print(f"train step {step:>7d}/{cfg.train_steps}: loss={float(loss):.6g}", flush=True)
    return params, np.asarray(steps, dtype=np.int32), np.asarray(losses, dtype=np.float64)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def add_common(row: dict, *, args, method: str, reward_stats: dict, weight_stats: dict, train_loss: float, cfg_sample=None) -> dict:
    out = dict(row)
    out.update({
        "dim": args.dim,
        "method": method,
        "dataset_seed": args.dataset_seed,
        "sample_budget": args.sample_budget,
        "beta_input": args.beta,
        "mala_steps": getattr(cfg_sample, "mala_steps", args.mala_steps),
        "mala_budget": getattr(cfg_sample, "mala_budget", args.mala_budget if args.mala_budget is not None else args.mala_steps * args.diffusion_steps),
        "train_loss_final": train_loss,
    })
    out.update(reward_stats)
    out.update(weight_stats)
    return out


def run_1d(args: argparse.Namespace, out: Path) -> list[dict]:
    load_runtime()
    cfg_base, weights, means, stds = make_1d_cfg(args, beta_for_sampler=args.beta, sampler="mala", output_dir=str(out))
    key_data = jax.random.key(args.seed + 100_003 * (args.dataset_seed + 1))
    dataset = np.asarray(sample_gmm_1d(
        key_data,
        jnp.asarray(weights, dtype=jnp.float32),
        jnp.asarray(means, dtype=jnp.float32),
        jnp.asarray(stds, dtype=jnp.float32),
        args.sample_budget,
    ))
    raw_rewards = np_reward_1d(dataset[:, 0], cfg_base.reward_center, cfg_base.reward_scale, cfg_base)
    norm_rewards, reward_stats = dataset_standardize_rewards(raw_rewards)
    weight_stats = weight_diagnostics(norm_rewards, args.beta)
    beta_sampler = args.beta / (reward_stats["reward_std_raw"] + 1e-8)

    cfg_eval, _w, _m, _s = make_1d_cfg(args, beta_for_sampler=beta_sampler, sampler="mala", output_dir=str(out))
    rows = []

    def evaluate(method: str, policy_params, cfg_sample, train_losses):
        if method in {"exponential_energy", "pi_orig_diffusion"}:
            cfg_sample = canonical_exponential_sampling_cfg(cfg_sample)
        actor = make_actor_1d(cfg_sample)
        if method == "orig_energy_mala":
            cfg_sample = with_mala_steps_per_level(actor, cfg_sample)
        model = LearnedToyModel(
            actor=actor,
            policy_params=policy_params,
            reward_center=cfg_sample.reward_center,
            reward_scale=cfg_sample.reward_scale,
            schedule=actor.schedule,
            num_timesteps=cfg_sample.diffusion_steps,
            mala_steps=cfg_sample.mala_steps,
            x_recon_clip_radius=cfg_sample.x_recon_clip_radius,
        )
        sample_seed_offset = {"exponential_energy": 0, "pi_orig_diffusion": 313, "orig_energy_mala": 999}[method]
        result = run_toy_mala_sampler_1d(jax.random.key(args.seed + 404 + sample_seed_offset), model, cfg_sample)
        raw_x0 = np.asarray(result.raw_x0[:, 0])
        grid = make_eval_grid_1d(raw_x0, cfg_eval, weights, means, stds, actor.schedule, None)
        target = target_density_1d(grid, cfg_eval, weights, means, stds, actor.schedule, None)
        metrics = metrics_from_samples_1d(raw_x0, grid, target, cfg_eval)
        metrics.update({
            "stage": "final_clean",
            "timestep": -1,
            "acceptance_rate": float(np.nanmean(np.asarray(result.per_level_acc))),
        })
        plot_name = {
            "exponential_energy": "final_clean_exponential.png",
            "pi_orig_diffusion": "final_clean_pi_orig_diffusion.png",
            "orig_energy_mala": "final_clean_mala.png",
        }[method]
        save_density_plot_1d(out / "plots" / plot_name, raw_x0, grid, target, f"{method} vs normalized target")
        panel_grid = fixed_target_grid_1d(cfg_eval, weights, means, stds, actor.schedule)
        panel_target = target_density_1d(panel_grid, cfg_eval, weights, means, stds, actor.schedule, None)
        save_panel_data_1d(
            out / "panel_data" / f"{method}.npz",
            method=method,
            samples=raw_x0,
            grid=panel_grid,
            target=panel_target,
        )
        return add_common(metrics, args=args, method=method, reward_stats=reward_stats, weight_stats=weight_stats, train_loss=float(train_losses[-1]), cfg_sample=cfg_sample)

    if args.methods in {"both", "exponential_energy"}:
        cfg_exp = canonical_exponential_sampling_cfg(cfg_base)
        actor = make_actor_1d(cfg_exp)
        params, train_steps, train_losses = train_policy_on_dataset(
            actor, cfg_exp, dataset, normalized_rewards=norm_rewards, beta=args.beta, seed_offset=17
        )
        rows.append(evaluate("exponential_energy", params, cfg_exp, train_losses))
        np.savetxt(out / "training_loss_exponential.csv", np.column_stack([train_steps, train_losses]), delimiter=",", header="step,loss", comments="")

    if args.methods in {"both", "orig_energy_mala"}:
        actor = make_actor_1d(cfg_base)
        params, train_steps, train_losses = train_policy_on_dataset(
            actor, cfg_base, dataset, normalized_rewards=None, beta=args.beta, seed_offset=29
        )
        cfg_mala = replace(cfg_base, sampler="mala", beta=beta_sampler)
        rows.append(evaluate("orig_energy_mala", params, cfg_mala, train_losses))
        np.savetxt(out / "training_loss_mala.csv", np.column_stack([train_steps, train_losses]), delimiter=",", header="step,loss", comments="")

    if args.methods == "pi_orig_diffusion":
        cfg_orig = canonical_pi_orig_diffusion_sampling_cfg(cfg_base)
        actor = make_actor_1d(cfg_orig)
        params, train_steps, train_losses = train_policy_on_dataset(
            actor, cfg_orig, dataset, normalized_rewards=None, beta=args.beta, seed_offset=31
        )
        rows.append(evaluate("pi_orig_diffusion", params, cfg_orig, train_losses))
        np.savetxt(out / "training_loss_pi_orig_diffusion.csv", np.column_stack([train_steps, train_losses]), delimiter=",", header="step,loss", comments="")

    return rows


def run_2d(args: argparse.Namespace, out: Path) -> list[dict]:
    load_runtime()
    cfg_base, weights, means, covs, _corrs = make_2d_cfg(args, beta_for_sampler=args.beta, sampler="mala", output_dir=str(out))
    key_data = jax.random.key(args.seed + 100_003 * (args.dataset_seed + 1))
    dataset = np.asarray(sample_gmm_2d(
        key_data,
        jnp.asarray(weights, dtype=jnp.float32),
        jnp.asarray(means, dtype=jnp.float32),
        jnp.asarray(covs, dtype=jnp.float32),
        args.sample_budget,
    ))
    raw_rewards = np_reward_2d(dataset, cfg_base)
    norm_rewards, reward_stats = dataset_standardize_rewards(raw_rewards)
    weight_stats = weight_diagnostics(norm_rewards, args.beta)
    beta_sampler = args.beta / (reward_stats["reward_std_raw"] + 1e-8)

    cfg_eval, _w, _m, _c, _corrs2 = make_2d_cfg(args, beta_for_sampler=beta_sampler, sampler="mala", output_dir=str(out))
    rows = []

    def evaluate(method: str, policy_params, cfg_sample, train_losses):
        if method in {"exponential_energy", "pi_orig_diffusion"}:
            cfg_sample = canonical_exponential_sampling_cfg(cfg_sample)
        actor = make_actor_2d(cfg_sample)
        if method == "orig_energy_mala":
            cfg_sample = with_mala_steps_per_level(actor, cfg_sample)
        model = LearnedToy2DModel(
            actor=actor,
            policy_params=policy_params,
            reward_type=cfg_sample.reward_type,
            reward_center=jnp.asarray(cfg_sample.reward_center, dtype=jnp.float32),
            reward_scales=jnp.asarray(cfg_sample.reward_scales, dtype=jnp.float32),
            reward_bump_centers=jnp.asarray(cfg_sample.reward_bump_centers, dtype=jnp.float32),
            reward_bump_widths=jnp.asarray(cfg_sample.reward_bump_widths, dtype=jnp.float32),
            reward_bump_weights=jnp.asarray(cfg_sample.reward_bump_weights, dtype=jnp.float32),
            reward_sin_amp=float(cfg_sample.reward_sin_amp),
            reward_sin_freqs=jnp.asarray(cfg_sample.reward_sin_freqs, dtype=jnp.float32),
            reward_sin_phase=float(cfg_sample.reward_sin_phase),
            reward_l2=float(cfg_sample.reward_l2),
            reward_l4=float(cfg_sample.reward_l4),
            schedule=actor.schedule,
            num_timesteps=cfg_sample.diffusion_steps,
            mala_steps=cfg_sample.mala_steps,
            x_recon_clip_radius=float(cfg_sample.x_recon_clip_radius),
        )
        sample_seed_offset = {"exponential_energy": 0, "pi_orig_diffusion": 313, "orig_energy_mala": 999}[method]
        result = run_toy_mala_sampler_2d(jax.random.key(args.seed + 404 + sample_seed_offset), model, cfg_sample)
        raw_x0 = np.asarray(result.raw_x0)
        grid_x, grid_y = make_eval_grid_2d(raw_x0, cfg_eval, means, covs, actor.schedule, None)
        target = target_density_grid_2d(grid_x, grid_y, cfg_eval, weights, means, covs, actor.schedule, None)
        rng = np.random.default_rng(args.seed + 2029)
        metrics = metrics_from_samples_2d(raw_x0, grid_x, grid_y, target, cfg_eval, rng)
        metrics.update({
            "stage": "final_clean",
            "timestep": -1,
            "acceptance_rate": float(np.nanmean(np.asarray(result.per_level_acc))),
        })
        plot_name = {
            "exponential_energy": "final_clean_exponential.png",
            "pi_orig_diffusion": "final_clean_pi_orig_diffusion.png",
            "orig_energy_mala": "final_clean_mala.png",
        }[method]
        save_contour_plot_2d(out / "plots" / plot_name, raw_x0, grid_x, grid_y, target, f"{method} vs normalized target")
        panel_grid_x, panel_grid_y = fixed_target_grid_2d(cfg_eval, means, covs, actor.schedule)
        panel_target = target_density_grid_2d(panel_grid_x, panel_grid_y, cfg_eval, weights, means, covs, actor.schedule, None)
        save_panel_data_2d(
            out / "panel_data" / f"{method}.npz",
            method=method,
            samples=raw_x0,
            grid_x=panel_grid_x,
            grid_y=panel_grid_y,
            target=panel_target,
        )
        return add_common(metrics, args=args, method=method, reward_stats=reward_stats, weight_stats=weight_stats, train_loss=float(train_losses[-1]), cfg_sample=cfg_sample)

    if args.methods in {"both", "exponential_energy"}:
        cfg_exp = canonical_exponential_sampling_cfg(cfg_base)
        actor = make_actor_2d(cfg_exp)
        params, train_steps, train_losses = train_policy_on_dataset(
            actor, cfg_exp, dataset, normalized_rewards=norm_rewards, beta=args.beta, seed_offset=17
        )
        rows.append(evaluate("exponential_energy", params, cfg_exp, train_losses))
        np.savetxt(out / "training_loss_exponential.csv", np.column_stack([train_steps, train_losses]), delimiter=",", header="step,loss", comments="")

    if args.methods in {"both", "orig_energy_mala"}:
        actor = make_actor_2d(cfg_base)
        params, train_steps, train_losses = train_policy_on_dataset(
            actor, cfg_base, dataset, normalized_rewards=None, beta=args.beta, seed_offset=29
        )
        cfg_mala = replace(cfg_base, sampler="mala", beta=beta_sampler)
        rows.append(evaluate("orig_energy_mala", params, cfg_mala, train_losses))
        np.savetxt(out / "training_loss_mala.csv", np.column_stack([train_steps, train_losses]), delimiter=",", header="step,loss", comments="")

    if args.methods == "pi_orig_diffusion":
        cfg_orig = canonical_pi_orig_diffusion_sampling_cfg(cfg_base)
        actor = make_actor_2d(cfg_orig)
        params, train_steps, train_losses = train_policy_on_dataset(
            actor, cfg_orig, dataset, normalized_rewards=None, beta=args.beta, seed_offset=31
        )
        rows.append(evaluate("pi_orig_diffusion", params, cfg_orig, train_losses))
        np.savetxt(out / "training_loss_pi_orig_diffusion.csv", np.column_stack([train_steps, train_losses]), delimiter=",", header="step,loss", comments="")

    return rows


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    config = vars(args).copy()
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    if args.dim == "1d":
        rows = run_1d(args, out)
    else:
        rows = run_2d(args, out)

    write_csv(out / "metrics.csv", rows)
    write_csv(out / "weight_diagnostics.csv", [
        {k: row.get(k, "") for k in row if k.startswith("reward_") or k.startswith("beta_reward") or k.startswith("weight_") or k.startswith("dataset_")}
        for row in rows[:1]
    ])
    np.savez_compressed(out / "run_summary.npz", metrics_rows=np.asarray(len(rows), dtype=np.int32))
    print(f"wrote results to {out}")
    for row in rows:
        summary_metric = row.get("w1", row.get("sliced_w1", float("nan")))
        print(
            f"{row['method']:>22s} JS={row.get('js', float('nan')):.5f} "
            f"KL={row.get('kl_sample_target', float('nan')):.5f} "
            f"W={summary_metric:.5f} acc={row.get('acceptance_rate', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
