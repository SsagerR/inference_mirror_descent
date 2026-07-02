#!/usr/bin/env python3
"""2D oracle toy experiment for MALA-guided diffusion sampling.

This is the 2D analogue of ``toy_mala_1d.py``. It keeps the sampler structure
close to the production MALA sampler, but replaces learned policy/Q networks
with a closed-form Gaussian-mixture base distribution and a hand-written
reward.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from relax.utils.diffusion import build_beta_schedule
from scripts.toy_mala.guidance_schedule import GUIDANCE_SCHEDULE_CHOICES, guidance_schedule_multiplier
from scripts.toy_mala.mala_step_schedule import MALA_STEP_SCHEDULE_CHOICES, allocate_mala_steps


@dataclass(frozen=True)
class Toy2DConfig:
    target_preset: str
    gmm_weights: list[float]
    gmm_means: list[list[float]]
    gmm_stds: list[list[float]]
    gmm_corrs: list[float]
    reward_type: str
    reward_center: list[float]
    reward_scales: list[float]
    reward_bump_centers: list[list[float]]
    reward_bump_widths: list[list[float]]
    reward_bump_weights: list[float]
    reward_sin_amp: float
    reward_sin_freqs: list[float]
    reward_sin_phase: float
    reward_l2: float
    reward_l4: float
    num_samples: int
    seed: int
    diffusion_steps: int
    beta_schedule_type: str
    snr_max: float
    alpha: float
    beta: float
    guidance_schedule: str
    sampler: str
    mala_steps: int
    mala_budget: int
    mala_step_schedule: str
    mala_steps_per_level: list[int]
    mala_eta: float
    langevin_steps: int
    langevin_eta: float
    denoising_predictor: str
    guidance_gradient_space: str
    x0_hat_method: str
    x0_hat_clip_radius: float
    x_recon_clip_radius: float
    mala_adapt_rate: float
    guidance_strength_multiplier: float
    batch_independent_guidance: bool
    advantage_normalization: bool
    initial_advantage_second_moment_ema: float
    grid_mode: str
    grid_min: float
    grid_max: float
    grid_points: int
    metric_target_samples: int
    metric_slices: int
    eval_timesteps: str
    output_dir: str


class Toy2DSamplerResult(NamedTuple):
    action: jax.Array
    raw_x0: jax.Array
    trace: jax.Array
    log_eta_scales: jax.Array
    per_level_acc: jax.Array
    per_level_clip: jax.Array


@dataclass
class Toy2DModel:
    weights: jax.Array
    means: jax.Array
    covs: jax.Array
    reward_type: str
    reward_center: jax.Array
    reward_scales: jax.Array
    reward_bump_centers: jax.Array
    reward_bump_widths: jax.Array
    reward_bump_weights: jax.Array
    reward_sin_amp: float
    reward_sin_freqs: jax.Array
    reward_sin_phase: float
    reward_l2: float
    reward_l4: float
    schedule: object
    num_timesteps: int
    mala_steps: int
    x_recon_clip_radius: float
    act_dim: int = 2

    def _component_params(self, t_idx):
        c = self.schedule.sqrt_alphas_cumprod[t_idx]
        sigma = self.schedule.sqrt_one_minus_alphas_cumprod[t_idx]
        loc = c * self.means
        eye = jnp.eye(self.act_dim, dtype=self.covs.dtype)
        cov = (c * c) * self.covs + (sigma * sigma) * eye[None, :, :]
        inv_cov = jnp.linalg.inv(cov)
        _, logdet = jnp.linalg.slogdet(cov)
        return loc, cov, inv_cov, logdet

    def _log_base_density(self, x, t_idx):
        loc, _cov, inv_cov, logdet = self._component_params(t_idx)
        diff = x[:, None, :] - loc[None, :, :]
        quad = jnp.einsum("nki,kij,nkj->nk", diff, inv_cov, diff)
        log_comp = jnp.log(self.weights)[None, :] - 0.5 * (
            self.act_dim * jnp.log(2.0 * jnp.pi) + logdet[None, :] + quad
        )
        return jax.nn.logsumexp(log_comp, axis=-1)

    def _responsibilities(self, x, t_idx):
        loc, _cov, inv_cov, logdet = self._component_params(t_idx)
        diff = x[:, None, :] - loc[None, :, :]
        quad = jnp.einsum("nki,kij,nkj->nk", diff, inv_cov, diff)
        log_comp = jnp.log(self.weights)[None, :] - 0.5 * (
            self.act_dim * jnp.log(2.0 * jnp.pi) + logdet[None, :] + quad
        )
        return jax.nn.softmax(log_comp, axis=-1), loc, inv_cov, diff

    def _score_base(self, x, t_idx):
        resp, _loc, inv_cov, diff = self._responsibilities(x, t_idx)
        component_scores = -jnp.einsum("kij,nkj->nki", inv_cov, diff)
        return jnp.sum(resp[:, :, None] * component_scores, axis=1)

    def x0_hat_from_xt(self, act, t_idx):
        resp, loc, inv_cov_t, diff = self._responsibilities(act, t_idx)
        c = self.schedule.sqrt_alphas_cumprod[t_idx]
        gains = c * jnp.einsum("kij,kjl->kil", self.covs, inv_cov_t)
        cond_mean = self.means[None, :, :] + jnp.einsum("kij,nkj->nki", gains, diff)
        return jnp.sum(resp[:, :, None] * cond_mean, axis=1)

    def energy_fn(self, params, obs, act, t_idx):
        del params, obs
        return -self._log_base_density(act, t_idx)

    def eps_pred(self, params, obs, act, t_idx):
        del params, obs
        sigma = self.schedule.sqrt_one_minus_alphas_cumprod[t_idx]
        return -sigma * self._score_base(act, t_idx)

    def q(self, params, obs, act):
        del params, obs
        return jax_reward(
            act,
            self.reward_type,
            self.reward_center,
            self.reward_scales,
            self.reward_bump_centers,
            self.reward_bump_widths,
            self.reward_bump_weights,
            self.reward_sin_amp,
            self.reward_sin_freqs,
            self.reward_sin_phase,
            self.reward_l2,
            self.reward_l4,
        )


def parse_vector(text: str, length: int = 2) -> list[float]:
    vals = [float(x) for x in text.replace(",", " ").split()]
    if len(vals) != length:
        raise ValueError(f"Expected {length} values, got {vals}.")
    return vals


def parse_matrix_rows(text: str, row_len: int = 2) -> list[list[float]]:
    rows = []
    for row in text.split(";"):
        row = row.strip()
        if row:
            rows.append(parse_vector(row, row_len))
    if not rows:
        raise ValueError("Expected at least one row.")
    return rows


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.replace(",", " ").split()]


def apply_target_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.target_preset == "manual":
        return args
    if args.target_preset not in {"complex_2d_v1", "complex_2d_v2"}:
        raise ValueError(f"Unknown target_preset: {args.target_preset}")

    # A deliberately awkward 7-component anisotropic GMM.  Several modes are
    # close enough to create bridges under diffusion, but the clean density is
    # still visibly multimodal.
    args.gmm_weights = "0.08,0.16,0.10,0.22,0.13,0.18,0.13"
    args.gmm_means = "-1.10,-0.72;-0.62,0.34;-0.18,-0.10;0.08,0.58;0.46,-0.42;0.83,0.20;1.12,0.78"
    args.gmm_stds = "0.10,0.16;0.07,0.11;0.16,0.07;0.09,0.15;0.14,0.08;0.08,0.13;0.12,0.09"
    args.gmm_corrs = "0.45,-0.55,0.35,-0.25,0.50,-0.40,0.20"
    args.reward_type = "rugged"
    if args.target_preset == "complex_2d_v2":
        # Stronger mode-reweighting target than v1.  It boosts the left-lower,
        # right-mid, and right-upper modes while suppressing central/right-lower
        # mass, giving a clean JS distance comparable to the hard 1D preset.
        args.reward_bump_centers = "-1.08,-0.66;-0.62,0.34;-0.18,-0.10;0.10,0.58;0.46,-0.42;0.83,0.20;1.12,0.78;0.34,0.02"
        args.reward_bump_widths = "0.13,0.16;0.11,0.14;0.18,0.09;0.12,0.17;0.15,0.10;0.12,0.15;0.16,0.13;0.22,0.18"
        args.reward_bump_weights = "2.9,-2.6,-2.0,1.2,-2.7,1.9,3.2,-1.2"
        args.reward_center = "0.58,0.42"
        args.reward_scales = "0.18,0.12"
        args.reward_sin_amp = 0.10
        args.reward_sin_freqs = "11.0,15.0"
        args.reward_sin_phase = 0.45
        args.reward_l2 = 0.025
        args.reward_l4 = 0.003
        return args

    # Positive bumps reward two low-density corridors and a right/top mode;
    # negative bumps suppress two high-mass original modes.  This makes
    # pi_target clearly different from pi_orig at beta=1.
    args.reward_bump_centers = "-0.98,-0.50;-0.52,0.08;0.03,0.34;0.42,-0.12;0.76,0.54;1.05,0.92"
    args.reward_bump_widths = "0.16,0.12;0.12,0.17;0.22,0.12;0.13,0.16;0.16,0.13;0.14,0.16"
    args.reward_bump_weights = "2.8,-2.2,2.4,-2.6,3.0,1.8"
    args.reward_center = "0.58,0.42"
    args.reward_scales = "0.18,0.12"
    args.reward_sin_amp = 0.12
    args.reward_sin_freqs = "10.0,13.0"
    args.reward_sin_phase = 0.35
    args.reward_l2 = 0.035
    args.reward_l4 = 0.004
    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target_preset", choices=["manual", "complex_2d_v1", "complex_2d_v2"], default="complex_2d_v2",
                   help="manual uses provided flags; complex_2d_v2 is the recommended hard 2D target.")
    p.add_argument("--gmm_weights", default="0.18,0.37,0.25,0.20")
    p.add_argument("--gmm_means", default="-0.85,-0.45;-0.15,0.18;0.10,-0.05;0.82,0.52")
    p.add_argument("--gmm_stds", default="0.09,0.14;0.06,0.10;0.16,0.07;0.08,0.11")
    p.add_argument("--gmm_corrs", default="0.20,-0.35,0.40,-0.20")
    p.add_argument("--reward_type", choices=["quadratic", "bumps", "rugged"], default="quadratic")
    p.add_argument("--reward_center", default="0.65,0.35")
    p.add_argument("--reward_scales", default="1.0,1.0")
    p.add_argument("--reward_bump_centers", default="0.55,0.28;-0.25,0.20;0.82,0.58")
    p.add_argument("--reward_bump_widths", default="0.16,0.14;0.20,0.12;0.12,0.16")
    p.add_argument("--reward_bump_weights", default="1.2,-0.7,0.8")
    p.add_argument("--reward_sin_amp", type=float, default=0.10)
    p.add_argument("--reward_sin_freqs", default="8.0,11.0")
    p.add_argument("--reward_sin_phase", type=float, default=0.2)
    p.add_argument("--reward_l2", type=float, default=0.05)
    p.add_argument("--reward_l4", type=float, default=0.006)
    p.add_argument("--num_samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--diffusion_steps", type=int, default=20)
    p.add_argument("--beta_schedule_type", choices=["linear", "cosine", "constant_kl"], default="cosine")
    p.add_argument("--snr_max", type=float, default=124.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--guidance_schedule", choices=GUIDANCE_SCHEDULE_CHOICES, default="constant",
                   help="Per-level guidance multiplier. constant is the current behavior; alpha_bar multiplies guidance by alpha_bar_t.")
    p.add_argument("--sampler", choices=["mala", "langevin", "dps", "mpgd", "unguided"], default="mala",
                   help="Algorithm family: MALA, unadjusted Langevin, or DDIM-only references.")
    p.add_argument("--mala_steps", type=int, default=4)
    p.add_argument("--mala_budget", type=int, default=None,
                   help="Total MALA step budget across diffusion levels. Defaults to mala_steps * diffusion_steps.")
    p.add_argument("--mala_step_schedule", choices=MALA_STEP_SCHEDULE_CHOICES, default="constant",
                   help="How to allocate mala_budget across diffusion levels.")
    p.add_argument("--mala_eta", type=float, default=1.0,
                   help="Initial/fixed MALA step multiplier: eta_t = mala_eta * beta_t when mala_adapt_rate=0.")
    p.add_argument("--langevin_steps", type=int, default=4)
    p.add_argument("--langevin_eta", type=float, default=1.0,
                   help="Fixed Langevin step multiplier: eta_t = langevin_eta * beta_t, clipped to [1e-8, 0.5].")
    p.add_argument("--denoising_predictor", choices=["Identity", "DDPM_mean", "DDIM"], default="DDPM_mean")
    p.add_argument("--guidance_gradient_space", choices=["xt", "x0hat", "x0hatclipped"], default="xt")
    p.add_argument("--x0_hat_method", choices=["posterior_mean", "tweedie"], default="tweedie",
                   help="Oracle posterior mean is stable; tweedie uses the standard epsilon-reconstruction formula.")
    p.add_argument("--x0_hat_clip_radius", type=float, default=10.0)
    p.add_argument("--x_recon_clip_radius", type=float, default=1.0)
    p.add_argument("--mala_adapt_rate", type=float, default=0.2)
    p.add_argument("--guidance_strength_multiplier", type=float, default=1.0)
    p.add_argument("--batch_independent_guidance", action="store_true")
    p.add_argument("--advantage_normalization", action="store_true")
    p.add_argument("--initial_advantage_second_moment_ema", type=float, default=1.0)

    p.add_argument("--grid_mode", choices=["auto", "fixed"], default="auto")
    p.add_argument("--grid_min", type=float, default=-3.0)
    p.add_argument("--grid_max", type=float, default=3.0)
    p.add_argument("--grid_points", type=int, default=181)
    p.add_argument("--metric_target_samples", type=int, default=20000)
    p.add_argument("--metric_slices", type=int, default=64)
    p.add_argument("--eval_timesteps", default="auto")
    p.add_argument("--output_dir", type=str, required=True)
    return apply_target_preset(p.parse_args())


def normalize_weights(weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("GMM weights must be finite nonnegative values with positive sum.")
    return w / w.sum()


def make_covariances(stds: np.ndarray, corrs: np.ndarray) -> np.ndarray:
    if np.any(stds <= 0):
        raise ValueError("GMM stds must be positive.")
    if np.any(np.abs(corrs) >= 1.0):
        raise ValueError("GMM correlations must be in (-1, 1).")
    covs = np.zeros((len(stds), 2, 2), dtype=np.float64)
    covs[:, 0, 0] = stds[:, 0] ** 2
    covs[:, 1, 1] = stds[:, 1] ** 2
    covs[:, 0, 1] = corrs * stds[:, 0] * stds[:, 1]
    covs[:, 1, 0] = covs[:, 0, 1]
    return covs


def validate_reward_params(cfg: Toy2DConfig) -> None:
    if cfg.reward_type == "quadratic":
        return
    n = len(cfg.reward_bump_centers)
    if not (n == len(cfg.reward_bump_widths) == len(cfg.reward_bump_weights)):
        raise ValueError("reward_bump_centers, reward_bump_widths, and reward_bump_weights must match.")
    if n == 0:
        raise ValueError("Bump/rugged rewards require at least one bump.")
    widths = np.asarray(cfg.reward_bump_widths, dtype=np.float64)
    if widths.shape[1] != 2 or np.any(widths <= 0):
        raise ValueError("reward_bump_widths must be positive 2D rows.")


def validate_sampler_params(cfg: Toy2DConfig) -> None:
    if cfg.sampler in {"dps", "mpgd", "unguided"} and cfg.denoising_predictor != "DDIM":
        raise ValueError(f"{cfg.sampler} is a DDIM-only reference; set --denoising_predictor DDIM.")


def jax_reward(
    action,
    reward_type: str,
    center,
    scales,
    bump_centers,
    bump_widths,
    bump_weights,
    sin_amp: float,
    sin_freqs,
    sin_phase: float,
    l2: float,
    l4: float,
):
    diff = action - center
    if reward_type == "quadratic":
        return -jnp.sum(scales * diff * diff, axis=-1)
    bump_diff = (action[:, None, :] - bump_centers[None, :, :]) / bump_widths[None, :, :]
    bumps = jnp.sum(bump_weights[None, :] * jnp.exp(-0.5 * jnp.sum(bump_diff * bump_diff, axis=-1)), axis=-1)
    radius2 = jnp.sum(action * action, axis=-1)
    penalty = jnp.float32(l2) * radius2 + jnp.float32(l4) * radius2 * radius2
    if reward_type == "bumps":
        return bumps - penalty
    if reward_type == "rugged":
        phase = action @ sin_freqs + jnp.float32(sin_phase)
        return bumps + jnp.float32(sin_amp) * jnp.sin(phase) - penalty
    raise ValueError(f"Unknown reward_type: {reward_type}")


def np_reward(action, cfg: Toy2DConfig):
    action = np.asarray(action, dtype=np.float64)
    center = np.asarray(cfg.reward_center, dtype=np.float64)
    scales = np.asarray(cfg.reward_scales, dtype=np.float64)
    diff = action - center
    if cfg.reward_type == "quadratic":
        return -np.sum(scales * diff * diff, axis=-1)
    bump_centers = np.asarray(cfg.reward_bump_centers, dtype=np.float64)
    bump_widths = np.asarray(cfg.reward_bump_widths, dtype=np.float64)
    bump_weights = np.asarray(cfg.reward_bump_weights, dtype=np.float64)
    bump_diff = (action[:, None, :] - bump_centers[None, :, :]) / bump_widths[None, :, :]
    bumps = np.sum(bump_weights[None, :] * np.exp(-0.5 * np.sum(bump_diff * bump_diff, axis=-1)), axis=-1)
    radius2 = np.sum(action * action, axis=-1)
    penalty = cfg.reward_l2 * radius2 + cfg.reward_l4 * radius2 * radius2
    if cfg.reward_type == "bumps":
        return bumps - penalty
    if cfg.reward_type == "rugged":
        freqs = np.asarray(cfg.reward_sin_freqs, dtype=np.float64)
        return bumps + cfg.reward_sin_amp * np.sin(action @ freqs + cfg.reward_sin_phase) - penalty
    raise ValueError(f"Unknown reward_type: {cfg.reward_type}")


def eval_timesteps(spec: str, timesteps: int) -> list[int]:
    if spec == "all":
        return list(range(timesteps))
    if spec == "auto":
        vals = [0, timesteps // 4, timesteps // 2, (3 * timesteps) // 4, timesteps - 1]
    else:
        vals = [int(x) for x in spec.replace(",", " ").split()]
    vals = sorted(set(v for v in vals if 0 <= v < timesteps))
    if not vals:
        raise ValueError("No valid eval timesteps selected.")
    return vals


def effective_beta(cfg: Toy2DConfig) -> float:
    beta = float(cfg.beta)
    if cfg.advantage_normalization:
        beta /= float(np.sqrt(max(float(cfg.initial_advantage_second_moment_ema), 1e-6)))
    return beta


def np_logsumexp(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def np_component_params(covs, means, schedule, t_idx: int | None):
    if t_idx is None:
        return means, covs
    c = float(np.asarray(schedule.sqrt_alphas_cumprod)[t_idx])
    sigma = float(np.asarray(schedule.sqrt_one_minus_alphas_cumprod)[t_idx])
    loc = c * means
    cov_t = (c * c) * covs + (sigma * sigma) * np.eye(2)[None, :, :]
    return loc, cov_t


def np_base_logpdf(points, weights, means, covs, schedule, t_idx: int | None):
    points = np.asarray(points, dtype=np.float64)
    loc, cov_t = np_component_params(covs, means, schedule, t_idx)
    inv_cov = np.linalg.inv(cov_t)
    sign, logdet = np.linalg.slogdet(cov_t)
    if not np.all(sign > 0):
        raise ValueError("Covariance must be positive definite.")
    diff = points[:, None, :] - loc[None, :, :]
    quad = np.einsum("nki,kij,nkj->nk", diff, inv_cov, diff)
    log_comp = np.log(weights)[None, :] - 0.5 * (2.0 * np.log(2.0 * np.pi) + logdet[None, :] + quad)
    return np_logsumexp(log_comp, axis=-1)


def np_x0_hat(points, weights, means, covs, schedule, t_idx: int, method: str = "posterior_mean"):
    points = np.asarray(points, dtype=np.float64)
    c = float(np.asarray(schedule.sqrt_alphas_cumprod)[t_idx])
    sigma = float(np.asarray(schedule.sqrt_one_minus_alphas_cumprod)[t_idx])
    loc, cov_t = np_component_params(covs, means, schedule, t_idx)
    inv_cov_t = np.linalg.inv(cov_t)
    log_comp = np_base_component_logpdf(points, weights, loc, cov_t, inv_cov_t)
    resp = np.exp(log_comp - np_logsumexp(log_comp, axis=-1)[:, None])
    diff = points[:, None, :] - loc[None, :, :]
    component_scores = -np.einsum("kij,nkj->nki", inv_cov_t, diff)
    score = np.sum(resp[:, :, None] * component_scores, axis=1)
    if method == "tweedie":
        return (points + sigma * sigma * score) / c
    if method != "posterior_mean":
        raise ValueError(f"Unknown x0_hat_method: {method}")
    gains = c * np.einsum("kij,kjl->kil", covs, inv_cov_t)
    cond_mean = means[None, :, :] + np.einsum("kij,nkj->nki", gains, diff)
    return np.sum(resp[:, :, None] * cond_mean, axis=1)


def np_base_component_logpdf(points, weights, loc, cov_t, inv_cov_t):
    sign, logdet = np.linalg.slogdet(cov_t)
    if not np.all(sign > 0):
        raise ValueError("Covariance must be positive definite.")
    diff = points[:, None, :] - loc[None, :, :]
    quad = np.einsum("nki,kij,nkj->nk", diff, inv_cov_t, diff)
    return np.log(weights)[None, :] - 0.5 * (2.0 * np.log(2.0 * np.pi) + logdet[None, :] + quad)


def normalize_grid_density(log_unnorm, grid_x, grid_y):
    shifted = log_unnorm - np.max(log_unnorm)
    dens = np.exp(shifted)
    dx = float(grid_x[1] - grid_x[0])
    dy = float(grid_y[1] - grid_y[0])
    z = float(np.sum(dens) * dx * dy)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Invalid grid normalization constant.")
    return dens / z


def target_density_grid(grid_x, grid_y, cfg, weights, means, covs, schedule, t_idx: int | None):
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    base_log = np_base_logpdf(points, weights, means, covs, schedule, t_idx)
    if t_idx is None:
        q_arg = points
    else:
        q_arg = np_x0_hat(points, weights, means, covs, schedule, t_idx, cfg.x0_hat_method)
        if np.isfinite(cfg.x0_hat_clip_radius):
            q_arg = np.clip(q_arg, -cfg.x0_hat_clip_radius, cfg.x0_hat_clip_radius)
    log_unnorm = cfg.alpha * base_log + effective_beta(cfg) * np_reward(q_arg, cfg)
    return normalize_grid_density(log_unnorm.reshape(len(grid_y), len(grid_x)), grid_x, grid_y)


def make_eval_grid(samples, cfg, means, covs, schedule, t_idx: int | None):
    if cfg.grid_mode == "fixed":
        grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_points, dtype=np.float64)
        return grid, grid
    samples = np.asarray(samples, dtype=np.float64)
    if t_idx is None:
        loc = means
        cov_t = covs
    else:
        loc, cov_t = np_component_params(covs, means, schedule, t_idx)
    std = np.sqrt(np.maximum(np.stack([cov_t[:, 0, 0], cov_t[:, 1, 1]], axis=1), 1e-12))
    base_low = np.min(loc - 5.0 * std, axis=0)
    base_high = np.max(loc + 5.0 * std, axis=0)
    sample_low = np.quantile(samples, 0.001, axis=0)
    sample_high = np.quantile(samples, 0.999, axis=0)
    low = np.minimum.reduce([base_low, sample_low, np.array([cfg.grid_min, cfg.grid_min])])
    high = np.maximum.reduce([base_high, sample_high, np.array([cfg.grid_max, cfg.grid_max])])
    margin = np.maximum(0.05 * (high - low), 1e-3)
    grid_x = np.linspace(low[0] - margin[0], high[0] + margin[0], cfg.grid_points)
    grid_y = np.linspace(low[1] - margin[1], high[1] + margin[1], cfg.grid_points)
    return grid_x, grid_y


def sample_from_grid_density(rng, grid_x, grid_y, dens, n_samples: int):
    dx = float(grid_x[1] - grid_x[0])
    dy = float(grid_y[1] - grid_y[0])
    probs = (dens.ravel() * dx * dy).astype(np.float64)
    probs = probs / probs.sum()
    idx = rng.choice(len(probs), size=n_samples, replace=True, p=probs)
    iy, ix = np.divmod(idx, len(grid_x))
    jitter_x = rng.uniform(-0.5 * dx, 0.5 * dx, size=n_samples)
    jitter_y = rng.uniform(-0.5 * dy, 0.5 * dy, size=n_samples)
    return np.column_stack([grid_x[ix] + jitter_x, grid_y[iy] + jitter_y])


def sliced_w1(samples, target_samples, n_slices: int, rng):
    theta = rng.normal(size=(n_slices, 2))
    theta = theta / np.linalg.norm(theta, axis=1, keepdims=True)
    vals = []
    n = min(len(samples), len(target_samples))
    for direction in theta:
        a = np.sort(samples[:n] @ direction)
        b = np.sort(target_samples[:n] @ direction)
        vals.append(np.mean(np.abs(a - b)))
    return float(np.mean(vals))


def marginal_ks(samples, target_samples, dim: int):
    a = np.sort(samples[:, dim])
    b = np.sort(target_samples[:, dim])
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(a, grid, side="right") / max(len(a), 1)
    fb = np.searchsorted(b, grid, side="right") / max(len(b), 1)
    return float(np.max(np.abs(fa - fb)))


def metrics_from_samples(
    samples,
    grid_x,
    grid_y,
    target_dens,
    cfg,
    rng,
    q_sample_arg=None,
    q_target_transform=None,
):
    samples = np.asarray(samples, dtype=np.float64)
    hist, x_edges, y_edges = np.histogram2d(samples[:, 0], samples[:, 1], bins=[grid_x, grid_y])
    p = hist.T.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    target_on_cells = 0.25 * (
        target_dens[:-1, :-1]
        + target_dens[1:, :-1]
        + target_dens[:-1, 1:]
        + target_dens[1:, 1:]
    )
    q = target_on_cells * np.diff(grid_x)[None, :] * np.diff(grid_y)[:, None]
    q = q / q.sum()
    eps = 1e-12
    kl_sample_target = float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))
    m = 0.5 * (p + q)
    js = float(
        0.5 * np.sum(p * (np.log(p + eps) - np.log(m + eps)))
        + 0.5 * np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    )
    target_samples = sample_from_grid_density(rng, grid_x, grid_y, target_dens, cfg.metric_target_samples)
    sw1 = sliced_w1(samples, target_samples, cfg.metric_slices, rng)
    ks_x = marginal_ks(samples, target_samples, 0)
    ks_y = marginal_ks(samples, target_samples, 1)
    if q_sample_arg is None:
        q_sample_arg = samples
    q_target_arg = target_samples if q_target_transform is None else q_target_transform(target_samples)
    q_sample = float(np.mean(np_reward(q_sample_arg, cfg)))
    q_target = float(np.mean(np_reward(q_target_arg, cfg)))
    return {
        "kl_sample_target": kl_sample_target,
        "js": js,
        "sliced_w1": sw1,
        "marginal_ks_x": ks_x,
        "marginal_ks_y": ks_y,
        "sample_mean_q": q_sample,
        "target_mean_q": q_target,
        "sample_mean_x": float(np.mean(samples[:, 0])),
        "sample_mean_y": float(np.mean(samples[:, 1])),
        "sample_std_x": float(np.std(samples[:, 0])),
        "sample_std_y": float(np.std(samples[:, 1])),
    }


def save_contour_plot(path, samples, grid_x, grid_y, target_dens, title):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    levels = 14
    ax.contour(grid_x, grid_y, target_dens, levels=levels, colors="tab:orange", linewidths=1.2)
    if len(samples) > 2500:
        rng = np.random.default_rng(0)
        samples = samples[rng.choice(len(samples), size=2500, replace=False)]
    ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.22, color="tab:blue", linewidths=0)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_surface_plot(path, grid_x, grid_y, density, title):
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    stride = max(1, len(grid_x) // 120)
    ax.plot_surface(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        density[::stride, ::stride],
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("density")
    ax.set_title(title)
    ax.view_init(elev=32, azim=-55)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_orig_target_comparison(path, grid_x, grid_y, orig_dens, target_dens, q_grid, title):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    panels = [
        (orig_dens, r"$\pi_{\mathrm{orig}}$", "viridis"),
        (target_dens, r"$\pi_{\mathrm{target}}$", "viridis"),
        (q_grid, "Q(x)", "coolwarm"),
    ]
    for ax, (values, label, cmap) in zip(axes, panels):
        im = ax.contourf(grid_x, grid_y, values, levels=28, cmap=cmap)
        ax.contour(grid_x, grid_y, values, levels=10, colors="black", linewidths=0.25, alpha=0.35)
        ax.set_title(label)
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(im, ax=ax, shrink=0.86)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_contour_panel(path, panels, title):
    n = len(panels)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.0 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, panel in zip(axes.ravel(), panels):
        samples = panel["samples"]
        if len(samples) > 1800:
            rng = np.random.default_rng(0)
            samples = samples[rng.choice(len(samples), size=1800, replace=False)]
        ax.contour(panel["grid_x"], panel["grid_y"], panel["target_dens"], levels=12, colors="tab:orange")
        ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.20, color="tab:blue", linewidths=0)
        ax.set_title(
            f"{panel['label']}\nJS={panel['js']:.4f}, SW1={panel['sliced_w1']:.4f}, acc={panel['acc']:.3f}",
            fontsize=9,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_toy_mala_sampler(key: jax.Array, model: Toy2DModel, cfg: Toy2DConfig) -> Toy2DSamplerResult:
    schedule = model.schedule
    action_shape = (cfg.num_samples, model.act_dim)
    reduce_over_batch = jnp.sum if cfg.batch_independent_guidance else jnp.mean
    beta_current = jnp.float32(effective_beta(cfg))
    sampler = getattr(cfg, "sampler", "mala")
    if sampler == "dps":
        guidance_gradient_space = "xt"
    elif sampler == "mpgd":
        guidance_gradient_space = "x0hat"
    else:
        guidance_gradient_space = cfg.guidance_gradient_space
    def guidance_beta(t_idx):
        if sampler == "unguided":
            return jnp.float32(0.0)
        multiplier = guidance_schedule_multiplier(cfg.guidance_schedule, schedule.alphas_cumprod, t_idx)
        return beta_current * jnp.asarray(multiplier, dtype=jnp.float32)

    def q_at_clipped_x0_hat(x0_hat):
        x0_clipped = jnp.clip(x0_hat, -cfg.x0_hat_clip_radius, cfg.x0_hat_clip_radius)
        return model.q(None, None, x0_clipped)

    def reconstruct_x0_from_noise(x_in, t_idx, noise_pred):
        return (
            x_in * schedule.sqrt_recip_alphas_cumprod[t_idx]
            - noise_pred * schedule.sqrt_recipm1_alphas_cumprod[t_idx]
        )

    def base_x0_hat(x_in, t_idx):
        if cfg.x0_hat_method == "posterior_mean":
            return model.x0_hat_from_xt(x_in, t_idx)
        if cfg.x0_hat_method == "tweedie":
            return reconstruct_x0_from_noise(x_in, t_idx, model.eps_pred(None, None, x_in, t_idx))
        raise ValueError(f"Unknown x0_hat_method: {cfg.x0_hat_method}")

    def energy_total(t_idx, x):
        E_vals, vjp_fn = jax.vjp(lambda a: model.energy_fn(None, None, a, t_idx), x)
        (e_grad,) = vjp_fn(jnp.ones_like(E_vals))
        x0_hat = base_x0_hat(x, t_idx)
        clip_frac = jnp.mean(jnp.any(jnp.abs(x0_hat) > cfg.x0_hat_clip_radius, axis=-1).astype(jnp.float32))
        beta_t = guidance_beta(t_idx)
        return jnp.float32(cfg.alpha) * E_vals - beta_t * q_at_clipped_x0_hat(x0_hat), clip_frac

    def guidance_value_from_x(x_in, t_idx):
        x0_hat = base_x0_hat(x_in, t_idx)
        q = q_at_clipped_x0_hat(x0_hat)
        return jnp.float32(cfg.guidance_strength_multiplier) * reduce_over_batch(q)

    def compute_guidance_gradient(x_in, t_idx):
        if sampler == "unguided":
            return jnp.zeros_like(x_in)
        if guidance_gradient_space == "xt":
            return jax.grad(lambda x: guidance_value_from_x(x, t_idx))(x_in)
        x0_hat = base_x0_hat(x_in, t_idx)
        if guidance_gradient_space == "x0hat":
            return jax.grad(
                lambda x0: jnp.float32(cfg.guidance_strength_multiplier) * reduce_over_batch(
                    q_at_clipped_x0_hat(x0)
                )
            )(jax.lax.stop_gradient(x0_hat))
        x0_clipped = jnp.clip(x0_hat, -cfg.x0_hat_clip_radius, cfg.x0_hat_clip_radius)
        return jax.grad(
            lambda action: jnp.float32(cfg.guidance_strength_multiplier) * reduce_over_batch(
                model.q(None, None, action)
            )
        )(jax.lax.stop_gradient(x0_clipped))

    def jacobian_free_energy_and_drift(t_idx, x):
        E_vals, vjp_fn = jax.vjp(lambda a: model.energy_fn(None, None, a, t_idx), x)
        (e_grad,) = vjp_fn(jnp.ones_like(E_vals))
        x0_hat = base_x0_hat(x, t_idx)
        x0_clipped = jnp.clip(x0_hat, -cfg.x0_hat_clip_radius, cfg.x0_hat_clip_radius)
        q = model.q(None, None, x0_clipped)
        if guidance_gradient_space == "x0hat":
            grad_q = jax.grad(lambda x0: jnp.sum(q_at_clipped_x0_hat(x0)))(
                jax.lax.stop_gradient(x0_hat)
            )
        else:
            grad_q = jax.grad(lambda action: jnp.sum(model.q(None, None, action)))(
                jax.lax.stop_gradient(x0_clipped)
            )
        beta_t = guidance_beta(t_idx)
        energy = jnp.float32(cfg.alpha) * E_vals - beta_t * q
        grad_energy = jnp.float32(cfg.alpha) * e_grad - beta_t * grad_q
        clip_frac = jnp.mean(jnp.any(jnp.abs(x0_hat) > cfg.x0_hat_clip_radius, axis=-1).astype(jnp.float32))
        return energy, grad_energy, clip_frac

    def guided_x0_and_eps(t_idx, x_in):
        eps_base = model.eps_pred(None, None, x_in, t_idx)
        grad_q = compute_guidance_gradient(x_in, t_idx)
        sigma_t = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
        c_t = schedule.sqrt_alphas_cumprod[t_idx]
        beta_t = guidance_beta(t_idx)
        eps_guided = jnp.float32(cfg.alpha) * eps_base - beta_t * sigma_t * grad_q
        x0_guided = (
            jnp.float32(cfg.alpha) * base_x0_hat(x_in, t_idx)
            + (jnp.float32(1.0) - jnp.float32(cfg.alpha)) * x_in / c_t
            + beta_t * (sigma_t * sigma_t / c_t) * grad_q
        )
        return x0_guided, eps_guided

    def ddpm_mean_step(t_idx, x_in):
        x0_guided, _ = guided_x0_and_eps(t_idx, x_in)
        x0_hat = jnp.clip(x0_guided, -model.x_recon_clip_radius, model.x_recon_clip_radius)
        return x0_hat * schedule.posterior_mean_coef1[t_idx] + x_in * schedule.posterior_mean_coef2[t_idx]

    def ddim_step(t_idx, x_in):
        x0_guided, eps_pred = guided_x0_and_eps(t_idx, x_in)
        c_prev = jnp.sqrt(schedule.alphas_cumprod_prev[t_idx])
        sigma_prev = jnp.sqrt(1.0 - schedule.alphas_cumprod_prev[t_idx])
        return c_prev * x0_guided + sigma_prev * eps_pred

    if cfg.denoising_predictor == "Identity":
        def denoising_step(t_idx, x_curr):
            del t_idx
            return x_curr
    elif cfg.denoising_predictor == "DDPM_mean":
        denoising_step = ddpm_mean_step
    elif cfg.denoising_predictor == "DDIM":
        denoising_step = ddim_step
    else:
        raise ValueError(f"Unknown denoising_predictor: {cfg.denoising_predictor}")

    log_eta_min = jnp.log(jnp.float32(1e-8) / jnp.maximum(jnp.max(schedule.betas), jnp.float32(1e-8)))
    log_eta_max = jnp.log(jnp.float32(0.5) / jnp.maximum(jnp.min(schedule.betas), jnp.float32(1e-8)))

    def run_mala_chain_at_level(t_idx, x_t, rng, log_eta_scales):
        eta_base_t = jnp.maximum(schedule.betas[t_idx], jnp.float32(1e-8))
        eta_upper = jnp.float32(0.5)

        def mala_body(_, state):
            x_current, rng_step, log_eta_scale, accept_rate_sum, clip_frac_sum = state
            if guidance_gradient_space == "xt":
                E_x, vjp_x, clip_x = jax.vjp(lambda xx: energy_total(t_idx, xx), x_current, has_aux=True)
                grad_E_x = vjp_x(jnp.ones_like(E_x))[0]
            else:
                E_x, grad_E_x, clip_x = jacobian_free_energy_and_drift(t_idx, x_current)
            step_size = jnp.clip(jnp.exp(log_eta_scale) * eta_base_t, jnp.float32(1e-8), eta_upper)
            proposal_mean = x_current - step_size * grad_E_x
            proposal_std = jnp.sqrt(jnp.float32(2.0) * step_size)
            rng_step, noise_key, u_key = jax.random.split(rng_step, 3)
            x_prop = proposal_mean + proposal_std * jax.random.normal(noise_key, x_current.shape)
            if guidance_gradient_space == "xt":
                E_x_prop, vjp_x_prop, _clip_prop = jax.vjp(lambda xx: energy_total(t_idx, xx), x_prop, has_aux=True)
                grad_E_x_prop = vjp_x_prop(jnp.ones_like(E_x_prop))[0]
            else:
                E_x_prop, grad_E_x_prop, _clip_prop = jacobian_free_energy_and_drift(t_idx, x_prop)
            reverse_mean = x_prop - step_size * grad_E_x_prop

            def gaussian_log_density(x, mean):
                diff = x - mean
                return -jnp.sum(diff * diff, axis=-1) / (jnp.float32(4.0) * step_size)

            proposal_log_prob = gaussian_log_density(x_prop, proposal_mean)
            reverse_log_prob = gaussian_log_density(x_current, reverse_mean)
            log_acceptance_ratio = (-E_x_prop + E_x) + (reverse_log_prob - proposal_log_prob)
            accept = jnp.log(jax.random.uniform(u_key, E_x.shape)) < jnp.minimum(
                jnp.float32(0.0), log_acceptance_ratio
            )
            x_next = jnp.where(accept[:, None], x_prop, x_current)
            acc_rate = jnp.mean(accept.astype(jnp.float32).reshape(-1))
            log_eta_scale = log_eta_scale + jnp.float32(cfg.mala_adapt_rate) * (acc_rate - jnp.float32(0.574))
            log_eta_scale = jnp.clip(log_eta_scale, log_eta_min, log_eta_max)
            return x_next, rng_step, log_eta_scale, accept_rate_sum + acc_rate, clip_frac_sum + clip_x

        mala_steps_t = int(cfg.mala_steps_per_level[t_idx])
        mala_x_t, rng_out, log_eta_scale_new, acc_sum, clip_sum = jax.lax.fori_loop(
            0,
            mala_steps_t,
            mala_body,
            (x_t, rng, log_eta_scales[t_idx], jnp.float32(0.0), jnp.float32(0.0)),
        )
        log_eta_scales = log_eta_scales.at[t_idx].set(log_eta_scale_new)
        denom = jnp.maximum(jnp.float32(mala_steps_t), jnp.float32(1.0))
        return mala_x_t, rng_out, log_eta_scales, acc_sum / denom, clip_sum / denom

    def run_langevin_chain_at_level(t_idx, x_t, rng):
        step_size = jnp.clip(
            jnp.float32(cfg.langevin_eta) * jnp.maximum(schedule.betas[t_idx], jnp.float32(1e-8)),
            jnp.float32(1e-8),
            jnp.float32(0.5),
        )
        proposal_std = jnp.sqrt(jnp.float32(2.0) * step_size)

        def langevin_body(_, state):
            x_current, rng_step, clip_frac_sum = state
            if guidance_gradient_space == "xt":
                E_x, vjp_x, clip_x = jax.vjp(lambda xx: energy_total(t_idx, xx), x_current, has_aux=True)
                grad_E_x = vjp_x(jnp.ones_like(E_x))[0]
            else:
                _E_x, grad_E_x, clip_x = jacobian_free_energy_and_drift(t_idx, x_current)
            rng_step, noise_key = jax.random.split(rng_step)
            x_next = x_current - step_size * grad_E_x + proposal_std * jax.random.normal(noise_key, x_current.shape)
            return x_next, rng_step, clip_frac_sum + clip_x

        langevin_x_t, rng_out, clip_sum = jax.lax.fori_loop(
            0,
            cfg.langevin_steps,
            langevin_body,
            (x_t, rng, jnp.float32(0.0)),
        )
        denom = jnp.maximum(jnp.float32(cfg.langevin_steps), jnp.float32(1.0))
        return langevin_x_t, rng_out, clip_sum / denom

    def _run(k):
        key_x, loop_key = jax.random.split(k, 2)
        x_t = jax.random.normal(key_x, action_shape)
        initial_log_eta = jnp.log(jnp.maximum(jnp.float32(cfg.mala_eta), jnp.float32(1e-8)))
        log_eta_scales = jnp.full((cfg.diffusion_steps,), initial_log_eta, dtype=jnp.float32)
        per_level_acc = jnp.zeros((cfg.diffusion_steps,), dtype=jnp.float32)
        per_level_clip = jnp.zeros((cfg.diffusion_steps,), dtype=jnp.float32)
        trace = jnp.zeros((cfg.diffusion_steps, cfg.num_samples, model.act_dim), dtype=jnp.float32)
        for i in range(cfg.diffusion_steps):
            t_idx = cfg.diffusion_steps - 1 - i
            if sampler == "mala":
                mala_x_t, loop_key, log_eta_scales, acc, clip_frac = run_mala_chain_at_level(
                    t_idx, x_t, loop_key, log_eta_scales
                )
            elif sampler == "langevin":
                mala_x_t, loop_key, clip_frac = run_langevin_chain_at_level(t_idx, x_t, loop_key)
                acc = jnp.float32(jnp.nan)
            else:
                mala_x_t = x_t
                x0_hat = base_x0_hat(mala_x_t, t_idx)
                acc = jnp.float32(jnp.nan)
                clip_frac = jnp.mean(jnp.any(jnp.abs(x0_hat) > cfg.x0_hat_clip_radius, axis=-1).astype(jnp.float32))
            trace = trace.at[t_idx].set(mala_x_t)
            per_level_acc = per_level_acc.at[t_idx].set(acc)
            per_level_clip = per_level_clip.at[t_idx].set(clip_frac)
            x_t = denoising_step(t_idx, mala_x_t)
        raw_x0 = x_t
        action = jnp.clip(raw_x0, -1.0, 1.0)
        return Toy2DSamplerResult(action, raw_x0, trace, log_eta_scales, per_level_acc, per_level_clip)

    return jax.jit(_run)(key)


def main() -> None:
    args = parse_args()
    weights = normalize_weights(parse_float_list(args.gmm_weights))
    means = np.asarray(parse_matrix_rows(args.gmm_means), dtype=np.float64)
    stds = np.asarray(parse_matrix_rows(args.gmm_stds), dtype=np.float64)
    corrs = np.asarray(parse_float_list(args.gmm_corrs), dtype=np.float64)
    if not (len(weights) == len(means) == len(stds) == len(corrs)):
        raise ValueError("GMM weights, means, stds, and corrs must have the same length.")
    covs = make_covariances(stds, corrs)
    reward_center = parse_vector(args.reward_center)
    reward_scales = parse_vector(args.reward_scales)
    reward_bump_centers = parse_matrix_rows(args.reward_bump_centers)
    reward_bump_widths = parse_matrix_rows(args.reward_bump_widths)
    reward_bump_weights = parse_float_list(args.reward_bump_weights)
    reward_sin_freqs = parse_vector(args.reward_sin_freqs)

    cfg = Toy2DConfig(
        target_preset=args.target_preset,
        gmm_weights=weights.tolist(),
        gmm_means=means.tolist(),
        gmm_stds=stds.tolist(),
        gmm_corrs=corrs.tolist(),
        reward_type=args.reward_type,
        reward_center=reward_center,
        reward_scales=reward_scales,
        reward_bump_centers=reward_bump_centers,
        reward_bump_widths=reward_bump_widths,
        reward_bump_weights=reward_bump_weights,
        reward_sin_amp=args.reward_sin_amp,
        reward_sin_freqs=reward_sin_freqs,
        reward_sin_phase=args.reward_sin_phase,
        reward_l2=args.reward_l2,
        reward_l4=args.reward_l4,
        num_samples=args.num_samples,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        beta_schedule_type=args.beta_schedule_type,
        snr_max=args.snr_max,
        alpha=args.alpha,
        beta=args.beta,
        guidance_schedule=args.guidance_schedule,
        sampler=args.sampler,
        mala_steps=args.mala_steps,
        mala_budget=args.mala_budget if args.mala_budget is not None else args.mala_steps * args.diffusion_steps,
        mala_step_schedule=args.mala_step_schedule,
        mala_steps_per_level=[],
        mala_eta=args.mala_eta,
        langevin_steps=args.langevin_steps,
        langevin_eta=args.langevin_eta,
        denoising_predictor=args.denoising_predictor,
        guidance_gradient_space=args.guidance_gradient_space,
        x0_hat_method=args.x0_hat_method,
        x0_hat_clip_radius=args.x0_hat_clip_radius,
        x_recon_clip_radius=args.x_recon_clip_radius,
        mala_adapt_rate=args.mala_adapt_rate,
        guidance_strength_multiplier=args.guidance_strength_multiplier,
        batch_independent_guidance=args.batch_independent_guidance,
        advantage_normalization=args.advantage_normalization,
        initial_advantage_second_moment_ema=args.initial_advantage_second_moment_ema,
        grid_mode=args.grid_mode,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        grid_points=args.grid_points,
        metric_target_samples=args.metric_target_samples,
        metric_slices=args.metric_slices,
        eval_timesteps=args.eval_timesteps,
        output_dir=args.output_dir,
    )
    validate_reward_params(cfg)
    validate_sampler_params(cfg)

    clip_cover = float(np.max(np.abs(means) + 4.0 * stds))
    if np.isfinite(cfg.x0_hat_clip_radius) and clip_cover > cfg.x0_hat_clip_radius:
        warnings.warn(
            f"x0_hat_clip_radius={cfg.x0_hat_clip_radius:g} may clip GMM mass; "
            f"max |mean|+4*std is {clip_cover:g}.",
            RuntimeWarning,
        )

    schedule = build_beta_schedule(cfg.diffusion_steps, cfg.beta_schedule_type, cfg.snr_max)
    mala_steps_per_level = allocate_mala_steps(
        cfg.mala_step_schedule,
        diffusion_steps=cfg.diffusion_steps,
        mala_budget=cfg.mala_budget,
        sqrt_alphas_cumprod=np.asarray(schedule.sqrt_alphas_cumprod),
        mala_steps=cfg.mala_steps,
    )
    cfg = replace(cfg, mala_budget=int(sum(mala_steps_per_level)), mala_steps_per_level=list(mala_steps_per_level))

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    model = Toy2DModel(
        weights=jnp.asarray(weights, dtype=jnp.float32),
        means=jnp.asarray(means, dtype=jnp.float32),
        covs=jnp.asarray(covs, dtype=jnp.float32),
        reward_type=cfg.reward_type,
        reward_center=jnp.asarray(reward_center, dtype=jnp.float32),
        reward_scales=jnp.asarray(reward_scales, dtype=jnp.float32),
        reward_bump_centers=jnp.asarray(reward_bump_centers, dtype=jnp.float32),
        reward_bump_widths=jnp.asarray(reward_bump_widths, dtype=jnp.float32),
        reward_bump_weights=jnp.asarray(reward_bump_weights, dtype=jnp.float32),
        reward_sin_amp=float(cfg.reward_sin_amp),
        reward_sin_freqs=jnp.asarray(reward_sin_freqs, dtype=jnp.float32),
        reward_sin_phase=float(cfg.reward_sin_phase),
        reward_l2=float(cfg.reward_l2),
        reward_l4=float(cfg.reward_l4),
        schedule=schedule,
        num_timesteps=cfg.diffusion_steps,
        mala_steps=cfg.mala_steps,
        x_recon_clip_radius=float(cfg.x_recon_clip_radius),
    )

    result = run_toy_mala_sampler(jax.random.key(cfg.seed), model, cfg)
    raw_x0 = np.asarray(result.raw_x0)
    action_clipped = np.asarray(result.action)
    trace = np.asarray(result.trace)
    eval_ts = eval_timesteps(cfg.eval_timesteps, cfg.diffusion_steps)
    rng = np.random.default_rng(cfg.seed + 2029)

    rows = []
    panels = []
    grid_x, grid_y = make_eval_grid(raw_x0, cfg, means, covs, schedule, None)
    final_target = target_density_grid(grid_x, grid_y, cfg, weights, means, covs, schedule, None)
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    clean_points = np.column_stack([xx.ravel(), yy.ravel()])
    clean_orig = np.exp(np_base_logpdf(clean_points, weights, means, covs, schedule, None)).reshape(
        len(grid_y), len(grid_x)
    )
    clean_q = np_reward(clean_points, cfg).reshape(len(grid_y), len(grid_x))
    save_orig_target_comparison(
        out / "plots" / "orig_vs_target_contours.png",
        grid_x,
        grid_y,
        clean_orig,
        final_target,
        clean_q,
        f"{cfg.target_preset}: clean pi_orig vs pi_target",
    )
    save_surface_plot(out / "plots" / "clean_pi_orig_surface.png", grid_x, grid_y, clean_orig, "clean pi_orig")
    save_surface_plot(out / "plots" / "clean_pi_target_surface.png", grid_x, grid_y, final_target, "clean pi_target")
    final_metrics = metrics_from_samples(raw_x0, grid_x, grid_y, final_target, cfg, rng)
    final_metrics.update(
        {
            "stage": "final_clean",
            "timestep": -1,
            "acceptance_rate": float(np.asarray(result.per_level_acc)[0]),
        }
    )
    rows.append(final_metrics)
    panels.append(
        {
            "label": "final clean",
            "samples": raw_x0,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "target_dens": final_target,
            "js": final_metrics["js"],
            "sliced_w1": final_metrics["sliced_w1"],
            "acc": final_metrics["acceptance_rate"],
        }
    )
    save_contour_plot(out / "plots" / "final_clean_contour.png", raw_x0, grid_x, grid_y, final_target,
                      "final raw x0 vs clean target")

    clipped_metrics = metrics_from_samples(action_clipped, grid_x, grid_y, final_target, cfg, rng)
    clipped_metrics.update(
        {
            "stage": "final_clipped_action",
            "timestep": -1,
            "acceptance_rate": float("nan"),
        }
    )
    rows.append(clipped_metrics)

    for t in eval_ts:
        samples_t = trace[t]
        grid_x, grid_y = make_eval_grid(samples_t, cfg, means, covs, schedule, t)
        dens = target_density_grid(grid_x, grid_y, cfg, weights, means, covs, schedule, t)
        sample_x0_hat = np_x0_hat(samples_t, weights, means, covs, schedule, t, cfg.x0_hat_method)
        if np.isfinite(cfg.x0_hat_clip_radius):
            sample_x0_hat = np.clip(sample_x0_hat, -cfg.x0_hat_clip_radius, cfg.x0_hat_clip_radius)

        def target_x0_hat(points, t_idx=t):
            x0_hat = np_x0_hat(points, weights, means, covs, schedule, t_idx, cfg.x0_hat_method)
            if np.isfinite(cfg.x0_hat_clip_radius):
                x0_hat = np.clip(x0_hat, -cfg.x0_hat_clip_radius, cfg.x0_hat_clip_radius)
            return x0_hat

        metrics = metrics_from_samples(
            samples_t,
            grid_x,
            grid_y,
            dens,
            cfg,
            rng,
            q_sample_arg=sample_x0_hat,
            q_target_transform=target_x0_hat,
        )
        metrics.update(
            {
                "stage": "intermediate",
                "timestep": int(t),
                "acceptance_rate": float(np.asarray(result.per_level_acc)[t]),
            }
        )
        rows.append(metrics)
        panels.append(
            {
                "label": f"intermediate t={t}",
                "samples": samples_t,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "target_dens": dens,
                "js": metrics["js"],
                "sliced_w1": metrics["sliced_w1"],
                "acc": metrics["acceptance_rate"],
            }
        )
        save_contour_plot(
            out / "plots" / f"intermediate_t{t:03d}_contour.png",
            samples_t,
            grid_x,
            grid_y,
            dens,
            f"post-MALA x_t at t={t} vs E_total target",
        )
        save_surface_plot(
            out / "plots" / f"intermediate_t{t:03d}_target_surface.png",
            grid_x,
            grid_y,
            dens,
            f"target density at t={t}",
        )

    save_contour_panel(
        out / "plots" / "contour_panel.png",
        panels,
        (
            f"sampler={cfg.sampler}, mala_schedule={cfg.mala_step_schedule}, budget={cfg.mala_budget}, "
            f"mala_steps={cfg.mala_steps}, langevin_steps={cfg.langevin_steps}, "
            f"mala_eta={cfg.mala_eta}, langevin_eta={cfg.langevin_eta}, gradient={cfg.guidance_gradient_space}, "
            f"predictor={cfg.denoising_predictor}, beta={cfg.beta}"
        ),
    )

    np.savez_compressed(
        out / "samples.npz",
        raw_x0=raw_x0,
        action_clipped=action_clipped,
        trace=trace,
        eval_timesteps=np.asarray(eval_ts, dtype=np.int32),
        per_level_acc=np.asarray(result.per_level_acc),
        per_level_clip=np.asarray(result.per_level_clip),
        mala_steps_per_level=np.asarray(cfg.mala_steps_per_level, dtype=np.int32),
        log_eta_scales=np.asarray(result.log_eta_scales),
    )

    fieldnames = [
        "stage",
        "timestep",
        "kl_sample_target",
        "js",
        "sliced_w1",
        "marginal_ks_x",
        "marginal_ks_y",
        "sample_mean_q",
        "target_mean_q",
        "sample_mean_x",
        "sample_mean_y",
        "sample_std_x",
        "sample_std_y",
        "acceptance_rate",
    ]
    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"wrote results to {out}")
    for row in rows:
        print(
            f"{row['stage']:>20s} t={row['timestep']:>3} "
            f"SW1={row['sliced_w1']:.5f} JS={row['js']:.5f} "
            f"acc={row['acceptance_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
