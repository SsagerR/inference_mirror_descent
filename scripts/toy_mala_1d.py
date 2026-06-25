#!/usr/bin/env python3
"""1D oracle toy experiment for MALA-guided diffusion sampling.

This script intentionally avoids online RL and policy training. It adapts the
project's production MALA sampler to a closed-form 1D Gaussian-mixture base
distribution and a hand-written reward Q(a), then compares final and
intermediate samples against the exact energy targeted at each level.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from relax.utils.diffusion import build_beta_schedule


@dataclass(frozen=True)
class ToyConfig:
    gmm_weights: list[float]
    gmm_means: list[float]
    gmm_stds: list[float]
    reward_type: str
    reward_center: float
    reward_scale: float
    num_samples: int
    seed: int
    diffusion_steps: int
    beta_schedule_type: str
    snr_max: float
    alpha: float
    beta: float
    mala_steps: int
    denoising_predictor: str
    guidance_gradient_space: str
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
    eval_timesteps: str
    output_dir: str


@dataclass
class ToyModel:
    weights: jax.Array
    means: jax.Array
    stds: jax.Array
    reward_center: float
    reward_scale: float
    schedule: object
    num_timesteps: int
    mala_steps: int
    x_recon_clip_radius: float
    act_dim: int = 1

    def _component_params(self, t_idx):
        sqrt_ab = self.schedule.sqrt_alphas_cumprod[t_idx]
        sigma = self.schedule.sqrt_one_minus_alphas_cumprod[t_idx]
        loc = sqrt_ab * self.means
        var = (sqrt_ab ** 2) * (self.stds ** 2) + sigma ** 2
        return loc, var

    def _log_base_density(self, x, t_idx):
        loc, var = self._component_params(t_idx)
        x = x[..., None]
        log_comp = (
            jnp.log(self.weights)
            - 0.5 * (jnp.log(2.0 * jnp.pi * var) + (x - loc) ** 2 / var)
        )
        return jax.nn.logsumexp(log_comp, axis=-1)

    def _score_base(self, x, t_idx):
        loc, var = self._component_params(t_idx)
        x_exp = x[..., None]
        log_comp = (
            jnp.log(self.weights)
            - 0.5 * (jnp.log(2.0 * jnp.pi * var) + (x_exp - loc) ** 2 / var)
        )
        resp = jax.nn.softmax(log_comp, axis=-1)
        return jnp.sum(resp * (-(x_exp - loc) / var), axis=-1)

    def x0_hat_from_xt(self, act, t_idx):
        loc, var = self._component_params(t_idx)
        sqrt_ab = self.schedule.sqrt_alphas_cumprod[t_idx]
        x = act[..., 0]
        x_exp = x[..., None]
        log_comp = (
            jnp.log(self.weights)
            - 0.5 * (jnp.log(2.0 * jnp.pi * var) + (x_exp - loc) ** 2 / var)
        )
        resp = jax.nn.softmax(log_comp, axis=-1)
        posterior_mean = self.means + (sqrt_ab * self.stds ** 2 / var) * (x_exp - loc)
        return jnp.sum(resp * posterior_mean, axis=-1)[..., None]

    def energy_fn(self, params, obs, act, t_idx):
        del params, obs
        x = act[..., 0]
        return -self._log_base_density(x, t_idx)

    def eps_pred(self, params, obs, act, t_idx):
        del params, obs
        x = act[..., 0]
        sigma = self.schedule.sqrt_one_minus_alphas_cumprod[t_idx]
        eps = -sigma * self._score_base(x, t_idx)
        return eps[..., None]

    def q(self, params, obs, act):
        del params, obs
        a = act[..., 0]
        return -self.reward_scale * (a - self.reward_center) ** 2


class ToySamplerResult(NamedTuple):
    action: jax.Array
    raw_x0: jax.Array
    trace: jax.Array
    log_eta_scales: jax.Array
    per_level_acc: jax.Array
    per_level_clip: jax.Array


def _parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.replace(",", " ").split()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gmm_weights", default="0.35,0.30,0.35")
    p.add_argument("--gmm_means", default="-0.75,0.0,0.75")
    p.add_argument("--gmm_stds", default="0.10,0.16,0.10")
    p.add_argument("--reward_type", choices=["quadratic"], default="quadratic")
    p.add_argument("--reward_center", type=float, default=0.45)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--num_samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--diffusion_steps", type=int, default=20)
    p.add_argument("--beta_schedule_type", choices=["linear", "cosine", "constant_kl"], default="linear")
    p.add_argument("--snr_max", type=float, default=124.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--mala_steps", type=int, default=4)
    p.add_argument("--denoising_predictor", choices=["Identity", "DDPM_mean", "DDIM"], default="DDPM_mean")
    p.add_argument("--guidance_gradient_space", choices=["xt", "x0hat", "x0hatclipped"], default="xt")
    p.add_argument("--x0_hat_clip_radius", type=float, default=10.0)
    p.add_argument("--x_recon_clip_radius", type=float, default=1.0,
                   help="DDPM_mean clean reconstruction clip radius. Default 1.0 matches train_setup.py.")
    p.add_argument("--mala_adapt_rate", type=float, default=0.05)
    p.add_argument("--guidance_strength_multiplier", type=float, default=1.0)
    p.add_argument("--batch_independent_guidance", action="store_true")
    p.add_argument("--advantage_normalization", action="store_true")
    p.add_argument("--initial_advantage_second_moment_ema", type=float, default=1.0)

    p.add_argument("--grid_mode", choices=["auto", "fixed"], default="auto",
                   help="Use an adaptive plotting/evaluation grid per timestep, or fixed [grid_min, grid_max].")
    p.add_argument("--grid_min", type=float, default=-6.0)
    p.add_argument("--grid_max", type=float, default=6.0)
    p.add_argument("--grid_points", type=int, default=2001)
    p.add_argument("--eval_timesteps", default="auto",
                   help="'auto', 'all', or comma/space-separated timestep indices such as '0,5,10,19'.")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def _normalize_weights(weights: Iterable[float]) -> np.ndarray:
    w = np.asarray(list(weights), dtype=np.float64)
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("GMM weights must be finite nonnegative values with positive sum.")
    return w / w.sum()


def _eval_timesteps(spec: str, timesteps: int) -> list[int]:
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


def _np_logsumexp(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def np_base_logpdf(x, weights, means, stds, schedule, t_idx: int | None):
    x = np.asarray(x, dtype=np.float64)
    if t_idx is None:
        loc = means
        var = stds ** 2
    else:
        sqrt_ab = float(np.asarray(schedule.sqrt_alphas_cumprod)[t_idx])
        sigma = float(np.asarray(schedule.sqrt_one_minus_alphas_cumprod)[t_idx])
        loc = sqrt_ab * means
        var = (sqrt_ab ** 2) * (stds ** 2) + sigma ** 2
    log_comp = (
        np.log(weights)[None, :]
        - 0.5 * (np.log(2.0 * np.pi * var)[None, :] + (x[:, None] - loc[None, :]) ** 2 / var[None, :])
    )
    return _np_logsumexp(log_comp, axis=-1)


def np_base_score(x, weights, means, stds, schedule, t_idx: int):
    x = np.asarray(x, dtype=np.float64)
    sqrt_ab = float(np.asarray(schedule.sqrt_alphas_cumprod)[t_idx])
    sigma = float(np.asarray(schedule.sqrt_one_minus_alphas_cumprod)[t_idx])
    loc = sqrt_ab * means
    var = (sqrt_ab ** 2) * (stds ** 2) + sigma ** 2
    log_comp = (
        np.log(weights)[None, :]
        - 0.5 * (np.log(2.0 * np.pi * var)[None, :] + (x[:, None] - loc[None, :]) ** 2 / var[None, :])
    )
    resp = np.exp(log_comp - _np_logsumexp(log_comp, axis=-1)[:, None])
    return np.sum(resp * (-(x[:, None] - loc[None, :]) / var[None, :]), axis=-1)


def np_reward(a, center, scale):
    return -scale * (np.asarray(a) - center) ** 2


def effective_beta(cfg: ToyConfig) -> float:
    beta = float(cfg.beta)
    if cfg.advantage_normalization:
        beta /= math_sqrt_max(cfg.initial_advantage_second_moment_ema, 1e-6)
    return beta


def math_sqrt_max(x: float, floor: float) -> float:
    return float(np.sqrt(max(float(x), float(floor))))


def np_x0_hat(x, weights, means, stds, schedule, t_idx: int):
    x = np.asarray(x, dtype=np.float64)
    sqrt_ab = float(np.asarray(schedule.sqrt_alphas_cumprod)[t_idx])
    sigma = float(np.asarray(schedule.sqrt_one_minus_alphas_cumprod)[t_idx])
    loc = sqrt_ab * means
    var = (sqrt_ab ** 2) * (stds ** 2) + sigma ** 2
    log_comp = (
        np.log(weights)[None, :]
        - 0.5 * (np.log(2.0 * np.pi * var)[None, :] + (x[:, None] - loc[None, :]) ** 2 / var[None, :])
    )
    resp = np.exp(log_comp - _np_logsumexp(log_comp, axis=-1)[:, None])
    posterior_mean = means[None, :] + (sqrt_ab * stds[None, :] ** 2 / var[None, :]) * (x[:, None] - loc[None, :])
    return np.sum(resp * posterior_mean, axis=-1)


def normalized_density_on_grid(log_unnorm, grid):
    log_unnorm = np.asarray(log_unnorm, dtype=np.float64)
    shifted = log_unnorm - np.max(log_unnorm)
    dens = np.exp(shifted)
    z = integrate_trapezoid(dens, grid)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Invalid density normalization constant.")
    return dens / z


def target_density(grid, cfg: ToyConfig, weights, means, stds, schedule, t_idx: int | None):
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


def metrics_from_samples(samples, grid, target_dens, cfg: ToyConfig, *,
                         q_sample_arg=None, q_grid_arg=None):
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


def save_density_plot(path, samples, grid, target_dens, title):
    plt.figure(figsize=(7.0, 4.5))
    plt.hist(samples, bins=100, density=True, alpha=0.38, label="samples")
    plt.plot(grid, target_dens, lw=2.0, label="target")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def save_density_panel(path, panels, title):
    n = len(panels)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, panel in zip(axes.ravel(), panels):
        samples = panel["samples"]
        grid = panel["grid"]
        target_dens = panel["target_dens"]
        ax.hist(samples, bins=80, density=True, alpha=0.38, label="samples")
        ax.plot(grid, target_dens, lw=1.8, label="target")
        subtitle = (
            f"{panel['label']}\n"
            f"W1={panel['w1']:.4f}, KS={panel['ks']:.4f}, JS={panel['js']:.4f}, acc={panel['acc']:.3f}"
        )
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("density")
    axes.ravel()[0].legend(loc="best", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_eval_grid(samples, cfg: ToyConfig, weights, means, stds, schedule, t_idx: int | None):
    if cfg.grid_mode == "fixed":
        return np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_points, dtype=np.float64)

    samples = np.asarray(samples, dtype=np.float64)
    if t_idx is None:
        loc = means
        sd = stds
    else:
        sqrt_ab = float(np.asarray(schedule.sqrt_alphas_cumprod)[t_idx])
        sigma = float(np.asarray(schedule.sqrt_one_minus_alphas_cumprod)[t_idx])
        loc = sqrt_ab * means
        sd = np.sqrt((sqrt_ab ** 2) * (stds ** 2) + sigma ** 2)

    base_low = float(np.min(loc - 6.0 * sd))
    base_high = float(np.max(loc + 6.0 * sd))
    sample_low, sample_high = np.quantile(samples, [0.001, 0.999])
    low = min(base_low, float(sample_low), cfg.grid_min)
    high = max(base_high, float(sample_high), cfg.grid_max)
    margin = max(0.05 * (high - low), 1e-3)
    return np.linspace(low - margin, high + margin, cfg.grid_points, dtype=np.float64)


def run_toy_mala_sampler(key: jax.Array, model: ToyModel, cfg: ToyConfig) -> ToySamplerResult:
    """Run a self-contained 1D traceable variant of the project's MALA sampler."""
    schedule = model.schedule
    action_shape = (cfg.num_samples, model.act_dim)
    reduce_over_batch = jnp.sum if cfg.batch_independent_guidance else jnp.mean
    beta_current = jnp.float32(effective_beta(cfg))

    def reconstruct_x0_from_noise(x_in, t_idx, noise_pred):
        return (
            x_in * schedule.sqrt_recip_alphas_cumprod[t_idx]
            - noise_pred * schedule.sqrt_recipm1_alphas_cumprod[t_idx]
        )

    def q_at_clipped_x0_hat(x0_hat):
        x0_clipped = jnp.clip(x0_hat, -cfg.x0_hat_clip_radius, cfg.x0_hat_clip_radius)
        return model.q(None, None, x0_clipped)

    def base_x0_hat(x_in, t_idx):
        return model.x0_hat_from_xt(x_in, t_idx)

    def energy_total(t_idx, x):
        E_vals, vjp_fn = jax.vjp(lambda a: model.energy_fn(None, None, a, t_idx), x)
        (e_grad,) = vjp_fn(jnp.ones_like(E_vals))
        x0_hat = base_x0_hat(x, t_idx)
        clip_frac = jnp.mean((jnp.abs(x0_hat) > cfg.x0_hat_clip_radius).astype(jnp.float32))
        return jnp.float32(cfg.alpha) * E_vals - beta_current * q_at_clipped_x0_hat(x0_hat), clip_frac

    def guidance_value_from_x(x_in, t_idx):
        x0_hat = base_x0_hat(x_in, t_idx)
        q = q_at_clipped_x0_hat(x0_hat)
        return jnp.float32(cfg.guidance_strength_multiplier) * reduce_over_batch(q)

    def compute_guidance_gradient(x_in, t_idx):
        if cfg.guidance_gradient_space == "xt":
            return jax.grad(lambda x: guidance_value_from_x(x, t_idx))(x_in)

        x0_hat = base_x0_hat(x_in, t_idx)
        if cfg.guidance_gradient_space == "x0hat":
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
        if cfg.guidance_gradient_space == "x0hat":
            grad_q = jax.grad(lambda x0: jnp.sum(q_at_clipped_x0_hat(x0)))(
                jax.lax.stop_gradient(x0_hat)
            )
        else:
            grad_q = jax.grad(lambda action: jnp.sum(model.q(None, None, action)))(
                jax.lax.stop_gradient(x0_clipped)
            )
        energy = jnp.float32(cfg.alpha) * E_vals - beta_current * q
        grad_energy = jnp.float32(cfg.alpha) * e_grad - beta_current * grad_q
        clip_frac = jnp.mean((jnp.abs(x0_hat) > cfg.x0_hat_clip_radius).astype(jnp.float32))
        return energy, grad_energy, clip_frac

    def guided_eps_pred(t_idx, x_in):
        noise_pred_scaled = jnp.float32(cfg.alpha) * model.eps_pred(None, None, x_in, t_idx)
        grad_q = compute_guidance_gradient(x_in, t_idx)
        sigma_t = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
        return noise_pred_scaled - beta_current * sigma_t * grad_q

    def guided_x0_and_eps(t_idx, x_in):
        eps_base = model.eps_pred(None, None, x_in, t_idx)
        grad_q = compute_guidance_gradient(x_in, t_idx)
        sigma_t = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
        sqrt_ab_t = schedule.sqrt_alphas_cumprod[t_idx]
        eps_guided = jnp.float32(cfg.alpha) * eps_base - beta_current * sigma_t * grad_q
        # Algebraically equal to reconstruct_x0_from_noise(x, t, eps_guided),
        # but uses the closed-form oracle E[x0 | x_t] to avoid high-noise cancellation.
        x0_guided = (
            jnp.float32(cfg.alpha) * base_x0_hat(x_in, t_idx)
            + (jnp.float32(1.0) - jnp.float32(cfg.alpha)) * x_in / sqrt_ab_t
            + beta_current * (sigma_t * sigma_t / sqrt_ab_t) * grad_q
        )
        return x0_guided, eps_guided

    def ddpm_mean_step(t_idx, x_in):
        x0_guided, _eps_pred = guided_x0_and_eps(t_idx, x_in)
        x0_hat = jnp.clip(
            x0_guided,
            -model.x_recon_clip_radius,
            model.x_recon_clip_radius,
        )
        return x0_hat * schedule.posterior_mean_coef1[t_idx] + x_in * schedule.posterior_mean_coef2[t_idx]

    def ddim_step(t_idx, x_in):
        x0_guided, eps_pred = guided_x0_and_eps(t_idx, x_in)
        sqrt_ab_prev = jnp.sqrt(schedule.alphas_cumprod_prev[t_idx])
        sqrt_one_minus_ab_prev = jnp.sqrt(1.0 - schedule.alphas_cumprod_prev[t_idx])
        return sqrt_ab_prev * x0_guided + sqrt_one_minus_ab_prev * eps_pred

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
            if cfg.guidance_gradient_space == "xt":
                E_x, vjp_x, clip_x = jax.vjp(lambda xx: energy_total(t_idx, xx), x_current, has_aux=True)
                grad_E_x = vjp_x(jnp.ones_like(E_x))[0]
            else:
                E_x, grad_E_x, clip_x = jacobian_free_energy_and_drift(t_idx, x_current)

            step_size = jnp.clip(jnp.exp(log_eta_scale) * eta_base_t, jnp.float32(1e-8), eta_upper)
            proposal_mean = x_current - step_size * grad_E_x
            proposal_std = jnp.sqrt(jnp.float32(2.0) * step_size)

            rng_step, noise_key, u_key = jax.random.split(rng_step, 3)
            x_prop = proposal_mean + proposal_std * jax.random.normal(noise_key, x_current.shape)

            if cfg.guidance_gradient_space == "xt":
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
            x_next = jnp.where(accept[..., None], x_prop, x_current)

            acc_rate = jnp.mean(accept.astype(jnp.float32).reshape(-1))
            log_eta_scale = log_eta_scale + jnp.float32(cfg.mala_adapt_rate) * (acc_rate - jnp.float32(0.574))
            log_eta_scale = jnp.clip(log_eta_scale, log_eta_min, log_eta_max)
            return (
                x_next,
                rng_step,
                log_eta_scale,
                accept_rate_sum + acc_rate,
                clip_frac_sum + clip_x,
            )

        mala_corrected_x_t, rng_out, log_eta_scale_new, acc_sum, clip_sum = jax.lax.fori_loop(
            0,
            cfg.mala_steps,
            mala_body,
            (x_t, rng, log_eta_scales[t_idx], jnp.float32(0.0), jnp.float32(0.0)),
        )
        log_eta_scales = log_eta_scales.at[t_idx].set(log_eta_scale_new)
        denom = jnp.maximum(jnp.float32(cfg.mala_steps), jnp.float32(1.0))
        return mala_corrected_x_t, rng_out, log_eta_scales, acc_sum / denom, clip_sum / denom

    def _run(k):
        key_x, loop_key = jax.random.split(k, 2)
        x_t = jax.random.normal(key_x, action_shape)
        log_eta_scales = jnp.zeros((cfg.diffusion_steps,), dtype=jnp.float32)
        per_level_acc = jnp.zeros((cfg.diffusion_steps,), dtype=jnp.float32)
        per_level_clip = jnp.zeros((cfg.diffusion_steps,), dtype=jnp.float32)
        trace = jnp.zeros((cfg.diffusion_steps, cfg.num_samples, model.act_dim), dtype=jnp.float32)

        for i in range(cfg.diffusion_steps):
            t_idx = cfg.diffusion_steps - 1 - i
            mala_x_t, loop_key, log_eta_scales, acc, clip_frac = run_mala_chain_at_level(
                t_idx, x_t, loop_key, log_eta_scales
            )
            trace = trace.at[t_idx].set(mala_x_t)
            per_level_acc = per_level_acc.at[t_idx].set(acc)
            per_level_clip = per_level_clip.at[t_idx].set(clip_frac)
            x_t = denoising_step(t_idx, mala_x_t)

        raw_x0 = x_t
        action = jnp.clip(raw_x0, -1.0, 1.0)
        return ToySamplerResult(
            action=action,
            raw_x0=raw_x0,
            trace=trace,
            log_eta_scales=log_eta_scales,
            per_level_acc=per_level_acc,
            per_level_clip=per_level_clip,
        )

    return jax.jit(_run)(key)


def main() -> None:
    args = _parse_args()
    weights = _normalize_weights(_parse_float_list(args.gmm_weights))
    means = np.asarray(_parse_float_list(args.gmm_means), dtype=np.float64)
    stds = np.asarray(_parse_float_list(args.gmm_stds), dtype=np.float64)
    if not (len(weights) == len(means) == len(stds)):
        raise ValueError("GMM weights, means, and stds must have the same length.")
    if np.any(stds <= 0):
        raise ValueError("GMM stds must be positive.")

    cfg = ToyConfig(
        gmm_weights=weights.tolist(),
        gmm_means=means.tolist(),
        gmm_stds=stds.tolist(),
        reward_type=args.reward_type,
        reward_center=args.reward_center,
        reward_scale=args.reward_scale,
        num_samples=args.num_samples,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        beta_schedule_type=args.beta_schedule_type,
        snr_max=args.snr_max,
        alpha=args.alpha,
        beta=args.beta,
        mala_steps=args.mala_steps,
        denoising_predictor=args.denoising_predictor,
        guidance_gradient_space=args.guidance_gradient_space,
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
        eval_timesteps=args.eval_timesteps,
        output_dir=args.output_dir,
    )

    clip_cover = float(np.max(np.abs(means) + 4.0 * stds))
    if np.isfinite(cfg.x0_hat_clip_radius) and clip_cover > cfg.x0_hat_clip_radius:
        warnings.warn(
            f"x0_hat_clip_radius={cfg.x0_hat_clip_radius:g} may clip GMM mass; "
            f"max |mean|+4*std is {clip_cover:g}.",
            RuntimeWarning,
        )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)

    schedule = build_beta_schedule(cfg.diffusion_steps, cfg.beta_schedule_type, cfg.snr_max)
    model = ToyModel(
        weights=jnp.asarray(weights, dtype=jnp.float32),
        means=jnp.asarray(means, dtype=jnp.float32),
        stds=jnp.asarray(stds, dtype=jnp.float32),
        reward_center=float(cfg.reward_center),
        reward_scale=float(cfg.reward_scale),
        schedule=schedule,
        num_timesteps=cfg.diffusion_steps,
        mala_steps=cfg.mala_steps,
        x_recon_clip_radius=float(cfg.x_recon_clip_radius),
    )

    key = jax.random.key(cfg.seed)
    result = run_toy_mala_sampler(key, model, cfg)

    action_clipped = np.asarray(result.action[:, 0])
    trace = np.asarray(result.trace[:, :, 0])
    raw_x0 = np.asarray(result.raw_x0[:, 0])
    eval_ts = _eval_timesteps(cfg.eval_timesteps, cfg.diffusion_steps)

    rows = []
    panels = []
    grid = make_eval_grid(raw_x0, cfg, weights, means, stds, schedule, None)
    final_target = target_density(grid, cfg, weights, means, stds, schedule, None)
    final_metrics = metrics_from_samples(raw_x0, grid, final_target, cfg)
    final_metrics.update({
        "stage": "final_clean",
        "timestep": -1,
        "acceptance_rate": float(np.asarray(result.per_level_acc)[0]),
        "clip_fraction": float(np.asarray(result.per_level_clip)[0]),
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
    save_density_plot(out / "plots" / "final_clean_density.png", raw_x0, grid, final_target,
                      "final raw x0 vs clean target")

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
            trace[t], grid, dens, cfg,
            q_sample_arg=sample_x0_hat, q_grid_arg=grid_x0_hat,
        )
        metrics.update({
            "stage": "intermediate",
            "timestep": int(t),
            "acceptance_rate": float(np.asarray(result.per_level_acc)[t]),
            "clip_fraction": float(np.asarray(result.per_level_clip)[t]),
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
        save_density_plot(out / "plots" / f"intermediate_t{t:03d}_density.png", trace[t], grid, dens,
                          f"post-MALA x_t at t={t} vs E_total target")
    save_density_panel(
        out / "plots" / "density_panel.png",
        panels,
        (
            f"mala_steps={cfg.mala_steps}, gradient={cfg.guidance_gradient_space}, "
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
        log_eta_scales=np.asarray(result.log_eta_scales),
        grid=make_eval_grid(raw_x0, cfg, weights, means, stds, schedule, None),
    )

    fieldnames = [
        "stage", "timestep", "kl_sample_target", "js", "w1", "ks",
        "sample_mean_q", "target_mean_q", "sample_mean", "sample_std",
        "acceptance_rate", "clip_fraction",
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
            f"W1={row['w1']:.5f} KS={row['ks']:.5f} JS={row['js']:.5f} "
            f"acc={row['acceptance_rate']:.3f} clip={row['clip_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
