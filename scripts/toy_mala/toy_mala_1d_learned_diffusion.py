#!/usr/bin/env python3
"""1D toy MALA experiment with a learned diffusion policy.

This script keeps the 1D oracle GMM/Q/evaluation setup from toy_mala_1d.py,
but replaces the oracle diffusion energy/score used by the sampler with the
repo's existing ActorCritic diffusion-policy architecture trained by eps-MSE.

The Q function and evaluation target remain oracle. This isolates the effect
of learned diffusion model error while keeping the rest of the toy experiment
close to the production MALA sampler.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from relax.network.actor_critic import ActorCritic
from scripts.toy_mala.toy_mala_1d import (
    _eval_timesteps,
    _normalize_weights,
    _parse_float_list,
    make_eval_grid,
    metrics_from_samples,
    np_x0_hat,
    run_toy_mala_sampler,
    save_density_panel,
    save_density_plot,
    target_density,
)


@dataclass(frozen=True)
class LearnedToyConfig:
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
    eval_timesteps: str
    output_dir: str

    hidden_num: int
    hidden_dim: int
    diffusion_hidden_dim: int
    policy_parameterization: str
    policy_final_layer: str
    train_steps: int
    train_batch_size: int
    train_lr: float
    train_log_every: int
    diagnostic_batch_size: int


@dataclass
class LearnedToyModel:
    actor: ActorCritic
    policy_params: object
    reward_center: float
    reward_scale: float
    schedule: object
    num_timesteps: int
    mala_steps: int
    x_recon_clip_radius: float
    act_dim: int = 1

    def _obs_like(self, act):
        return jnp.zeros((*act.shape[:-1], 1), dtype=act.dtype)

    def energy_fn(self, params, obs, act, t_idx):
        del params, obs
        return self.actor.energy_fn(self.policy_params, self._obs_like(act), act, t_idx)

    def eps_pred(self, params, obs, act, t_idx):
        del params, obs
        return self.actor.eps_pred(self.policy_params, self._obs_like(act), act, t_idx)

    def x0_hat_from_xt(self, act, t_idx):
        eps = self.eps_pred(None, None, act, t_idx)
        return (
            act * self.schedule.sqrt_recip_alphas_cumprod[t_idx]
            - eps * self.schedule.sqrt_recipm1_alphas_cumprod[t_idx]
        )

    def q(self, params, obs, act):
        del params, obs
        a = act[..., 0]
        return -self.reward_scale * (a - self.reward_center) ** 2


def _mish(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jax.nn.softplus(x))


def parse_args() -> argparse.Namespace:
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
    p.add_argument("--beta_schedule_type", choices=["linear", "cosine", "constant_kl"], default="cosine")
    p.add_argument("--snr_max", type=float, default=124.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--mala_steps", type=int, default=4)
    p.add_argument("--denoising_predictor", choices=["Identity", "DDPM_mean", "DDIM"], default="DDPM_mean")
    p.add_argument("--guidance_gradient_space", choices=["xt", "x0hat", "x0hatclipped"], default="xt")
    p.add_argument("--x0_hat_method", choices=["tweedie"], default="tweedie",
                   help="Learned diffusion has no closed-form posterior mean; use production-style Tweedie.")
    p.add_argument("--x0_hat_clip_radius", type=float, default=10.0)
    p.add_argument("--x_recon_clip_radius", type=float, default=1.0)
    p.add_argument("--mala_adapt_rate", type=float, default=0.2)
    p.add_argument("--guidance_strength_multiplier", type=float, default=1.0)
    p.add_argument("--batch_independent_guidance", action="store_true")
    p.add_argument("--advantage_normalization", action="store_true")
    p.add_argument("--initial_advantage_second_moment_ema", type=float, default=1.0)

    p.add_argument("--grid_mode", choices=["auto", "fixed"], default="auto")
    p.add_argument("--grid_min", type=float, default=-6.0)
    p.add_argument("--grid_max", type=float, default=6.0)
    p.add_argument("--grid_points", type=int, default=2001)
    p.add_argument("--eval_timesteps", default="auto")
    p.add_argument("--output_dir", type=str, required=True)

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


def sample_gmm(key, weights, means, stds, batch_size: int):
    key_comp, key_noise = jax.random.split(key)
    comp = jax.random.categorical(key_comp, jnp.log(weights), shape=(batch_size,))
    noise = jax.random.normal(key_noise, (batch_size, 1))
    x0 = means[comp, None] + stds[comp, None] * noise
    return x0


def make_actor(cfg: LearnedToyConfig) -> ActorCritic:
    hidden_sizes = [cfg.hidden_dim] * cfg.hidden_num
    diffusion_hidden_sizes = [cfg.diffusion_hidden_dim] * cfg.hidden_num
    return ActorCritic.create(
        obs_dim=1,
        act_dim=1,
        hidden_sizes=hidden_sizes,
        diffusion_hidden_sizes=diffusion_hidden_sizes,
        activation=_mish,
        num_timesteps=cfg.diffusion_steps,
        beta_schedule_type=cfg.beta_schedule_type,
        mala_steps=cfg.mala_steps,
        x_recon_clip_radius=cfg.x_recon_clip_radius,
        snr_max=cfg.snr_max,
        num_q_networks=2,
        policy_parameterization=cfg.policy_parameterization,
        policy_final_layer=cfg.policy_final_layer,
    )


def train_policy(actor: ActorCritic, cfg: LearnedToyConfig, weights, means, stds):
    weights_j = jnp.asarray(weights, dtype=jnp.float32)
    means_j = jnp.asarray(means, dtype=jnp.float32)
    stds_j = jnp.asarray(stds, dtype=jnp.float32)
    opt = optax.adam(cfg.train_lr)

    init_key, loop_key = jax.random.split(jax.random.key(cfg.seed + 17))
    policy_params = actor.init_params(init_key).policy
    opt_state = opt.init(policy_params)

    def loss_fn(params, key):
        key_x0, key_t, key_noise = jax.random.split(key, 3)
        x0 = sample_gmm(key_x0, weights_j, means_j, stds_j, cfg.train_batch_size)
        t = jax.random.randint(key_t, (cfg.train_batch_size,), 0, cfg.diffusion_steps)
        noise = jax.random.normal(key_noise, x0.shape)
        x_t = actor.q_sample(t, x0, noise)
        obs = jnp.zeros((cfg.train_batch_size, 1), dtype=x0.dtype)
        pred = actor.eps_pred(params, obs, x_t, t)
        return optax.squared_error(pred, noise).mean()

    @jax.jit
    def train_step(params, opt_state, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, key)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    steps = []
    losses = []
    for step in range(1, cfg.train_steps + 1):
        loop_key, step_key = jax.random.split(loop_key)
        policy_params, opt_state, loss = train_step(policy_params, opt_state, step_key)
        if step == 1 or step == cfg.train_steps or step % cfg.train_log_every == 0:
            steps.append(step)
            losses.append(float(loss))
            print(f"train step {step:>7d}/{cfg.train_steps}: eps_mse={float(loss):.6g}", flush=True)
    return policy_params, np.asarray(steps, dtype=np.int32), np.asarray(losses, dtype=np.float64)


def eps_mse_by_t(actor: ActorCritic, policy_params, cfg: LearnedToyConfig, weights, means, stds):
    weights_j = jnp.asarray(weights, dtype=jnp.float32)
    means_j = jnp.asarray(means, dtype=jnp.float32)
    stds_j = jnp.asarray(stds, dtype=jnp.float32)

    @jax.jit
    def mse_one_t(key, t_idx):
        key_x0, key_noise = jax.random.split(key)
        x0 = sample_gmm(key_x0, weights_j, means_j, stds_j, cfg.diagnostic_batch_size)
        noise = jax.random.normal(key_noise, x0.shape)
        t = jnp.full((cfg.diagnostic_batch_size,), t_idx, dtype=jnp.int32)
        x_t = actor.q_sample(t, x0, noise)
        obs = jnp.zeros((cfg.diagnostic_batch_size, 1), dtype=x0.dtype)
        pred = actor.eps_pred(policy_params, obs, x_t, t)
        x0_hat = (
            x_t * actor.schedule.sqrt_recip_alphas_cumprod[t_idx]
            - pred * actor.schedule.sqrt_recipm1_alphas_cumprod[t_idx]
        )
        return optax.squared_error(pred, noise).mean(), optax.squared_error(x0_hat, x0).mean()

    keys = jax.random.split(jax.random.key(cfg.seed + 2027), cfg.diffusion_steps)
    eps_mse = []
    x0_mse = []
    for t in range(cfg.diffusion_steps):
        eps_err, x0_err = mse_one_t(keys[t], jnp.int32(t))
        eps_mse.append(float(eps_err))
        x0_mse.append(float(x0_err))
    return np.asarray(eps_mse), np.asarray(x0_mse)


def learned_x0_hat_np(model: LearnedToyModel, points, t_idx: int):
    x = jnp.asarray(np.asarray(points, dtype=np.float32)[:, None])
    eps = model.eps_pred(None, None, x, t_idx)
    x0 = (
        x * model.schedule.sqrt_recip_alphas_cumprod[t_idx]
        - eps * model.schedule.sqrt_recipm1_alphas_cumprod[t_idx]
    )
    return np.asarray(x0[:, 0])


def write_metrics(path: Path, rows: list[dict]) -> None:
    fieldnames = [
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
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_training_diagnostics(out: Path, train_steps, train_losses, eps_mse, x0_mse) -> None:
    np.savetxt(
        out / "training_diagnostics.csv",
        np.column_stack([np.arange(len(eps_mse)), eps_mse, x0_mse]),
        delimiter=",",
        header="timestep,eps_mse,x0_hat_mse",
        comments="",
    )
    with open(out / "training_loss.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "eps_mse"])
        writer.writerows(zip(train_steps.tolist(), train_losses.tolist()))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(train_steps, train_losses)
    ax.set_xlabel("train step")
    ax.set_ylabel("eps MSE")
    ax.set_title("Diffusion policy training loss")
    fig.tight_layout()
    fig.savefig(out / "plots" / "training_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(np.arange(len(eps_mse)), eps_mse, label="eps MSE")
    ax.plot(np.arange(len(x0_mse)), x0_mse, label="x0_hat MSE")
    ax.set_xlabel("timestep")
    ax.set_ylabel("MSE")
    ax.set_title("Learned diffusion diagnostics by timestep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plots" / "diffusion_diagnostics_by_t.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    weights = _normalize_weights(_parse_float_list(args.gmm_weights))
    means = np.asarray(_parse_float_list(args.gmm_means), dtype=np.float64)
    stds = np.asarray(_parse_float_list(args.gmm_stds), dtype=np.float64)
    if not (len(weights) == len(means) == len(stds)):
        raise ValueError("GMM weights, means, and stds must have the same length.")
    if np.any(stds <= 0):
        raise ValueError("GMM stds must be positive.")

    cfg = LearnedToyConfig(
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
        eval_timesteps=args.eval_timesteps,
        output_dir=args.output_dir,
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

    if cfg.policy_parameterization == "E" and cfg.policy_final_layer in {"L2", "IP"}:
        warnings.warn(
            "L2/IP final layers constrain the scalar energy form; use ff/default for the closest generic baseline.",
            RuntimeWarning,
        )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))

    actor = make_actor(cfg)
    policy_params, train_steps, train_losses = train_policy(actor, cfg, weights, means, stds)
    eps_mse, x0_mse = eps_mse_by_t(actor, policy_params, cfg, weights, means, stds)
    save_training_diagnostics(out, train_steps, train_losses, eps_mse, x0_mse)

    model = LearnedToyModel(
        actor=actor,
        policy_params=policy_params,
        reward_center=cfg.reward_center,
        reward_scale=cfg.reward_scale,
        schedule=actor.schedule,
        num_timesteps=cfg.diffusion_steps,
        mala_steps=cfg.mala_steps,
        x_recon_clip_radius=cfg.x_recon_clip_radius,
    )

    result = run_toy_mala_sampler(jax.random.key(cfg.seed + 404), model, cfg)
    action_clipped = np.asarray(result.action[:, 0])
    trace = np.asarray(result.trace[:, :, 0])
    raw_x0 = np.asarray(result.raw_x0[:, 0])
    schedule = actor.schedule
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
                      "learned final raw x0 vs clean oracle target")

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
        sample_x0_hat = learned_x0_hat_np(model, trace[t], t)
        grid_x0_hat = np_x0_hat(grid, weights, means, stds, schedule, t, "tweedie")
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
                          f"learned post-MALA x_t at t={t} vs oracle target")

    save_density_panel(
        out / "plots" / "density_panel.png",
        panels,
        (
            f"learned diffusion: mala_steps={cfg.mala_steps}, gradient={cfg.guidance_gradient_space}, "
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
        train_steps=train_steps,
        train_losses=train_losses,
        eps_mse_by_t=eps_mse,
        x0_hat_mse_by_t=x0_mse,
    )
    write_metrics(out / "metrics.csv", rows)

    print(f"wrote results to {out}")
    print(f"final train eps_mse {train_losses[-1]:.6g}")
    for row in rows:
        print(
            f"{row['stage']:>20s} t={row['timestep']:>3} "
            f"W1={row['w1']:.5f} KS={row['ks']:.5f} JS={row['js']:.5f} "
            f"acc={row['acceptance_rate']:.3f} clip={row['clip_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
