#!/usr/bin/env python3
"""2D toy MALA experiment with a learned diffusion policy.

This is the learned-score analogue of ``toy_mala_2d.py``.  The target GMM,
reward, MALA sampler, denoising logic, metrics, and plots stay aligned with
the oracle 2D script; only the diffusion base model is replaced by an
ActorCritic diffusion policy trained by epsilon MSE.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from relax.network.actor_critic import ActorCritic
from scripts.toy_mala.denoising_schedule import DENOISING_SCHEDULE_CHOICES
from scripts.toy_mala.guidance_schedule import GUIDANCE_SCHEDULE_CHOICES
from scripts.toy_mala.mala_step_schedule import MALA_STEP_SCHEDULE_CHOICES, allocate_mala_steps
from scripts.toy_mala.toy_mala_2d import (
    apply_target_preset,
    eval_timesteps,
    jax_reward,
    make_covariances,
    make_eval_grid,
    metrics_from_samples,
    normalize_weights,
    np_base_logpdf,
    np_reward,
    np_x0_hat,
    parse_float_list,
    parse_matrix_rows,
    parse_vector,
    run_toy_mala_sampler,
    save_contour_panel,
    save_contour_plot,
    save_orig_target_comparison,
    save_surface_plot,
    target_density_grid,
    validate_reward_params,
    validate_sampler_params,
)


@dataclass(frozen=True)
class LearnedToy2DConfig:
    score_source: str
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
    denoising_schedule: str
    guidance_gradient_space: str
    x0_hat_method: str
    x0_hat_clip_radius: float
    x_recon_clip_radius: float
    action_clip_radius: float
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
class LearnedToy2DModel:
    actor: ActorCritic
    policy_params: object
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


def _mish(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jax.nn.softplus(x))


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
    p.add_argument("--guidance_schedule", choices=GUIDANCE_SCHEDULE_CHOICES, default="constant")
    p.add_argument("--sampler", choices=["mala", "langevin", "dps", "mpgd", "unguided"], default="mala")
    p.add_argument("--mala_steps", type=int, default=4)
    p.add_argument("--mala_budget", type=int, default=None,
                   help="Total MALA step budget across diffusion levels. Defaults to mala_steps * diffusion_steps.")
    p.add_argument("--mala_step_schedule", choices=MALA_STEP_SCHEDULE_CHOICES, default="constant")
    p.add_argument("--mala_eta", type=float, default=1.0)
    p.add_argument("--langevin_steps", type=int, default=4)
    p.add_argument("--langevin_eta", type=float, default=1.0)
    p.add_argument("--denoising_predictor", choices=["Identity", "DDPM_mean", "DDIM"], default="DDPM_mean")
    p.add_argument("--denoising_schedule", choices=DENOISING_SCHEDULE_CHOICES, default="from_predictor")
    p.add_argument("--guidance_gradient_space", choices=["xt", "x0hat", "x0hatclipped"], default="xt")
    p.add_argument("--x0_hat_method", choices=["tweedie"], default="tweedie",
                   help="Learned diffusion has no closed-form posterior mean; use production-style Tweedie.")
    p.add_argument("--x0_hat_clip_radius", type=float, default=1_000_000.0,
                   help="Guidance x0_hat clip radius. Default is intentionally huge so toy experiments do not inherit bounded-action clipping.")
    p.add_argument("--x_recon_clip_radius", type=float, default=1_000_000.0,
                   help="DDPM_mean clean reconstruction clip radius. Default is intentionally huge for unbounded toy targets; pass 1.0 explicitly to mimic bounded-action MGMD.")
    p.add_argument("--action_clip_radius", type=float, default=1_000_000.0,
                   help="Final action clip radius used only for the final_clipped_action diagnostic stage. Default is intentionally huge for toy experiments.")
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
    return apply_target_preset(p.parse_args())


def sample_gmm(key, weights, means, covs, batch_size: int):
    key_comp, key_noise = jax.random.split(key)
    comp = jax.random.categorical(key_comp, jnp.log(weights), shape=(batch_size,))
    noise = jax.random.normal(key_noise, (batch_size, 2))
    chol = jnp.linalg.cholesky(covs)
    return means[comp] + jnp.einsum("bij,bj->bi", chol[comp], noise)


def make_actor(cfg: LearnedToy2DConfig) -> ActorCritic:
    hidden_sizes = [cfg.hidden_dim] * cfg.hidden_num
    diffusion_hidden_sizes = [cfg.diffusion_hidden_dim] * cfg.hidden_num
    return ActorCritic.create(
        obs_dim=1,
        act_dim=2,
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


def train_policy(actor: ActorCritic, cfg: LearnedToy2DConfig, weights, means, covs):
    weights_j = jnp.asarray(weights, dtype=jnp.float32)
    means_j = jnp.asarray(means, dtype=jnp.float32)
    covs_j = jnp.asarray(covs, dtype=jnp.float32)
    opt = optax.adam(cfg.train_lr)

    init_key, loop_key = jax.random.split(jax.random.key(cfg.seed + 17))
    policy_params = actor.init_params(init_key).policy
    opt_state = opt.init(policy_params)

    def loss_fn(params, key):
        key_x0, key_t, key_noise = jax.random.split(key, 3)
        x0 = sample_gmm(key_x0, weights_j, means_j, covs_j, cfg.train_batch_size)
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


def eps_mse_by_t(actor: ActorCritic, policy_params, cfg: LearnedToy2DConfig, weights, means, covs):
    weights_j = jnp.asarray(weights, dtype=jnp.float32)
    means_j = jnp.asarray(means, dtype=jnp.float32)
    covs_j = jnp.asarray(covs, dtype=jnp.float32)

    @jax.jit
    def mse_one_t(key, t_idx):
        key_x0, key_noise = jax.random.split(key)
        x0 = sample_gmm(key_x0, weights_j, means_j, covs_j, cfg.diagnostic_batch_size)
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


def learned_x0_hat_np(model: LearnedToy2DModel, points, t_idx: int):
    x = jnp.asarray(np.asarray(points, dtype=np.float32))
    eps = model.eps_pred(None, None, x, t_idx)
    x0 = (
        x * model.schedule.sqrt_recip_alphas_cumprod[t_idx]
        - eps * model.schedule.sqrt_recipm1_alphas_cumprod[t_idx]
    )
    return np.asarray(x0)


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


def write_metrics(path: Path, rows: list[dict]) -> None:
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
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


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

    cfg = LearnedToy2DConfig(
        score_source="learned",
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
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        grid_points=args.grid_points,
        metric_target_samples=args.metric_target_samples,
        metric_slices=args.metric_slices,
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
    validate_reward_params(cfg)
    validate_sampler_params(cfg)

    clip_cover = float(np.max(np.abs(means) + 4.0 * stds))
    if np.isfinite(cfg.x0_hat_clip_radius) and clip_cover > cfg.x0_hat_clip_radius:
        warnings.warn(
            f"x0_hat_clip_radius={cfg.x0_hat_clip_radius:g} may clip GMM mass; "
            f"max |mean|+4*std is {clip_cover:g}.",
            RuntimeWarning,
        )
    if cfg.policy_parameterization == "E" and cfg.policy_final_layer in {"L2", "IP"}:
        warnings.warn(
            "L2/IP final layers constrain the scalar energy form; use ff/default for the closest generic baseline.",
            RuntimeWarning,
        )

    actor = make_actor(cfg)
    mala_steps_per_level = allocate_mala_steps(
        cfg.mala_step_schedule,
        diffusion_steps=cfg.diffusion_steps,
        mala_budget=cfg.mala_budget,
        sqrt_alphas_cumprod=np.asarray(actor.schedule.sqrt_alphas_cumprod),
        mala_steps=cfg.mala_steps,
    )
    cfg = replace(cfg, mala_budget=int(sum(mala_steps_per_level)), mala_steps_per_level=list(mala_steps_per_level))

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))

    policy_params, train_steps, train_losses = train_policy(actor, cfg, weights, means, covs)
    eps_mse, x0_mse = eps_mse_by_t(actor, policy_params, cfg, weights, means, covs)
    save_training_diagnostics(out, train_steps, train_losses, eps_mse, x0_mse)

    model = LearnedToy2DModel(
        actor=actor,
        policy_params=policy_params,
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
        schedule=actor.schedule,
        num_timesteps=cfg.diffusion_steps,
        mala_steps=cfg.mala_steps,
        x_recon_clip_radius=float(cfg.x_recon_clip_radius),
    )

    result = run_toy_mala_sampler(jax.random.key(cfg.seed + 404), model, cfg)
    raw_x0 = np.asarray(result.raw_x0)
    action_clipped = np.asarray(result.action)
    trace = np.asarray(result.trace)
    eval_ts = eval_timesteps(cfg.eval_timesteps, cfg.diffusion_steps)
    rng = np.random.default_rng(cfg.seed + 2029)
    schedule = actor.schedule

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
    final_metrics.update({"stage": "final_clean", "timestep": -1, "acceptance_rate": float(np.asarray(result.per_level_acc)[0])})
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
                      "learned final raw x0 vs clean target")

    clipped_metrics = metrics_from_samples(action_clipped, grid_x, grid_y, final_target, cfg, rng)
    clipped_metrics.update({"stage": "final_clipped_action", "timestep": -1, "acceptance_rate": float("nan")})
    rows.append(clipped_metrics)

    for t in eval_ts:
        samples_t = trace[t]
        grid_x, grid_y = make_eval_grid(samples_t, cfg, means, covs, schedule, t)
        dens = target_density_grid(grid_x, grid_y, cfg, weights, means, covs, schedule, t)
        sample_x0_hat = learned_x0_hat_np(model, samples_t, t)
        if np.isfinite(cfg.x0_hat_clip_radius):
            sample_x0_hat = np.clip(sample_x0_hat, -cfg.x0_hat_clip_radius, cfg.x0_hat_clip_radius)

        def target_x0_hat(points, t_idx=t):
            x0_hat = np_x0_hat(points, weights, means, covs, schedule, t_idx, "tweedie")
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
        metrics.update({"stage": "intermediate", "timestep": int(t), "acceptance_rate": float(np.asarray(result.per_level_acc)[t])})
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
            f"learned post-MALA x_t at t={t} vs oracle target",
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
            f"learned diffusion: sampler={cfg.sampler}, mala_schedule={cfg.mala_step_schedule}, "
            f"budget={cfg.mala_budget}, mala_steps={cfg.mala_steps}, gradient={cfg.guidance_gradient_space}, "
            f"predictor={cfg.denoising_predictor}, denoise_schedule={cfg.denoising_schedule}, beta={cfg.beta}"
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
            f"SW1={row['sliced_w1']:.5f} JS={row['js']:.5f} "
            f"acc={row['acceptance_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
