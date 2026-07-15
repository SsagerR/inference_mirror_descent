#!/usr/bin/env python3
"""Compute-budget comparison: RSM target diffusion vs MALA distillation.

Pipeline shared by all methods:

1. Train a base diffusion model on pi_orig.
2. Use the base model to acquire target-training data under a fixed acquisition
   compute budget.
3. Train a target diffusion model on that acquired data.
4. Sample the target diffusion model and evaluate against the tilted toy target.

The key budget convention is:

* one ordinary diffusion sample costs 1 unit;
* one MALA-corrected sample costs ``mala_cost_ratio`` units;
* therefore RSM receives ``B`` ordinary diffusion samples, while MALA
  distillation receives ``floor(B / mala_cost_ratio)`` MALA samples.

Reward normalization is computed once from the ordinary diffusion samples for
each ``(dim, dataset_seed, compute_budget, beta)`` cell.  That same mean/std
defines the target for RSM, MALA distillation, and evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.toy_mala import toy_mala_exponential_vs_mala as expmala


METHODS = ["pi_orig_distill", "rsm", "mala_distill"]
METHOD_SPECS = ["all", "rsm_refs", *METHODS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", choices=["1d", "2d"], required=True)
    p.add_argument("--mode", choices=["run", "benchmark_cost"], default="run")
    p.add_argument("--methods", choices=METHOD_SPECS, default="all")
    p.add_argument("--compute_budget", type=int, required=True)
    p.add_argument("--mala_cost_ratio", type=float, default=1.0)
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
    p.add_argument("--mala_steps", type=int, default=2)
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
    p.add_argument("--train_batch_size", type=int, default=4096)
    p.add_argument("--train_lr", type=float, default=3e-4)
    p.add_argument("--train_log_every", type=int, default=500)
    p.add_argument("--diagnostic_batch_size", type=int, default=20000)
    p.add_argument("--orig_train_budget", type=int, default=100000)
    p.add_argument("--orig_train_steps", type=int, default=30000)
    p.add_argument("--target_train_steps", type=int, default=20000)
    p.add_argument("--benchmark_samples", type=int, default=20000)
    p.add_argument("--benchmark_repeats", type=int, default=3)
    return p.parse_args()


def method_list(spec: str) -> list[str]:
    if spec == "all":
        return METHODS
    if spec == "rsm_refs":
        return ["pi_orig_distill", "rsm"]
    return [spec]


def mala_distill_sample_count(compute_budget: int, mala_cost_ratio: float) -> int:
    if compute_budget <= 0:
        raise ValueError("compute_budget must be positive")
    if not np.isfinite(mala_cost_ratio) or mala_cost_ratio <= 0:
        raise ValueError("mala_cost_ratio must be positive")
    return max(1, int(math.floor(float(compute_budget) / float(mala_cost_ratio))))


def with_train_steps(args: argparse.Namespace, train_steps: int, *, num_samples: int | None = None) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    out.train_steps = int(train_steps)
    if num_samples is not None:
        out.num_samples = int(num_samples)
    return out


def make_cfg(dim: str, args: argparse.Namespace, *, beta_for_sampler: float, sampler: str, output_dir: str):
    if dim == "1d":
        return expmala.make_1d_cfg(args, beta_for_sampler=beta_for_sampler, sampler=sampler, output_dir=output_dir)
    return expmala.make_2d_cfg(args, beta_for_sampler=beta_for_sampler, sampler=sampler, output_dir=output_dir)


def make_actor(dim: str, cfg):
    return expmala.make_actor_1d(cfg) if dim == "1d" else expmala.make_actor_2d(cfg)


def sample_true_pi_orig(dim: str, args: argparse.Namespace, weights, means, extra, n: int) -> np.ndarray:
    expmala.load_runtime()
    key = expmala.jax.random.key(args.seed + 100_003 * (args.dataset_seed + 1) + 17 * n)
    if dim == "1d":
        stds = extra
        return np.asarray(
            expmala.sample_gmm_1d(
                key,
                expmala.jnp.asarray(weights, dtype=expmala.jnp.float32),
                expmala.jnp.asarray(means, dtype=expmala.jnp.float32),
                expmala.jnp.asarray(stds, dtype=expmala.jnp.float32),
                n,
            )
        )
    covs = extra
    return np.asarray(
        expmala.sample_gmm_2d(
            key,
            expmala.jnp.asarray(weights, dtype=expmala.jnp.float32),
            expmala.jnp.asarray(means, dtype=expmala.jnp.float32),
            expmala.jnp.asarray(covs, dtype=expmala.jnp.float32),
            n,
        )
    )


def reward_values(dim: str, samples: np.ndarray, cfg) -> np.ndarray:
    if dim == "1d":
        return expmala.np_reward_1d(samples[:, 0], cfg.reward_center, cfg.reward_scale, cfg)
    return expmala.np_reward_2d(samples, cfg)


def build_model(dim: str, actor, policy_params, cfg):
    if dim == "1d":
        return expmala.LearnedToyModel(
            actor=actor,
            policy_params=policy_params,
            reward_center=cfg.reward_center,
            reward_scale=cfg.reward_scale,
            schedule=actor.schedule,
            num_timesteps=cfg.diffusion_steps,
            mala_steps=cfg.mala_steps,
            x_recon_clip_radius=cfg.x_recon_clip_radius,
        )
    return expmala.LearnedToy2DModel(
        actor=actor,
        policy_params=policy_params,
        reward_type=cfg.reward_type,
        reward_center=expmala.jnp.asarray(cfg.reward_center, dtype=expmala.jnp.float32),
        reward_scales=expmala.jnp.asarray(cfg.reward_scales, dtype=expmala.jnp.float32),
        reward_bump_centers=expmala.jnp.asarray(cfg.reward_bump_centers, dtype=expmala.jnp.float32),
        reward_bump_widths=expmala.jnp.asarray(cfg.reward_bump_widths, dtype=expmala.jnp.float32),
        reward_bump_weights=expmala.jnp.asarray(cfg.reward_bump_weights, dtype=expmala.jnp.float32),
        reward_sin_amp=float(cfg.reward_sin_amp),
        reward_sin_freqs=expmala.jnp.asarray(cfg.reward_sin_freqs, dtype=expmala.jnp.float32),
        reward_sin_phase=float(cfg.reward_sin_phase),
        reward_l2=float(cfg.reward_l2),
        reward_l4=float(cfg.reward_l4),
        schedule=actor.schedule,
        num_timesteps=cfg.diffusion_steps,
        mala_steps=cfg.mala_steps,
        x_recon_clip_radius=float(cfg.x_recon_clip_radius),
    )


def sample_with_policy(
    dim: str,
    policy_params,
    cfg,
    *,
    n_samples: int,
    seed_offset: int,
    use_mala: bool,
) -> tuple[np.ndarray, float, object]:
    cfg_sample = replace(cfg, num_samples=int(n_samples))
    if not use_mala:
        cfg_sample = expmala.canonical_pi_orig_diffusion_sampling_cfg(cfg_sample)
    actor = make_actor(dim, cfg_sample)
    if use_mala:
        cfg_sample = expmala.with_mala_steps_per_level(actor, cfg_sample)
    model = build_model(dim, actor, policy_params, cfg_sample)
    key = expmala.jax.random.key(cfg.seed + seed_offset)
    if dim == "1d":
        result = expmala.run_toy_mala_sampler_1d(key, model, cfg_sample)
        samples = np.asarray(result.raw_x0)
    else:
        result = expmala.run_toy_mala_sampler_2d(key, model, cfg_sample)
        samples = np.asarray(result.raw_x0)
    acceptance = float(np.nanmean(np.asarray(result.per_level_acc)))
    return samples, acceptance, cfg_sample


def evaluate_target_samples(
    dim: str,
    samples: np.ndarray,
    cfg_eval,
    weights,
    means,
    extra,
    seed: int,
    *,
    panel_path: Path | None = None,
    method: str | None = None,
) -> dict:
    actor = make_actor(dim, cfg_eval)
    if dim == "1d":
        stds = extra
        raw_x0 = np.asarray(samples[:, 0])
        grid = expmala.make_eval_grid_1d(raw_x0, cfg_eval, weights, means, stds, actor.schedule, None)
        target = expmala.target_density_1d(grid, cfg_eval, weights, means, stds, actor.schedule, None)
        if panel_path is not None:
            expmala.save_panel_data_1d(panel_path, method=method or "unknown", samples=raw_x0, grid=grid, target=target)
        return expmala.metrics_from_samples_1d(raw_x0, grid, target, cfg_eval)
    covs = extra
    grid_x, grid_y = expmala.make_eval_grid_2d(samples, cfg_eval, means, covs, actor.schedule, None)
    target = expmala.target_density_grid_2d(grid_x, grid_y, cfg_eval, weights, means, covs, actor.schedule, None)
    if panel_path is not None:
        expmala.save_panel_data_2d(panel_path, method=method or "unknown", samples=samples, grid_x=grid_x, grid_y=grid_y, target=target)
    rng = np.random.default_rng(seed)
    return expmala.metrics_from_samples_2d(samples, grid_x, grid_y, target, cfg_eval, rng)


def train_diffusion(dim: str, cfg, dataset: np.ndarray, *, normalized_rewards: np.ndarray | None, beta: float, seed_offset: int):
    actor = make_actor(dim, cfg)
    return expmala.train_policy_on_dataset(
        actor,
        cfg,
        dataset,
        normalized_rewards=normalized_rewards,
        beta=beta,
        seed_offset=seed_offset,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def add_common(
    metrics: dict,
    *,
    args: argparse.Namespace,
    method: str,
    acquisition_sample_count: int,
    reward_stats: dict,
    weight_stats: dict,
    train_loss: float,
    acquisition_acceptance_rate: float,
) -> dict:
    out = dict(metrics)
    out.update(
        {
            "stage": "final_clean",
            "timestep": -1,
            "dim": args.dim,
            "method": method,
            "dataset_seed": args.dataset_seed,
            "compute_budget": args.compute_budget,
            "acquisition_sample_count": acquisition_sample_count,
            "beta_input": args.beta,
            "mala_steps": args.mala_steps if method == "mala_distill" else 0,
            "mala_cost_ratio": args.mala_cost_ratio,
            "orig_train_budget": args.orig_train_budget,
            "orig_train_steps": args.orig_train_steps,
            "target_train_steps": args.target_train_steps,
            "train_loss_final": train_loss,
            "acquisition_acceptance_rate": acquisition_acceptance_rate,
        }
    )
    out.update(reward_stats)
    out.update(weight_stats)
    return out


def run_pipeline(args: argparse.Namespace, out: Path) -> list[dict]:
    expmala.load_runtime()
    base_args = with_train_steps(args, args.orig_train_steps, num_samples=args.num_samples)
    cfg_base, weights, means, extra, *_rest = make_cfg(args.dim, base_args, beta_for_sampler=0.0, sampler="mala", output_dir=str(out))
    base_dataset = sample_true_pi_orig(args.dim, args, weights, means, extra, args.orig_train_budget)
    base_policy, base_steps, base_losses = train_diffusion(
        args.dim,
        expmala.canonical_pi_orig_diffusion_sampling_cfg(cfg_base),
        base_dataset,
        normalized_rewards=None,
        beta=0.0,
        seed_offset=101,
    )
    np.savetxt(out / "training_loss_base_pi_orig.csv", np.column_stack([base_steps, base_losses]), delimiter=",", header="step,loss", comments="")

    rsm_samples, ordinary_acceptance, _ordinary_cfg = sample_with_policy(
        args.dim,
        base_policy,
        cfg_base,
        n_samples=args.compute_budget,
        seed_offset=1001,
        use_mala=False,
    )
    raw_rewards = reward_values(args.dim, rsm_samples, cfg_base)
    norm_rewards, reward_stats = expmala.dataset_standardize_rewards(raw_rewards)
    weight_stats = expmala.weight_diagnostics(norm_rewards, args.beta)
    beta_sampler = args.beta / (reward_stats["reward_std_raw"] + 1e-8)
    cfg_eval, weights_eval, means_eval, extra_eval, *_ = make_cfg(
        args.dim,
        with_train_steps(args, args.target_train_steps),
        beta_for_sampler=beta_sampler,
        sampler="mala",
        output_dir=str(out),
    )

    rows = []
    target_args = with_train_steps(args, args.target_train_steps, num_samples=args.num_samples)
    cfg_target, _w, _m, _e, *_ = make_cfg(args.dim, target_args, beta_for_sampler=0.0, sampler="mala", output_dir=str(out))
    cfg_target_unguided = expmala.canonical_pi_orig_diffusion_sampling_cfg(cfg_target)

    selected = method_list(args.methods)
    if "pi_orig_distill" in selected:
        params, steps, losses = train_diffusion(args.dim, cfg_target_unguided, rsm_samples, normalized_rewards=None, beta=0.0, seed_offset=201)
        samples, _acc, _cfg = sample_with_policy(args.dim, params, cfg_target_unguided, n_samples=args.num_samples, seed_offset=2001, use_mala=False)
        metrics = evaluate_target_samples(
            args.dim,
            samples,
            cfg_eval,
            weights_eval,
            means_eval,
            extra_eval,
            args.seed + 301,
            panel_path=out / "panel_data" / "pi_orig_distill.npz",
            method="pi_orig_distill",
        )
        rows.append(
            add_common(
                metrics,
                args=args,
                method="pi_orig_distill",
                acquisition_sample_count=len(rsm_samples),
                reward_stats=reward_stats,
                weight_stats=weight_stats,
                train_loss=float(losses[-1]),
                acquisition_acceptance_rate=ordinary_acceptance,
            )
        )
        np.savetxt(out / "training_loss_pi_orig_distill.csv", np.column_stack([steps, losses]), delimiter=",", header="step,loss", comments="")

    if "rsm" in selected:
        params, steps, losses = train_diffusion(args.dim, cfg_target_unguided, rsm_samples, normalized_rewards=norm_rewards, beta=args.beta, seed_offset=301)
        samples, _acc, _cfg = sample_with_policy(args.dim, params, cfg_target_unguided, n_samples=args.num_samples, seed_offset=3001, use_mala=False)
        metrics = evaluate_target_samples(
            args.dim,
            samples,
            cfg_eval,
            weights_eval,
            means_eval,
            extra_eval,
            args.seed + 401,
            panel_path=out / "panel_data" / "rsm.npz",
            method="rsm",
        )
        rows.append(
            add_common(
                metrics,
                args=args,
                method="rsm",
                acquisition_sample_count=len(rsm_samples),
                reward_stats=reward_stats,
                weight_stats=weight_stats,
                train_loss=float(losses[-1]),
                acquisition_acceptance_rate=ordinary_acceptance,
            )
        )
        np.savetxt(out / "training_loss_rsm.csv", np.column_stack([steps, losses]), delimiter=",", header="step,loss", comments="")

    if "mala_distill" in selected:
        n_mala = mala_distill_sample_count(args.compute_budget, args.mala_cost_ratio)
        mala_cfg = replace(cfg_base, sampler="mala", beta=beta_sampler, num_samples=n_mala)
        mala_samples, mala_acceptance, _mala_cfg = sample_with_policy(
            args.dim,
            base_policy,
            mala_cfg,
            n_samples=n_mala,
            seed_offset=4001,
            use_mala=True,
        )
        params, steps, losses = train_diffusion(args.dim, cfg_target_unguided, mala_samples, normalized_rewards=None, beta=0.0, seed_offset=401)
        samples, _acc, _cfg = sample_with_policy(args.dim, params, cfg_target_unguided, n_samples=args.num_samples, seed_offset=5001, use_mala=False)
        metrics = evaluate_target_samples(
            args.dim,
            samples,
            cfg_eval,
            weights_eval,
            means_eval,
            extra_eval,
            args.seed + 501,
            panel_path=out / "panel_data" / "mala_distill.npz",
            method="mala_distill",
        )
        rows.append(
            add_common(
                metrics,
                args=args,
                method="mala_distill",
                acquisition_sample_count=len(mala_samples),
                reward_stats=reward_stats,
                weight_stats=weight_stats,
                train_loss=float(losses[-1]),
                acquisition_acceptance_rate=mala_acceptance,
            )
        )
        np.savetxt(out / "training_loss_mala_distill.csv", np.column_stack([steps, losses]), delimiter=",", header="step,loss", comments="")

    np.savez_compressed(
        out / "acquisition_data.npz",
        rsm_samples=rsm_samples.astype(np.float32),
        normalized_rewards=norm_rewards.astype(np.float32),
    )
    return rows


def benchmark_cost(args: argparse.Namespace, out: Path) -> list[dict]:
    expmala.load_runtime()
    bench_args = with_train_steps(args, max(1, min(args.orig_train_steps, 2000)), num_samples=args.benchmark_samples)
    cfg_base, weights, means, extra, *_ = make_cfg(args.dim, bench_args, beta_for_sampler=0.0, sampler="mala", output_dir=str(out))
    base_dataset = sample_true_pi_orig(args.dim, args, weights, means, extra, min(args.orig_train_budget, args.benchmark_samples))
    base_policy, _steps, _losses = train_diffusion(
        args.dim,
        expmala.canonical_pi_orig_diffusion_sampling_cfg(cfg_base),
        base_dataset,
        normalized_rewards=None,
        beta=0.0,
        seed_offset=701,
    )
    ordinary_times = []
    mala_times = []
    for repeat in range(args.benchmark_repeats + 1):
        start = time.perf_counter()
        ordinary_samples, _ordinary_acc, _ = sample_with_policy(
            args.dim,
            base_policy,
            cfg_base,
            n_samples=args.benchmark_samples,
            seed_offset=8001 + repeat,
            use_mala=False,
        )
        expmala.jax.block_until_ready(expmala.jnp.asarray(ordinary_samples))
        ordinary_elapsed = time.perf_counter() - start

        raw_rewards = reward_values(args.dim, ordinary_samples, cfg_base)
        _norm_rewards, reward_stats = expmala.dataset_standardize_rewards(raw_rewards)
        beta_sampler = args.beta / (reward_stats["reward_std_raw"] + 1e-8)
        mala_cfg = replace(cfg_base, sampler="mala", beta=beta_sampler)
        start = time.perf_counter()
        mala_samples, mala_acc, _ = sample_with_policy(
            args.dim,
            base_policy,
            mala_cfg,
            n_samples=args.benchmark_samples,
            seed_offset=9001 + repeat,
            use_mala=True,
        )
        expmala.jax.block_until_ready(expmala.jnp.asarray(mala_samples))
        mala_elapsed = time.perf_counter() - start
        if repeat > 0:
            ordinary_times.append(ordinary_elapsed)
            mala_times.append(mala_elapsed)
        print(
            f"benchmark repeat {repeat}: ordinary={ordinary_elapsed:.4f}s "
            f"mala={mala_elapsed:.4f}s acc={mala_acc:.3f}",
            flush=True,
        )
    ordinary_mean = float(np.mean(ordinary_times))
    mala_mean = float(np.mean(mala_times))
    ratio = mala_mean / ordinary_mean if ordinary_mean > 0 else float("nan")
    return [
        {
            "dim": args.dim,
            "method": "benchmark_cost",
            "benchmark_samples": args.benchmark_samples,
            "benchmark_repeats": args.benchmark_repeats,
            "mala_steps": args.mala_steps,
            "ordinary_seconds_mean": ordinary_mean,
            "mala_seconds_mean": mala_mean,
            "mala_cost_ratio": ratio,
        }
    ]


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    if args.mode == "benchmark_cost":
        rows = benchmark_cost(args, out)
        write_csv(out / "cost_benchmark.csv", rows)
    else:
        rows = run_pipeline(args, out)
        write_csv(out / "metrics.csv", rows)
    print(f"wrote results to {out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
