#!/usr/bin/env python3
"""Visualize 2D oracle guided targets across diffusion levels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from relax.utils.diffusion import build_beta_schedule

from toy_mala_2d import (
    Toy2DConfig,
    apply_target_preset,
    effective_beta,
    eval_timesteps,
    make_covariances,
    normalize_grid_density,
    normalize_weights,
    np_base_logpdf,
    np_reward,
    parse_float_list,
    parse_matrix_rows,
    parse_vector,
    save_orig_target_comparison,
    save_surface_plot,
    target_density_grid,
    validate_reward_params,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target_preset", choices=["manual", "complex_2d_v1", "complex_2d_v2"], default="complex_2d_v2")
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
    p.add_argument("--diffusion_steps", type=int, default=20)
    p.add_argument("--beta_schedule_type", choices=["linear", "cosine", "constant_kl"], default="cosine")
    p.add_argument("--snr_max", type=float, default=124.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--advantage_normalization", action="store_true")
    p.add_argument("--initial_advantage_second_moment_ema", type=float, default=1.0)
    p.add_argument("--x0_hat_method", choices=["posterior_mean", "tweedie"], default="tweedie")
    p.add_argument("--x0_hat_clip_radius", type=float, default=10.0)
    p.add_argument("--grid_min", type=float, default=-2.6)
    p.add_argument("--grid_max", type=float, default=2.6)
    p.add_argument("--grid_points", type=int, default=181)
    p.add_argument("--eval_timesteps", default="auto")
    p.add_argument("--output_dir", type=Path, default=Path("scripts/toy_mala/target_viz_complex_2d_v2_tweedie"))
    return apply_target_preset(p.parse_args())


def grid_js(p: np.ndarray, q: np.ndarray, dx: float, dy: float) -> float:
    p_mass = p * dx * dy
    q_mass = q * dx * dy
    p_mass = p_mass / p_mass.sum()
    q_mass = q_mass / q_mass.sum()
    m = 0.5 * (p_mass + q_mass)
    eps = 1e-12
    return float(
        0.5 * np.sum(p_mass * (np.log(p_mass + eps) - np.log(m + eps)))
        + 0.5 * np.sum(q_mass * (np.log(q_mass + eps) - np.log(m + eps)))
    )


def save_target_panel(path: Path, panels: list[dict], title: str) -> None:
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.7 * rows), squeeze=False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax, panel in zip(axes.ravel(), panels):
        im = ax.contourf(panel["grid_x"], panel["grid_y"], panel["density"], levels=30, cmap="viridis")
        ax.contour(panel["grid_x"], panel["grid_y"], panel["density"], levels=10, colors="white", linewidths=0.35)
        ax.set_title(panel["label"])
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    weights = normalize_weights(parse_float_list(args.gmm_weights))
    means = np.asarray(parse_matrix_rows(args.gmm_means), dtype=np.float64)
    stds = np.asarray(parse_matrix_rows(args.gmm_stds), dtype=np.float64)
    corrs = np.asarray(parse_float_list(args.gmm_corrs), dtype=np.float64)
    covs = make_covariances(stds, corrs)
    cfg = Toy2DConfig(
        target_preset=args.target_preset,
        gmm_weights=weights.tolist(),
        gmm_means=means.tolist(),
        gmm_stds=stds.tolist(),
        gmm_corrs=corrs.tolist(),
        reward_type=args.reward_type,
        reward_center=parse_vector(args.reward_center),
        reward_scales=parse_vector(args.reward_scales),
        reward_bump_centers=parse_matrix_rows(args.reward_bump_centers),
        reward_bump_widths=parse_matrix_rows(args.reward_bump_widths),
        reward_bump_weights=parse_float_list(args.reward_bump_weights),
        reward_sin_amp=args.reward_sin_amp,
        reward_sin_freqs=parse_vector(args.reward_sin_freqs),
        reward_sin_phase=args.reward_sin_phase,
        reward_l2=args.reward_l2,
        reward_l4=args.reward_l4,
        num_samples=0,
        seed=0,
        diffusion_steps=args.diffusion_steps,
        beta_schedule_type=args.beta_schedule_type,
        snr_max=args.snr_max,
        alpha=args.alpha,
        beta=args.beta,
        sampler="mala",
        mala_steps=0,
        mala_eta=1.0,
        langevin_steps=0,
        langevin_eta=1.0,
        denoising_predictor="DDIM",
        guidance_gradient_space="xt",
        x0_hat_method=args.x0_hat_method,
        x0_hat_clip_radius=args.x0_hat_clip_radius,
        x_recon_clip_radius=1.0,
        mala_adapt_rate=0.0,
        guidance_strength_multiplier=1.0,
        batch_independent_guidance=False,
        advantage_normalization=args.advantage_normalization,
        initial_advantage_second_moment_ema=args.initial_advantage_second_moment_ema,
        grid_mode="fixed",
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        grid_points=args.grid_points,
        metric_target_samples=0,
        metric_slices=0,
        eval_timesteps=args.eval_timesteps,
        output_dir=str(args.output_dir),
    )
    validate_reward_params(cfg)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "surfaces").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))

    schedule = build_beta_schedule(cfg.diffusion_steps, cfg.beta_schedule_type, cfg.snr_max)
    grid_x = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_points)
    grid_y = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_points)
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    clean_orig = np.exp(np_base_logpdf(points, weights, means, covs, schedule, None)).reshape(len(grid_y), len(grid_x))
    clean_target = target_density_grid(grid_x, grid_y, cfg, weights, means, covs, schedule, None)
    clean_q = np_reward(points, cfg).reshape(len(grid_y), len(grid_x))
    save_orig_target_comparison(
        out / "orig_vs_target_contours.png",
        grid_x,
        grid_y,
        clean_orig,
        clean_target,
        clean_q,
        f"{cfg.target_preset}: clean pi_orig vs pi_target",
    )
    save_surface_plot(out / "surfaces" / "clean_pi_orig_surface.png", grid_x, grid_y, clean_orig, "clean pi_orig")
    save_surface_plot(out / "surfaces" / "clean_pi_target_surface.png", grid_x, grid_y, clean_target, "clean pi_target")

    panels = [
        {"label": "clean pi_orig", "grid_x": grid_x, "grid_y": grid_y, "density": clean_orig},
        {"label": "clean pi_target", "grid_x": grid_x, "grid_y": grid_y, "density": clean_target},
    ]
    for t in eval_timesteps(cfg.eval_timesteps, cfg.diffusion_steps):
        dens = target_density_grid(grid_x, grid_y, cfg, weights, means, covs, schedule, t)
        panels.append({"label": f"guided target t={t}", "grid_x": grid_x, "grid_y": grid_y, "density": dens})
        save_surface_plot(out / "surfaces" / f"target_t{t:03d}_surface.png", grid_x, grid_y, dens, f"target t={t}")
    save_target_panel(out / "target_panel.png", panels, f"{cfg.target_preset}: guided 2D targets")

    dx = float(grid_x[1] - grid_x[0])
    dy = float(grid_y[1] - grid_y[0])
    diagnostics = {
        "beta_eff": effective_beta(cfg),
        "clean_js_pi_orig_pi_target": grid_js(clean_orig, clean_target, dx, dy),
        "clean_orig_mean_q": float(np.sum(clean_orig * clean_q) * dx * dy / max(np.sum(clean_orig) * dx * dy, 1e-12)),
        "clean_target_mean_q": float(np.sum(clean_target * clean_q) * dx * dy),
        "outputs": {
            "orig_vs_target_contours": str(out / "orig_vs_target_contours.png"),
            "target_panel": str(out / "target_panel.png"),
            "surfaces": str(out / "surfaces"),
        },
    }
    (out / "target_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(f"wrote target visualization to {out}")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
