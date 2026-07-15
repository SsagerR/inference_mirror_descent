#!/usr/bin/env python3
"""Compute pi_orig-vs-target references for exp-vs-MALA toy sweeps.

This script is intentionally pure NumPy/Matplotlib so it can run locally
without JAX.  It reads ``sweep_metrics.csv`` and computes a clean-distribution
reference: treat samples as coming from pi_orig, compare them against the same
pi_target used by the run, then report method_metric - baseline_metric.

Negative method-minus-baseline values mean the method improved over the
pi_orig reference for a lower-is-better divergence metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_1D_METRICS = ["js", "kl_sample_target", "w1", "ks"]
DEFAULT_2D_METRICS = ["js", "kl_sample_target", "sliced_w1", "marginal_ks_x", "marginal_ks_y"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dim", choices=["1d", "2d", "auto"], default="auto")
    p.add_argument("--stage", default="final_clean")
    p.add_argument("--metrics", default="auto")
    p.add_argument("--mala-steps", default="2")
    p.add_argument("--grid-points-1d", type=int, default=4001)
    p.add_argument("--grid-points-2d", type=int, default=241)
    p.add_argument("--sliced-samples", type=int, default=20000)
    p.add_argument("--sliced-slices", type=int, default=64)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def as_float(value, default=math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def beta_value(row: dict[str, str]) -> str:
    return row.get("beta_input") or row.get("beta") or ""


def numeric_sorted_unique(values) -> list[str]:
    vals = []
    seen = set()
    for value in values:
        if value is None or value == "":
            continue
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        vals.append(key)
    return sorted(vals, key=lambda x: as_float(x))


def metric_list(rows: list[dict[str, str]], requested: str, dim: str) -> list[str]:
    available = set().union(*(row.keys() for row in rows)) if rows else set()
    if requested.strip().lower() == "auto":
        defaults = DEFAULT_1D_METRICS if dim == "1d" else DEFAULT_2D_METRICS
        return [metric for metric in defaults if metric in available]
    return [m for m in requested.replace(",", " ").split() if m and m in available]


def np_logsumexp(values: np.ndarray, axis=-1) -> np.ndarray:
    m = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(values - m), axis=axis))


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def normalize_1d(log_unnorm: np.ndarray, grid: np.ndarray) -> np.ndarray:
    dens = np.exp(log_unnorm - np.max(log_unnorm))
    z = trapz(dens, grid)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("invalid 1D density normalization")
    return dens / z


def complex_1d_v2_params():
    return {
        "weights": np.asarray([0.07, 0.18, 0.11, 0.24, 0.08, 0.19, 0.13], dtype=np.float64),
        "means": np.asarray([-1.25, -0.82, -0.46, -0.08, 0.22, 0.61, 1.05], dtype=np.float64),
        "stds": np.asarray([0.055, 0.10, 0.045, 0.15, 0.035, 0.09, 0.06], dtype=np.float64),
        "bump_centers": np.asarray([-1.04, -0.63, -0.08, 0.39, 0.61, 0.88], dtype=np.float64),
        "bump_widths": np.asarray([0.055, 0.055, 0.12, 0.06, 0.10, 0.055], dtype=np.float64),
        "bump_weights": np.asarray([3.2, 2.4, -3.0, 3.0, -2.6, 2.8], dtype=np.float64),
        "sin_amp": 0.10,
        "sin_freq": 18.0,
        "sin_phase": 0.20,
        "l2": 0.03,
        "l4": 0.01,
    }


def reward_1d(x: np.ndarray, params: dict) -> np.ndarray:
    bumps = np.sum(
        params["bump_weights"][None, :]
        * np.exp(-0.5 * ((x[..., None] - params["bump_centers"][None, :]) / params["bump_widths"][None, :]) ** 2),
        axis=-1,
    )
    return (
        bumps
        + params["sin_amp"] * np.sin(params["sin_freq"] * x + params["sin_phase"])
        - params["l2"] * x**2
        - params["l4"] * x**4
    )


def orig_density_1d(grid: np.ndarray, params: dict) -> np.ndarray:
    weights = params["weights"] / params["weights"].sum()
    means = params["means"]
    stds = params["stds"]
    log_comp = (
        np.log(weights)[None, :]
        - 0.5 * (np.log(2.0 * np.pi * stds**2)[None, :] + (grid[:, None] - means[None, :]) ** 2 / stds[None, :] ** 2)
    )
    return np.exp(np_logsumexp(log_comp, axis=-1))


def metrics_1d(beta_eff: float, grid_points: int) -> dict[str, float]:
    params = complex_1d_v2_params()
    grid = np.linspace(-6.0, 6.0, grid_points, dtype=np.float64)
    pi_orig = orig_density_1d(grid, params)
    pi_orig = pi_orig / trapz(pi_orig, grid)
    pi_target = normalize_1d(np.log(pi_orig + 1e-300) + beta_eff * reward_1d(grid, params), grid)
    eps = 1e-12
    kl = trapz(pi_orig * (np.log(pi_orig + eps) - np.log(pi_target + eps)), grid)
    mix = 0.5 * (pi_orig + pi_target)
    js = 0.5 * trapz(pi_orig * (np.log(pi_orig + eps) - np.log(mix + eps)), grid)
    js += 0.5 * trapz(pi_target * (np.log(pi_target + eps) - np.log(mix + eps)), grid)
    cdf_orig = np.cumsum(pi_orig)
    cdf_orig = cdf_orig / cdf_orig[-1]
    cdf_target = np.cumsum(pi_target)
    cdf_target = cdf_target / cdf_target[-1]
    return {
        "js": float(js),
        "kl_sample_target": float(kl),
        "w1": trapz(np.abs(cdf_orig - cdf_target), grid),
        "ks": float(np.max(np.abs(cdf_orig - cdf_target))),
    }


def complex_2d_v2_params():
    weights = np.asarray([0.08, 0.16, 0.10, 0.22, 0.13, 0.18, 0.13], dtype=np.float64)
    means = np.asarray(
        [
            [-1.10, -0.72],
            [-0.62, 0.34],
            [-0.18, -0.10],
            [0.08, 0.58],
            [0.46, -0.42],
            [0.83, 0.20],
            [1.12, 0.78],
        ],
        dtype=np.float64,
    )
    stds = np.asarray([[0.10, 0.16], [0.07, 0.11], [0.16, 0.07], [0.09, 0.15], [0.14, 0.08], [0.08, 0.13], [0.12, 0.09]], dtype=np.float64)
    corrs = np.asarray([0.45, -0.55, 0.35, -0.25, 0.50, -0.40, 0.20], dtype=np.float64)
    covs = np.zeros((len(stds), 2, 2), dtype=np.float64)
    covs[:, 0, 0] = stds[:, 0] ** 2
    covs[:, 1, 1] = stds[:, 1] ** 2
    covs[:, 0, 1] = corrs * stds[:, 0] * stds[:, 1]
    covs[:, 1, 0] = covs[:, 0, 1]
    return {
        "weights": weights / weights.sum(),
        "means": means,
        "covs": covs,
        "bump_centers": np.asarray(
            [
                [-1.08, -0.66],
                [-0.62, 0.34],
                [-0.18, -0.10],
                [0.10, 0.58],
                [0.46, -0.42],
                [0.83, 0.20],
                [1.12, 0.78],
                [0.34, 0.02],
            ],
            dtype=np.float64,
        ),
        "bump_widths": np.asarray(
            [[0.13, 0.16], [0.11, 0.14], [0.18, 0.09], [0.12, 0.17], [0.15, 0.10], [0.12, 0.15], [0.16, 0.13], [0.22, 0.18]],
            dtype=np.float64,
        ),
        "bump_weights": np.asarray([2.9, -2.6, -2.0, 1.2, -2.7, 1.9, 3.2, -1.2], dtype=np.float64),
        "sin_amp": 0.10,
        "sin_freqs": np.asarray([11.0, 15.0], dtype=np.float64),
        "sin_phase": 0.45,
        "l2": 0.025,
        "l4": 0.003,
    }


def reward_2d(points: np.ndarray, params: dict) -> np.ndarray:
    diff = (points[:, None, :] - params["bump_centers"][None, :, :]) / params["bump_widths"][None, :, :]
    bumps = np.sum(params["bump_weights"][None, :] * np.exp(-0.5 * np.sum(diff * diff, axis=-1)), axis=-1)
    radius2 = np.sum(points * points, axis=-1)
    return bumps + params["sin_amp"] * np.sin(points @ params["sin_freqs"] + params["sin_phase"]) - params["l2"] * radius2 - params["l4"] * radius2 * radius2


def orig_density_2d(points: np.ndarray, params: dict) -> np.ndarray:
    weights = params["weights"]
    means = params["means"]
    covs = params["covs"]
    inv_cov = np.linalg.inv(covs)
    _sign, logdet = np.linalg.slogdet(covs)
    diff = points[:, None, :] - means[None, :, :]
    quad = np.einsum("nki,kij,nkj->nk", diff, inv_cov, diff)
    log_comp = np.log(weights)[None, :] - 0.5 * (2.0 * np.log(2.0 * np.pi) + logdet[None, :] + quad)
    return np.exp(np_logsumexp(log_comp, axis=-1))


def normalize_2d(log_unnorm: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dens = np.exp(log_unnorm - np.max(log_unnorm))
    z = float(np.sum(dens) * dx * dy)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("invalid 2D density normalization")
    return dens / z


def seed_from_key(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def sample_from_grid(rng: np.random.Generator, grid_x: np.ndarray, grid_y: np.ndarray, density: np.ndarray, n: int) -> np.ndarray:
    dx = float(grid_x[1] - grid_x[0])
    dy = float(grid_y[1] - grid_y[0])
    probs = density.ravel() * dx * dy
    probs = probs / probs.sum()
    idx = rng.choice(len(probs), size=n, replace=True, p=probs)
    iy, ix = np.divmod(idx, len(grid_x))
    jitter_x = rng.uniform(-0.5 * dx, 0.5 * dx, size=n)
    jitter_y = rng.uniform(-0.5 * dy, 0.5 * dy, size=n)
    return np.column_stack([grid_x[ix] + jitter_x, grid_y[iy] + jitter_y])


def sliced_w1(samples_a: np.ndarray, samples_b: np.ndarray, n_slices: int, rng: np.random.Generator) -> float:
    theta = rng.normal(size=(n_slices, 2))
    theta = theta / np.linalg.norm(theta, axis=1, keepdims=True)
    n = min(len(samples_a), len(samples_b))
    vals = []
    for direction in theta:
        vals.append(float(np.mean(np.abs(np.sort(samples_a[:n] @ direction) - np.sort(samples_b[:n] @ direction)))))
    return float(np.mean(vals))


def marginal_ks_from_densities(dens_a: np.ndarray, dens_b: np.ndarray, dx: float, dy: float, axis: int) -> float:
    if axis == 0:
        marg_a = np.sum(dens_a, axis=0) * dy
        marg_b = np.sum(dens_b, axis=0) * dy
        step = dx
    else:
        marg_a = np.sum(dens_a, axis=1) * dx
        marg_b = np.sum(dens_b, axis=1) * dx
        step = dy
    cdf_a = np.cumsum(marg_a) * step
    cdf_b = np.cumsum(marg_b) * step
    cdf_a = cdf_a / cdf_a[-1]
    cdf_b = cdf_b / cdf_b[-1]
    return float(np.max(np.abs(cdf_a - cdf_b)))


def metrics_2d(beta_eff: float, grid_points: int, sliced_samples: int, sliced_slices: int, seed: int) -> dict[str, float]:
    params = complex_2d_v2_params()
    grid_x = np.linspace(-3.0, 3.0, grid_points, dtype=np.float64)
    grid_y = np.linspace(-3.0, 3.0, grid_points, dtype=np.float64)
    dx = float(grid_x[1] - grid_x[0])
    dy = float(grid_y[1] - grid_y[0])
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    pi_orig = orig_density_2d(points, params).reshape(len(grid_y), len(grid_x))
    pi_orig = pi_orig / (np.sum(pi_orig) * dx * dy)
    pi_target = normalize_2d(np.log(pi_orig.ravel() + 1e-300) + beta_eff * reward_2d(points, params), dx, dy).reshape(len(grid_y), len(grid_x))
    p = pi_orig * dx * dy
    q = pi_target * dx * dy
    p = p / p.sum()
    q = q / q.sum()
    eps = 1e-12
    kl = float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))
    mix = 0.5 * (p + q)
    js = float(0.5 * np.sum(p * (np.log(p + eps) - np.log(mix + eps))) + 0.5 * np.sum(q * (np.log(q + eps) - np.log(mix + eps))))
    rng = np.random.default_rng(seed)
    samples_orig = sample_from_grid(rng, grid_x, grid_y, pi_orig, sliced_samples)
    samples_target = sample_from_grid(rng, grid_x, grid_y, pi_target, sliced_samples)
    return {
        "js": js,
        "kl_sample_target": kl,
        "sliced_w1": sliced_w1(samples_orig, samples_target, sliced_slices, rng),
        "marginal_ks_x": marginal_ks_from_densities(pi_orig, pi_target, dx, dy, axis=0),
        "marginal_ks_y": marginal_ks_from_densities(pi_orig, pi_target, dx, dy, axis=1),
    }


def baseline_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (str(row.get("sample_budget", "")), str(beta_value(row)), str(row.get("dataset_seed", "")), str(row.get("reward_std_raw", "")))


def effective_beta_from_row(row: dict[str, str]) -> float:
    beta = as_float(beta_value(row))
    reward_std = as_float(row.get("reward_std_raw"), 1.0)
    if np.isfinite(reward_std) and reward_std > 0:
        return beta / reward_std
    return beta


def compute_baselines(rows: list[dict[str, str]], dim: str, args: argparse.Namespace) -> dict[tuple[str, str, str, str], dict]:
    out = {}
    for key in sorted({baseline_key(row) for row in rows}, key=lambda k: (as_float(k[0]), as_float(k[1]), as_float(k[2]))):
        sample_budget, beta, dataset_seed, reward_std = key
        template = next(row for row in rows if baseline_key(row) == key)
        beta_eff = effective_beta_from_row(template)
        if dim == "1d":
            metrics = metrics_1d(beta_eff, args.grid_points_1d)
        else:
            seed = seed_from_key(dim, sample_budget, beta, dataset_seed, reward_std)
            metrics = metrics_2d(beta_eff, args.grid_points_2d, args.sliced_samples, args.sliced_slices, seed)
        out[key] = {
            "dim": dim,
            "baseline_distribution": "pi_orig",
            "sample_budget": sample_budget,
            "beta": beta,
            "dataset_seed": dataset_seed,
            "reward_std_raw": reward_std,
            "beta_eff": beta_eff,
            **metrics,
        }
    return out


def method_rows(rows: list[dict[str, str]], metrics: list[str], baselines: dict[tuple[str, str, str, str], dict], mala_steps: str) -> list[dict]:
    out = []
    for row in rows:
        method = row.get("method", "")
        if method == "orig_energy_mala" and str(row.get("mala_steps", "")) != str(mala_steps):
            continue
        if method not in {"exponential_energy", "orig_energy_mala"}:
            continue
        baseline = baselines[baseline_key(row)]
        method_steps = "independent" if method == "exponential_energy" else str(row.get("mala_steps", ""))
        for metric in metrics:
            method_value = as_float(row.get(metric))
            baseline_value = as_float(baseline.get(metric))
            if not (np.isfinite(method_value) and np.isfinite(baseline_value)):
                continue
            delta = method_value - baseline_value
            out.append(
                {
                    "metric": metric,
                    "dim": row.get("dim", ""),
                    "method": method,
                    "mala_steps": method_steps,
                    "sample_budget": row.get("sample_budget", ""),
                    "beta": beta_value(row),
                    "dataset_seed": row.get("dataset_seed", ""),
                    "method_value": method_value,
                    "pi_orig_baseline": baseline_value,
                    "method_minus_baseline": delta,
                    "interpretation": "method_better_than_pi_orig_reference" if delta < 0 else "method_worse_than_pi_orig_reference",
                }
            )
    return out


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["metric"], row["method"], row["mala_steps"], row["beta"], row["sample_budget"])
        groups.setdefault(key, []).append(row)
    out = []
    for (metric, method, steps, beta, budget), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], as_float(kv[0][3]), as_float(kv[0][4]), kv[0][2])):
        method_vals = np.asarray([float(row["method_value"]) for row in group], dtype=np.float64)
        base_vals = np.asarray([float(row["pi_orig_baseline"]) for row in group], dtype=np.float64)
        deltas = np.asarray([float(row["method_minus_baseline"]) for row in group], dtype=np.float64)
        out.append(
            {
                "metric": metric,
                "method": method,
                "mala_steps": steps,
                "beta": beta,
                "sample_budget": budget,
                "n": len(group),
                "method_mean": float(np.mean(method_vals)),
                "method_median": float(np.median(method_vals)),
                "pi_orig_baseline_mean": float(np.mean(base_vals)),
                "pi_orig_baseline_median": float(np.median(base_vals)),
                "method_minus_baseline_mean": float(np.mean(deltas)),
                "method_minus_baseline_median": float(np.median(deltas)),
                "method_better_fraction": float(np.mean(deltas < 0)),
            }
        )
    return out


def finite_symmetric_limit(mats: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([mat[np.isfinite(mat)] for mat in mats if np.isfinite(mat).any()]) if any(np.isfinite(mat).any() for mat in mats) else np.array([1.0])
    lim = float(np.max(np.abs(vals)))
    return (-lim if lim > 0 else -1.0, lim if lim > 0 else 1.0)


def add_cell_text(ax, mat: np.ndarray) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=7, color="black")


def matrix_from_cells(rows: list[dict], metric: str, method: str, steps: str, betas: list[str], budgets: list[str], value_key: str) -> np.ndarray:
    mat = np.full((len(betas), len(budgets)), np.nan, dtype=np.float64)
    for i, beta in enumerate(betas):
        for j, budget in enumerate(budgets):
            vals = [
                as_float(row.get(value_key))
                for row in rows
                if row.get("metric") == metric
                and row.get("method") == method
                and row.get("mala_steps") == steps
                and str(row.get("beta")) == str(beta)
                and str(row.get("sample_budget")) == str(budget)
            ]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                mat[i, j] = float(np.mean(vals))
    return mat


def plot_delta_heatmaps(cell_rows: list[dict], metrics: list[str], betas: list[str], budgets: list[str], out_dir: Path, dpi: int) -> int:
    methods = [("exponential_energy", "independent"), ("orig_energy_mala", "2")]
    count = 0
    for metric in metrics:
        mats = [matrix_from_cells(cell_rows, metric, method, steps, betas, budgets, "method_minus_baseline_mean") for method, steps in methods]
        if not any(np.isfinite(mat).any() for mat in mats):
            continue
        vmin, vmax = finite_symmetric_limit(mats)
        fig, axes = plt.subplots(1, len(methods), figsize=(4.0 * len(methods), 3.5), squeeze=False)
        for ax, (method, steps), mat in zip(axes.ravel(), methods, mats):
            im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
            add_cell_text(ax, mat)
            title = "exponential_energy" if method == "exponential_energy" else f"orig_energy_mala\nmala_steps={steps}"
            ax.set_title(title, fontsize=9)
            ax.set_xticks(range(len(budgets)), budgets, rotation=30, ha="right")
            ax.set_yticks(range(len(betas)), betas)
            ax.set_xlabel("sample_budget")
            ax.set_ylabel("beta")
            fig.colorbar(im, ax=ax, shrink=0.82)
        fig.suptitle(f"{metric}: method - pi_orig reference; negative is better", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path = out_dir / "heatmaps" / "method_minus_pi_orig_baseline" / f"{metric}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    rows = [row for row in read_rows(args.metrics_file) if row.get("stage") == args.stage]
    if args.dim == "auto":
        dims = sorted({str(row.get("dim", "")).lower() for row in rows})
        if len(dims) != 1 or dims[0] not in {"1d", "2d"}:
            raise SystemExit(f"could not infer dim from CSV; found {dims}")
        dim = dims[0]
    else:
        dim = args.dim
        rows = [row for row in rows if str(row.get("dim", "")).lower() in {"", dim}]
    if not rows:
        raise SystemExit("no matching rows")
    metrics = metric_list(rows, args.metrics, dim)
    if not metrics:
        raise SystemExit("no requested metrics found in CSV")

    baselines = compute_baselines(rows, dim, args)
    baseline_rows = list(baselines.values())
    delta_rows = method_rows(rows, metrics, baselines, args.mala_steps)
    cell_rows = aggregate_rows(delta_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "baseline_metrics.csv", baseline_rows)
    write_rows(args.out_dir / "method_minus_baseline_rows.csv", delta_rows)
    write_rows(args.out_dir / "method_minus_baseline_cells.csv", cell_rows)

    plot_count = 0
    if not args.no_plots:
        betas = numeric_sorted_unique(row.get("beta") for row in cell_rows)
        budgets = numeric_sorted_unique(row.get("sample_budget") for row in cell_rows)
        plot_count = plot_delta_heatmaps(cell_rows, metrics, betas, budgets, args.out_dir, args.dpi)

    print(f"wrote baseline reference for {len(baseline_rows)} unique beta/budget/seed cells")
    print(f"wrote {len(cell_rows)} method-minus-baseline aggregate rows")
    if not args.no_plots:
        print(f"wrote {plot_count} baseline-delta heatmap image(s) to {args.out_dir / 'heatmaps'}")


if __name__ == "__main__":
    main()
