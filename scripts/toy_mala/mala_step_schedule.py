from __future__ import annotations

import math
from typing import Iterable

import numpy as np


MALA_STEP_SCHEDULE_CHOICES = (
    "constant",
    "continuous_uniform_sqrt_alpha_bar",
    "continuous_uniform_sqrt_alpha_bar_min1",
    "linear_low_noise",
    "quadratic_low_noise",
)


def _as_positive_budget(mala_budget: int | None, diffusion_steps: int, mala_steps: int | None) -> int:
    if diffusion_steps <= 0:
        raise ValueError("diffusion_steps must be positive.")
    if mala_budget is None:
        if mala_steps is None:
            raise ValueError("Either mala_budget or mala_steps must be provided.")
        mala_budget = int(mala_steps) * int(diffusion_steps)
    mala_budget = int(mala_budget)
    if mala_budget < 0:
        raise ValueError("mala_budget must be non-negative.")
    return mala_budget


def _largest_remainder_allocate(weights: Iterable[float], budget: int, min_steps: int) -> tuple[int, ...]:
    weights_arr = np.asarray(list(weights), dtype=np.float64)
    if weights_arr.ndim != 1 or weights_arr.size == 0:
        raise ValueError("weights must be a non-empty 1D sequence.")
    if np.any(weights_arr < 0.0) or not np.all(np.isfinite(weights_arr)):
        raise ValueError("weights must be finite and non-negative.")
    if float(weights_arr.sum()) <= 0.0:
        raise ValueError("At least one weight must be positive.")

    n = int(weights_arr.size)
    if budget < min_steps * n:
        raise ValueError(
            f"mala_budget={budget} is too small for min_steps={min_steps} over {n} diffusion levels."
        )

    out = np.full(n, int(min_steps), dtype=np.int64)
    remaining = int(budget - out.sum())
    if remaining == 0:
        return tuple(int(x) for x in out)

    quotas = remaining * weights_arr / float(weights_arr.sum())
    floors = np.floor(quotas).astype(np.int64)
    out += floors
    leftover = int(budget - out.sum())
    if leftover:
        remainders = quotas - floors
        # Larger remainder wins.  Ties go to lower-noise levels, which have
        # larger weights under the low-noise schedules.
        order = sorted(range(n), key=lambda i: (remainders[i], weights_arr[i]), reverse=True)
        for i in order[:leftover]:
            out[i] += 1
    return tuple(int(x) for x in out)


def _continuous_uniform_sqrt_alpha_bar(
    sqrt_alphas_cumprod: Iterable[float],
    budget: int,
    min_steps: int,
) -> tuple[int, ...]:
    c = np.asarray(list(sqrt_alphas_cumprod), dtype=np.float64)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("sqrt_alphas_cumprod must be a non-empty 1D sequence.")
    if not np.all(np.isfinite(c)):
        raise ValueError("sqrt_alphas_cumprod must be finite.")

    n = int(c.size)
    if budget < min_steps * n:
        raise ValueError(
            f"mala_budget={budget} is too small for min_steps={min_steps} over {n} diffusion levels."
        )

    out = np.full(n, int(min_steps), dtype=np.int64)
    remaining = int(budget - out.sum())
    if remaining == 0:
        return tuple(int(x) for x in out)

    c_min = float(c.min())
    c_max = float(c.max())
    if math.isclose(c_min, c_max):
        return _largest_remainder_allocate(np.ones(n, dtype=np.float64), budget, min_steps)

    points = c_min + ((np.arange(remaining, dtype=np.float64) + 0.5) / remaining) * (c_max - c_min)
    for point in points:
        # Tie-break: if a point is exactly between two levels, choose the
        # lower-noise level, i.e. the one with larger sqrt(alpha_bar).
        best = min(range(n), key=lambda i: (abs(float(point - c[i])), -float(c[i])))
        out[best] += 1
    return tuple(int(x) for x in out)


def allocate_mala_steps(
    schedule_name: str,
    diffusion_steps: int,
    mala_budget: int | None = None,
    sqrt_alphas_cumprod: Iterable[float] | None = None,
    mala_steps: int | None = None,
) -> tuple[int, ...]:
    budget = _as_positive_budget(mala_budget, diffusion_steps, mala_steps)
    n = int(diffusion_steps)

    if schedule_name == "constant":
        return _largest_remainder_allocate(np.ones(n, dtype=np.float64), budget, min_steps=0)

    if schedule_name == "continuous_uniform_sqrt_alpha_bar":
        if sqrt_alphas_cumprod is None:
            raise ValueError("continuous_uniform_sqrt_alpha_bar requires sqrt_alphas_cumprod.")
        return _continuous_uniform_sqrt_alpha_bar(sqrt_alphas_cumprod, budget, min_steps=0)

    if schedule_name == "continuous_uniform_sqrt_alpha_bar_min1":
        if sqrt_alphas_cumprod is None:
            raise ValueError("continuous_uniform_sqrt_alpha_bar_min1 requires sqrt_alphas_cumprod.")
        return _continuous_uniform_sqrt_alpha_bar(sqrt_alphas_cumprod, budget, min_steps=1)

    if schedule_name == "linear_low_noise":
        weights = np.arange(n, 0, -1, dtype=np.float64)
        return _largest_remainder_allocate(weights, budget, min_steps=1)

    if schedule_name == "quadratic_low_noise":
        weights = np.arange(n, 0, -1, dtype=np.float64) ** 2
        return _largest_remainder_allocate(weights, budget, min_steps=1)

    raise ValueError(f"Unknown mala_step_schedule: {schedule_name}")
