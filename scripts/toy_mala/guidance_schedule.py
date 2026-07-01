"""Small helpers for per-diffusion-level guidance scaling."""

from __future__ import annotations


GUIDANCE_SCHEDULE_CHOICES = ("constant", "alpha_bar")


def guidance_schedule_multiplier(schedule_name: str, alphas_cumprod, t_idx):
    """Return the guidance multiplier for one diffusion level.

    This intentionally avoids importing JAX.  When ``alphas_cumprod`` is a JAX
    array, indexing and arithmetic still produce a traced scalar.
    """
    if schedule_name == "constant":
        return alphas_cumprod[t_idx] * 0.0 + 1.0
    if schedule_name == "alpha_bar":
        return alphas_cumprod[t_idx]
    raise ValueError(f"Unknown guidance_schedule: {schedule_name}")
