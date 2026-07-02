from __future__ import annotations


DENOISING_SCHEDULE_SWEEP_CHOICES = (
    "all_DDPM_mean",
    "all_DDIM",
    "all_Identity",
    "DDPM_then_last1_DDIM",
    "DDPM_then_last2_DDIM",
    "Identity_then_last1_DDIM",
    "Identity_then_last2_DDIM",
)

DENOISING_SCHEDULE_CHOICES = ("from_predictor",) + DENOISING_SCHEDULE_SWEEP_CHOICES


def denoising_predictor_for_step(schedule_name: str, legacy_predictor: str, t_idx: int) -> str:
    if schedule_name == "from_predictor":
        return legacy_predictor
    if schedule_name == "all_DDPM_mean":
        return "DDPM_mean"
    if schedule_name == "all_DDIM":
        return "DDIM"
    if schedule_name == "all_Identity":
        return "Identity"
    if schedule_name == "DDPM_then_last1_DDIM":
        return "DDIM" if int(t_idx) < 1 else "DDPM_mean"
    if schedule_name == "DDPM_then_last2_DDIM":
        return "DDIM" if int(t_idx) < 2 else "DDPM_mean"
    if schedule_name == "Identity_then_last1_DDIM":
        return "DDIM" if int(t_idx) < 1 else "Identity"
    if schedule_name == "Identity_then_last2_DDIM":
        return "DDIM" if int(t_idx) < 2 else "Identity"
    raise ValueError(f"Unknown denoising_schedule: {schedule_name}")
