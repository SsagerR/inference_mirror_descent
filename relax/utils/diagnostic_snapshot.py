"""Utilities for host-side diagnostic snapshots.

These snapshots are intentionally separate from long-term training
checkpoints. They capture one training state's policy/Q parameters plus fixed
observations and replay transitions so sampler-distribution diagnostics can be
rerun without perturbing the original training loop.
"""
from pathlib import Path
import json
import pickle
from typing import Any, Sequence, Tuple

import numpy as np

try:
    import jax
except ModuleNotFoundError:  # pragma: no cover - local docs/tests may not have JAX.
    jax = None


def sample_replay_batches_for_snapshot(
    buffers: Sequence[Any],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[list, np.ndarray]:
    """Gather fixed replay minibatches without advancing each buffer's RNG."""
    batches = []
    indices_per_run = []
    for buffer in buffers:
        if buffer.len <= 0:
            raise ValueError("cannot save a diagnostic replay batch from an empty buffer")
        indices = rng.integers(0, buffer.len, size=batch_size)
        batches.append(buffer.gather_indices(indices))
        indices_per_run.append(indices)
    return batches, np.stack(indices_per_run, axis=0)


def save_diagnostic_snapshot(
    *,
    root_dir: Path,
    env_name: str,
    step: int,
    algorithm_state: Any,
    rollout_obs: np.ndarray,
    replay_batches: Sequence[Any],
    replay_indices: np.ndarray,
    metadata: dict,
    hparams: dict,
) -> Path:
    """Write a diagnostic snapshot and return the created directory."""
    root_dir = Path(root_dir)
    snapshot_dir = root_dir / f"{env_name}_step_{int(step)}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    metadata_out = dict(metadata)
    metadata_out.update({"env_name": env_name, "step": int(step)})

    _write_json(snapshot_dir / "metadata.json", metadata_out)
    _write_json(snapshot_dir / "hparams.json", hparams)
    np.save(snapshot_dir / "rollout_obs.npy", np.asarray(rollout_obs))
    np.save(snapshot_dir / "replay_indices.npy", np.asarray(replay_indices))

    with (snapshot_dir / "algorithm_state.pkl").open("wb") as f:
        pickle.dump(_device_get(algorithm_state), f, protocol=pickle.HIGHEST_PROTOCOL)
    with (snapshot_dir / "replay_batches.pkl").open("wb") as f:
        pickle.dump(_device_get(list(replay_batches)), f, protocol=pickle.HIGHEST_PROTOCOL)

    return snapshot_dir


def _device_get(x: Any) -> Any:
    if jax is None:
        return x
    return jax.device_get(x)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
