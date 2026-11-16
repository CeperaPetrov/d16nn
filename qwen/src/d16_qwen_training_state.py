"""Checkpoint persistence for resumable D16 Qwen training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import tempfile

import torch
from torch import nn


@dataclass
class TrainingState:
    """Progress at an optimizer-step boundary.

    ``epoch`` starts at one, ``batch`` counts completed batches in this epoch,
    and ``step`` counts completed updates across all epochs. ``totals`` stores
    the token-weighted metric sums for the current epoch.
    """

    epoch: int = 1
    batch: int = 0
    step: int = 0
    best_epoch: int = 0
    best_validation: dict[str, float | int] | None = None
    totals: dict[str, float] = field(default_factory=dict)
    tokens: int = 0
    baseline: dict[str, float | int] | None = None


def _atomic_save(path: Path, payload: dict) -> None:
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        pending = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            pending.replace(path)
        except BaseException:
            stream.close()
            pending.unlink(missing_ok=True)
            raise


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    metadata: dict,
) -> None:
    """Persist model, optimizer, cursor and RNG after a complete update."""
    device = next(model.parameters()).device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    _atomic_save(path, {
        "format_version": 1,
        "metadata": metadata,
        "training_state": asdict(state),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": cuda_rng,
    })


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_metadata: dict,
) -> TrainingState:
    """Restore the run after validating its requested configuration and data."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported training checkpoint format in {path}")
    metadata = payload["metadata"]
    mismatched = [key for key, value in expected_metadata.items() if key not in metadata or metadata[key] != value]
    if mismatched:
        raise ValueError(f"Training checkpoint metadata differs for {', '.join(sorted(mismatched))}")
    state = TrainingState(**payload["training_state"])
    if state.epoch < 1 or min(state.batch, state.step, state.tokens, state.best_epoch) < 0:
        raise ValueError(f"Invalid training cursor in {path}")

    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["cpu_rng"])
    device = next(model.parameters()).device
    if device.type == "cuda" and payload["cuda_rng"] is not None:
        torch.cuda.set_rng_state(payload["cuda_rng"], device)
    return state


def save_inference_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    config: dict,
    epoch: int,
    step: int,
    validation: dict[str, float | int],
    teacher_revision: str | None = None,
) -> None:
    """Save weights and metrics in the same format as the pilot checkpoint."""
    _atomic_save(path, {
        "config": config,
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "validation": validation,
        "teacher_revision": teacher_revision,
    })
