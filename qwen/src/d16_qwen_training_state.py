"""Checkpoint persistence for resumable D16 Qwen training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

import torch
from torch import nn


class CheckpointOptimizer(Protocol):
    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


@dataclass
class TrainingState:
    """Progress at an optimizer-step boundary.

    ``epoch`` starts at one, ``batch`` counts completed optimizer steps in this epoch,
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


def _detach_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().contiguous()
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(v) for v in obj)
    return obj


def _atomic_save(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _build_d16_config_dict(model: nn.Module, config: dict) -> dict:
    """Build a `load_d16_checkpoint`-compatible `d16_config` dictionary."""
    d16_cfg = dict(config)
    model_cfg = getattr(model, "config", None)
    if model_cfg is not None and hasattr(model_cfg, "complex_hidden_size") and hasattr(model_cfg, "hidden_size"):
        d16_cfg["is_1to1_import"] = bool(model_cfg.complex_hidden_size == model_cfg.hidden_size)
    else:
        d16_cfg.setdefault("is_1to1_import", False)
    first_param = next(model.parameters(), None)
    if first_param is not None:
        d16_cfg["dtype"] = str(first_param.dtype).replace("torch.", "")
    return d16_cfg


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: CheckpointOptimizer,
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
        "state_dict": _detach_to_cpu(model.state_dict()),
        "optimizer": _detach_to_cpu(optimizer.state_dict()),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": cuda_rng,
    })


def read_training_metadata(path: Path) -> dict:
    """Identify the configuration and optimizer before constructing the model.

    Avoids `mmap=True` so Windows does not retain an open file handle when
    `_atomic_save` later replaces `latest.pt`.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported training checkpoint format in {path}")
    return payload["metadata"]


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: CheckpointOptimizer,
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
    weight_storage: str = "d16",
) -> None:
    """Save weights, config (also compatible with `load_d16_checkpoint`), and validation metrics."""
    _atomic_save(path, {
        "config": config,
        "d16_config": _build_d16_config_dict(model, config),
        "state_dict": _detach_to_cpu(model.state_dict()),
        "epoch": epoch,
        "step": step,
        "validation": validation,
        "teacher_revision": teacher_revision,
        "weight_storage": weight_storage,
    })
