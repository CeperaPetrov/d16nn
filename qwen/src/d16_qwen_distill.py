"""Distill a frozen Qwen2.5 checkpoint (local or HuggingFace Hub) into a native D16 model trained from scratch."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Optional

import torch
from torch.nn import functional as F

from d16_adamw import D16AdamW
from d16_complex import D16ComplexTensor
from d16_qwen_model2 import (
    D16Linear,
    D16QwenConfig,
    D16QwenForCausalLM,
    IndiffThreshold,
    encode_token_ids,
)
from d16_qwen_training_state import (
    TrainingState,
    load_training_checkpoint,
    read_training_metadata,
    save_inference_checkpoint,
    save_training_checkpoint,
)

DEFAULT_HF_TEACHER_ID = "Qwen/Qwen2.5-0.5B"


def _parse_indiff_threshold(raw: str) -> IndiffThreshold:
    text = str(raw).strip().lower()
    if text == "auto":
        return "auto"
    val = float(text)
    if not math.isfinite(val) or val < 0.0:
        raise argparse.ArgumentTypeError("indiff-eps must be 'auto' or a nonnegative finite float")
    return val


def _safe_perplexity(nll: float) -> float:
    if not math.isfinite(nll):
        return float("inf")
    return math.exp(min(nll, 50.0))


def _resolve_dtype(name: str, device: torch.device, *, prefer_bf16_on_cuda: bool = True) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "auto":
        if device.type == "cuda":
            if prefer_bf16_on_cuda and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float32
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name!r}")


def _is_valid_local_qwen_dir(candidate: Path) -> bool:
    if not candidate.is_dir():
        return False
    has_config = (candidate / "config.json").is_file()
    has_weights = any(candidate.glob("*.safetensors")) or any(candidate.glob("*.bin"))
    return has_config and has_weights


def resolve_teacher_source(
    explicit_source: Optional[str],
    *,
    force_local_only: bool = False,
) -> tuple[str, bool]:
    """Resolve Qwen teacher from an explicit path/HF ID, local search, or HF Hub fallback.

    Returns ``(source_str, local_files_only)``.
    """
    if explicit_source is not None:
        raw = explicit_source.strip()
        path_cand = Path(raw).expanduser()
        if path_cand.is_file():
            return str(path_cand.parent.resolve()), True
        if path_cand.is_dir():
            return str(path_cand.resolve()), True
        # Treat as HuggingFace Hub repo ID (keep POSIX forward slash intact on Windows!)
        return raw, force_local_only

    # Auto-detect local Qwen weights using the same search strategy as the notebook
    roots = list(dict.fromkeys((Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent)))
    candidates: list[Path] = []
    for root in roots:
        candidates.extend((
            root / "model.safetensors",
            root / ".d16-distill-cache" / "teacher",
            root / "qwen",
            root / "Qwen2.5-0.5B",
            root / "Qwen2.5-0.5B-Instruct",
        ))

    for cand in candidates:
        if cand.is_file() and cand.name == "model.safetensors" and (cand.parent / "config.json").is_file():
            return str(cand.parent.resolve()), True
        if _is_valid_local_qwen_dir(cand):
            return str(cand.resolve()), True

    return DEFAULT_HF_TEACHER_ID, force_local_only


def _load_teacher_model(
    teacher_source: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    local_files_only: bool = False,
    revision: Optional[str] = None,
):
    from transformers import AutoModelForCausalLM

    kwargs: dict = {"local_files_only": local_files_only}
    if revision is not None:
        kwargs["revision"] = revision
    try:
        teacher = AutoModelForCausalLM.from_pretrained(teacher_source, dtype=dtype, **kwargs)
    except TypeError:
        teacher = AutoModelForCausalLM.from_pretrained(teacher_source, torch_dtype=dtype, **kwargs)
    return teacher.to(device).eval().requires_grad_(False)


def _load_student_init_weights(checkpoint_path: Path, student: D16QwenForCausalLM) -> None:
    """Load only model weights from a saved inference or training checkpoint (fresh optimizer/schedule)."""
    payload = torch.load(checkpoint_path.expanduser(), map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model", "student"):
            if key in payload and isinstance(payload[key], dict):
                state_dict = payload[key]
                break
        else:
            state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
    student.load_state_dict(state_dict, strict=True)


def _scheduled_lr(
    step: int,
    total_steps: int,
    max_lr: float,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """Linear warmup followed by cosine decay down to `max_lr * min_lr_ratio`."""
    if total_steps <= 1:
        return max_lr
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / float(decay_steps), 0.0), 1.0)
    min_lr = max_lr * min_lr_ratio
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def distillation_kl(
    student_logits: D16ComplexTensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    *,
    top_k: int = 0,
) -> D16ComplexTensor:
    """Mean token-level KL(teacher || student) aligned with D16 rotation-free wedge_strength.

    Uses the exact same rotation-free wedge strength (`sqrt(x^2 + y^2)`) as
    `wedge_cross_entropy`, `sampling_probabilities`, and `generate()` (`wedge_argmax`),
    keeping training and inference strictly consistent.
    """
    if student_logits.shape != teacher_logits.shape or teacher_logits.ndim < 2 or not teacher_logits.numel():
        raise ValueError("Distillation requires nonempty, matching [..., vocabulary] logits")
    if not 0.0 < temperature < float("inf"):
        raise ValueError("Distillation temperature must be positive and finite")

    calc_dtype = (
        torch.float64
        if torch.float64 in (student_logits.dtype, teacher_logits.dtype)
        else torch.float32
    )
    student_scores = student_logits.wedge_strength(eps=0.0, rel=0.0).to(calc_dtype) / temperature
    student_log_probs = F.log_softmax(student_scores, dim=-1)

    vocab = teacher_logits.shape[-1]
    if top_k > 0 and top_k < vocab:
        with torch.no_grad():
            t_vals, t_idx = (teacher_logits.to(calc_dtype) / temperature).topk(top_k, dim=-1)
            t_log_probs = F.log_softmax(t_vals, dim=-1)
            t_probs = t_log_probs.exp()
            t_log_probs = t_log_probs.masked_fill(t_probs == 0, 0.0)
            t_idx = t_idx.to(device=student_log_probs.device)
            t_probs = t_probs.to(device=student_log_probs.device)
            t_log_probs = t_log_probs.to(device=student_log_probs.device)
        s_log_probs_k = student_log_probs.gather(-1, t_idx)
        kl = (t_probs * (t_log_probs - s_log_probs_k)).sum(dim=-1).mean().clamp_min(0.0)
    else:
        with torch.no_grad():
            t_log_probs = F.log_softmax(teacher_logits.to(calc_dtype) / temperature, dim=-1)
            t_probs = t_log_probs.exp()
            t_log_probs = t_log_probs.masked_fill(t_probs == 0, 0.0)
            t_probs = t_probs.to(device=student_log_probs.device)
            t_log_probs = t_log_probs.to(device=student_log_probs.device)
        kl = (t_probs * (t_log_probs - student_log_probs)).sum(dim=-1).mean().clamp_min(0.0)

    undefined = student_logits.is_nan().any() | torch.isnan(kl)
    return D16ComplexTensor.from_real(kl.masked_fill(undefined, float("nan")))


def _distillation_kl_from_topk(
    student_logits: D16ComplexTensor,
    top_indices: torch.Tensor,
    top_probs: torch.Tensor,
    top_log_probs: torch.Tensor,
    temperature: float,
) -> D16ComplexTensor:
    """Memory-efficient KL using pre-extracted teacher Top-K probabilities."""
    calc_dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    student_scores = student_logits.wedge_strength(eps=0.0, rel=0.0).to(calc_dtype) / temperature
    student_log_probs = F.log_softmax(student_scores, dim=-1)
    s_log_probs_k = student_log_probs.gather(-1, top_indices)
    kl = (top_probs.to(calc_dtype) * (top_log_probs.to(calc_dtype) - s_log_probs_k)).sum(dim=-1).mean().clamp_min(0.0)
    undefined = student_logits.is_nan().any() | torch.isnan(kl)
    return D16ComplexTensor.from_real(kl.masked_fill(undefined, float("nan")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-dir",
        type=str,
        default=None,
        help=(
            "Local Qwen directory/file OR HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-0.5B'). "
            "If omitted, searches local folders first and falls back to 'Qwen/Qwen2.5-0.5B'."
        ),
    )
    parser.add_argument("--teacher-revision", type=str, default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disallow downloading teacher weights/tokenizer from HuggingFace Hub",
    )
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--validation-text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    # Student architecture (default: 12 layers, native D16 bilinear MLP)
    parser.add_argument("--num-layers", type=int, default=12, help="Number of decoder layers in D16 student (default: 12)")
    parser.add_argument("--hidden-size", type=int, help="Real hidden width (2x complex channels); defaults to teacher hidden_size")
    parser.add_argument("--intermediate-size", type=int, help="MLP width; defaults to teacher intermediate_size")
    parser.add_argument(
        "--mlp-activation",
        choices=("bilinear", "gate", "sigmoid", "sum", "rel"),
        default="bilinear",
        help="D16QwenMLP activation mode (default: bilinear for training from scratch)",
    )
    parser.add_argument("--indiff-eps", type=_parse_indiff_threshold, default=0.0, help="Model activation 0xFFFF cutoff ('auto' or float)")
    parser.add_argument("--indiff-rel", type=float, default=0.25, help="Relative threshold when indiff-eps='auto'")
    parser.add_argument("--optimizer-indiff-eps", type=_parse_indiff_threshold, default=0.0, help="D16AdamW weight 0xFFFF cutoff ('auto' or float)")
    parser.add_argument("--optimizer-indiff-rel", type=float, default=0.25, help="Relative threshold for D16AdamW when optimizer-indiff-eps='auto'")

    # Training schedule & distillation hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-blocks", type=int, help="Limit the training split to a seeded sample")
    parser.add_argument("--validation-blocks", type=int, default=24, help="Fixed held-out sample; zero uses the full split")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size per forward pass")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Micro-batches accumulated per D16AdamW step")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50, help="Linear LR warmup optimizer steps")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="Final cosine decay LR fraction")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--distill-top-k", type=int, default=128, help="Teacher Top-K tokens for KL (0 = full vocabulary)")
    parser.add_argument("--seed", type=int, default=42)

    # Devices, precision & performance
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Student device")
    parser.add_argument("--teacher-device", default="cpu", help="Teacher device (default: cpu to save GPU VRAM)")
    parser.add_argument("--teacher-dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--student-dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=0, help="PyTorch CPU threads (0 = auto; uses up to 12 cores for CPU teacher)")
    parser.add_argument("--weight-tile-elements", type=int, default=1 << 24, help="D16Linear._MAX_WEIGHT_TILE (0 = keep class default)")
    parser.add_argument("--tf32", action="store_true", help="Permit TF32 for FP32 matrix products on CUDA")

    # Checkpointing & control
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save resumable state every N updates")
    parser.add_argument("--evaluate-every", type=int, default=250, help="Validate every N updates and at each epoch end")
    parser.add_argument("--init-from", type=Path, default=None, help="Initialize student weights from checkpoint (e.g. best.pt) with fresh optimizer/LR")
    parser.add_argument("--resume", action="store_true", help="Restore an existing run from latest.pt")
    parser.add_argument("--stop-after-step", type=int, help="Save and pause after this absolute optimizer step")

    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accum_steps, args.checkpoint_every, args.evaluate_every) <= 0:
        parser.error("epochs, batch size, grad-accum-steps and intervals must be positive")
    if args.num_layers is not None and args.num_layers <= 0:
        parser.error("num-layers must be positive")
    if args.hidden_size is not None and args.hidden_size <= 0:
        parser.error("hidden-size must be positive")
    if args.intermediate_size is not None and args.intermediate_size <= 0:
        parser.error("intermediate-size must be positive")
    if args.validation_blocks < 0 or (args.train_blocks is not None and args.train_blocks <= 0):
        parser.error("invalid block count")
    if args.stop_after_step is not None and args.stop_after_step <= 0:
        parser.error("stop-after-step must be positive")
    if args.sequence_length < 2:
        parser.error("sequence length must be at least two")
    if args.warmup_steps < 0 or not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("invalid warmup-steps or min-lr-ratio")
    if args.distill_top_k < 0 or args.cpu_threads < 0 or args.weight_tile_elements < 0:
        parser.error("distill-top-k, cpu-threads and weight-tile-elements must be nonnegative")
    if not 0.0 <= args.distillation_weight <= 1.0:
        parser.error("distillation weight must be between zero and one")
    if not 0 < args.learning_rate < math.inf or not 0 < args.temperature < math.inf:
        parser.error("learning rate and temperature must be positive and finite")
    if args.init_from is not None:
        if args.resume:
            parser.error("cannot combine --init-from with --resume")
        if not args.init_from.expanduser().is_file():
            parser.error(f"init-from checkpoint not found: {args.init_from}")
    if not args.train_text.expanduser().is_file():
        parser.error(f"train-text file not found: {args.train_text}")
    if not args.validation_text.expanduser().is_file():
        parser.error(f"validation-text file not found: {args.validation_text}")
    if args.train_text.expanduser().resolve() == args.validation_text.expanduser().resolve():
        parser.error("provide separate training and validation text files")
    return args


def sample_blocks(path: Path, tokenizer, count: Optional[int], length: int, seed: int):
    source = path.expanduser().read_bytes()
    ids = tokenizer(source.decode("utf-8"), add_special_tokens=False, verbose=False).input_ids
    available = len(ids) // length
    if not available or (count and count > available):
        raise ValueError(f"{path} has {available} blocks, but {count} were requested")
    blocks = torch.tensor(ids[:available * length], dtype=torch.long).reshape(available, length)
    if count:
        generator = torch.Generator().manual_seed(seed)
        blocks = blocks[torch.randperm(available, generator=generator)[:count]]
    return blocks, hashlib.sha256(source).hexdigest()


def _extract_rope_theta(teacher_config) -> float:
    theta = getattr(teacher_config, "rope_theta", None)
    if theta is not None:
        return float(theta)
    rope_params = getattr(teacher_config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        if rope_params.get("rope_theta") is not None:
            return float(rope_params["rope_theta"])
        for sub in rope_params.values():
            if isinstance(sub, dict) and sub.get("rope_theta") is not None:
                return float(sub["rope_theta"])
    return 1000000.0


def student_config(teacher_config, args: argparse.Namespace) -> D16QwenConfig:
    num_layers = int(args.num_layers) if args.num_layers is not None else int(teacher_config.num_hidden_layers)
    hidden_size = int(args.hidden_size) if args.hidden_size is not None else int(teacher_config.hidden_size)
    intermediate_size = (
        int(args.intermediate_size)
        if args.intermediate_size is not None
        else int(teacher_config.intermediate_size)
    )
    return D16QwenConfig(
        vocab_size=int(teacher_config.vocab_size),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=int(teacher_config.num_attention_heads),
        num_key_value_heads=int(getattr(teacher_config, "num_key_value_heads", teacher_config.num_attention_heads)),
        rms_norm_eps=float(getattr(teacher_config, "rms_norm_eps", 1e-6)),
        rope_theta=_extract_rope_theta(teacher_config),
        tie_word_embeddings=bool(getattr(teacher_config, "tie_word_embeddings", True)),
        attention_bias=bool(getattr(teacher_config, "attention_bias", True)),
        mlp_bias=bool(getattr(teacher_config, "mlp_bias", False)),
        indiff_eps=args.indiff_eps,
        indiff_rel=float(args.indiff_rel),
        mlp_activation=args.mlp_activation,
        collect_d16_stats=False,
    )


def _teacher_forward(
    teacher,
    ids_cpu: torch.Tensor,
    teacher_device: torch.device,
    student_device: torch.device,
    temperature: float,
    top_k: int,
    *,
    compute_teacher_nll: bool = False,
):
    """Run frozen teacher on `teacher_device` and transfer compact targets to `student_device`."""
    with torch.no_grad():
        t_ids = ids_cpu.to(teacher_device)
        t_logits = teacher(input_ids=t_ids, use_cache=False).logits[:, :-1]
        teacher_nll = None
        if compute_teacher_nll:
            teacher_nll = float(
                F.cross_entropy(t_logits.float().flatten(0, 1), t_ids[:, 1:].reshape(-1)).item()
            )
        vocab = t_logits.shape[-1]
        if top_k > 0 and top_k < vocab:
            t_vals, t_idx = (t_logits.float() / temperature).topk(top_k, dim=-1)
            t_log_p = F.log_softmax(t_vals, dim=-1)
            t_p = t_log_p.exp()
            t_log_p = t_log_p.masked_fill(t_p == 0, 0.0)
            payload = (
                "topk",
                t_idx.to(student_device, non_blocking=True),
                t_p.to(student_device, non_blocking=True),
                t_log_p.to(student_device, non_blocking=True),
            )
        else:
            payload = ("full", t_logits.to(student_device, non_blocking=True))
        del t_logits
    return payload, teacher_nll


def batch_losses(
    student: D16QwenForCausalLM,
    teacher,
    ids_cpu: torch.Tensor,
    *,
    teacher_device: torch.device,
    student_device: torch.device,
    temperature: float,
    top_k: int,
    compute_teacher_nll: bool = False,
):
    teacher_payload, teacher_nll = _teacher_forward(
        teacher,
        ids_cpu,
        teacher_device,
        student_device,
        temperature,
        top_k,
        compute_teacher_nll=compute_teacher_nll,
    )
    ids_student = ids_cpu.to(student_device, non_blocking=True)
    d16_ids = encode_token_ids(ids_student)
    out = student(input_ids=d16_ids, labels=d16_ids)
    assert out.loss is not None
    ce = out.loss
    logits = out.logits[:, :-1, :]

    if teacher_payload[0] == "topk":
        _, t_idx, t_p, t_log_p = teacher_payload
        kl = _distillation_kl_from_topk(logits, t_idx, t_p, t_log_p, temperature)
    else:
        _, t_logits = teacher_payload
        kl = distillation_kl(logits, t_logits, temperature, top_k=0)

    sampling_nll = None
    if compute_teacher_nll:
        with torch.no_grad():
            sampling_nll = float(logits.sampling_nll(ids_student[:, 1:], 1.0, eps=0.0, rel=0.0).item())
    return ce, kl, teacher_nll, sampling_nll


@torch.no_grad()
def evaluate(
    student: D16QwenForCausalLM,
    teacher,
    blocks: torch.Tensor,
    args: argparse.Namespace,
    student_device: torch.device,
    teacher_device: torch.device,
) -> dict[str, float | int]:
    student.eval()
    prev_collect = student.config.collect_d16_stats
    student.set_collect_d16_stats(True, reset=True)
    totals = {"student_nll": 0.0, "student_sampling_nll": 0.0, "teacher_nll": 0.0, "distillation_kl": 0.0}
    token_count = 0
    try:
        for block in blocks.split(args.batch_size):
            ce, kl, teacher_nll, sampling_nll = batch_losses(
                student,
                teacher,
                block,
                teacher_device=teacher_device,
                student_device=student_device,
                temperature=args.temperature,
                top_k=args.distill_top_k,
                compute_teacher_nll=True,
            )
            n_tok = int(block.shape[0] * (block.shape[1] - 1))
            ce_val = float(ce.x.masked_fill(ce.is_nan(), float("nan")).item())
            kl_val = float(kl.x.masked_fill(kl.is_nan(), float("nan")).item())
            totals["student_nll"] += ce_val * n_tok
            totals["student_sampling_nll"] += float(sampling_nll) * n_tok
            totals["teacher_nll"] += float(teacher_nll) * n_tok
            totals["distillation_kl"] += kl_val * n_tok
            token_count += n_tok

        act_summary = student.get_d16_state_summary()
    finally:
        student.set_collect_d16_stats(prev_collect, reset=True)

    weight_zero = 0
    weight_total = 0
    for param in student.d16_parameters():
        if param.x.ndim >= 2:
            val = param.value()
            weight_zero += int(val.is_zero().sum().item())
            weight_total += val.numel()

    metrics: dict[str, float | int] = {name: total / token_count for name, total in totals.items()}
    metrics["student_perplexity"] = _safe_perplexity(float(metrics["student_sampling_nll"]))
    metrics["teacher_perplexity"] = _safe_perplexity(float(metrics["teacher_nll"]))
    metrics["act_indiff_pct"] = float(act_summary["indifferent_pct"])
    metrics["weight_indiff_pct"] = (weight_zero / max(weight_total, 1)) * 100.0
    metrics["tokens"] = token_count
    return metrics


def write_event(log, event: str, **fields) -> None:
    encoded = json.dumps(
        {"event": event, "time": datetime.now(timezone.utc).isoformat(), **fields},
        ensure_ascii=False,
        allow_nan=False,
    )
    print(encoded, file=log, flush=True)
    print(encoded, flush=True)


def _update_accumulated(
    student: D16QwenForCausalLM,
    teacher,
    optimizer: D16AdamW,
    micro_batches: list[torch.Tensor],
    args: argparse.Namespace,
    student_device: torch.device,
    teacher_device: torch.device,
    cur_lr: float,
) -> tuple[dict[str, float], int]:
    optimizer.lr = cur_lr
    optimizer.zero_grad(set_to_none=True)

    total_tokens = sum(int(b.shape[0] * (b.shape[1] - 1)) for b in micro_batches)
    ce_acc, kl_acc, obj_acc = 0.0, 0.0, 0.0
    kd_scale = args.distillation_weight * (args.temperature ** 2)
    ce_scale = 1.0 - args.distillation_weight

    for ids_cpu in micro_batches:
        n_tok = int(ids_cpu.shape[0] * (ids_cpu.shape[1] - 1))
        weight = n_tok / float(total_tokens)
        ce, kl, _, _ = batch_losses(
            student,
            teacher,
            ids_cpu,
            teacher_device=teacher_device,
            student_device=student_device,
            temperature=args.temperature,
            top_k=args.distill_top_k,
            compute_teacher_nll=False,
        )
        objective = ce_scale * ce.x.float() + kd_scale * kl.x.float()
        vals = [float(ce.x.item()), float(kl.x.item()), float(objective.item())]
        if bool(ce.is_nan()) or bool(kl.is_nan()) or not all(math.isfinite(v) for v in vals):
            raise RuntimeError(f"Undefined loss during distillation step: ce={vals[0]}, kl={vals[1]}, obj={vals[2]}")

        (objective * weight).backward()
        ce_acc += vals[0] * weight
        kl_acc += vals[1] * weight
        obj_acc += vals[2] * weight
        del ce, kl, objective

    grad_norm = float(optimizer.step(max_grad_norm=1.0, release_grads=True).item())
    return {
        "lr": cur_lr,
        "student_nll": ce_acc,
        "distillation_kl": kl_acc,
        "objective": obj_acc,
        "grad_norm": grad_norm,
    }, total_tokens


def _validate_and_save_best(
    student: D16QwenForCausalLM,
    teacher,
    validation_blocks: torch.Tensor,
    args: argparse.Namespace,
    student_device: torch.device,
    teacher_device: torch.device,
    state: TrainingState,
    metadata: dict,
) -> dict[str, float | int]:
    metrics = evaluate(student, teacher, validation_blocks, args, student_device, teacher_device)
    if state.best_validation is None or metrics["student_nll"] < state.best_validation["student_nll"]:
        save_inference_checkpoint(
            args.output_dir / "best.pt",
            student,
            config=metadata["config"],
            epoch=state.epoch,
            step=state.step,
            validation=metrics,
            teacher_revision=args.teacher_revision,
            weight_storage="d16",
        )
        state.best_epoch = state.epoch
        state.best_validation = metrics
    return metrics


def train(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer, __version__ as transformers_version

    started = perf_counter()
    student_device = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device)
    torch.manual_seed(args.seed)

    # Use up to 12 physical cores on multi-core CPUs (e.g. Ryzen 9) when teacher runs on CPU
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    else:
        avail = os.cpu_count() or 4
        target_threads = min(12, avail) if teacher_device.type == "cpu" else min(4, avail)
        torch.set_num_threads(target_threads)

    if args.tf32 and student_device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    if args.weight_tile_elements > 0:
        D16Linear._MAX_WEIGHT_TILE = args.weight_tile_elements

    teacher_source, local_only = resolve_teacher_source(
        args.teacher_dir, force_local_only=args.local_files_only
    )

    latest = args.output_dir / "latest.pt"
    if args.resume and not latest.is_file():
        raise FileNotFoundError(f"Missing resumable checkpoint {latest}")
    if args.resume:
        _ = read_training_metadata(latest)

    # Load teacher & tokenizer BEFORE creating output_dir so bad paths/downloads don't leave broken folders
    tok_kwargs: dict = {"local_files_only": local_only}
    if args.teacher_revision is not None:
        tok_kwargs["revision"] = args.teacher_revision
    tokenizer = AutoTokenizer.from_pretrained(teacher_source, **tok_kwargs)
    teacher_dtype = _resolve_dtype(args.teacher_dtype, teacher_device, prefer_bf16_on_cuda=True)
    student_dtype = _resolve_dtype(args.student_dtype, student_device, prefer_bf16_on_cuda=True)
    teacher = _load_teacher_model(
        teacher_source,
        teacher_dtype,
        teacher_device,
        local_files_only=local_only,
        revision=args.teacher_revision,
    )

    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    log_path = args.output_dir / "training_log.jsonl"

    with log_path.open("a" if args.resume else "x", encoding="utf-8") as log:
        write_event(
            log,
            "setup",
            resumed=args.resume,
            teacher_source=teacher_source,
            teacher_local_only=local_only,
            arguments={key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
            pytorch=torch.__version__,
            transformers=transformers_version,
            student_device=torch.cuda.get_device_name(student_device) if student_device.type == "cuda" else str(student_device),
            teacher_device=str(teacher_device),
            cpu_threads=torch.get_num_threads(),
        )

        config = student_config(teacher.config, args)
        student = D16QwenForCausalLM(config)
        if args.init_from is not None:
            _load_student_init_weights(args.init_from, student)
            write_event(log, "init_from", checkpoint=str(args.init_from))
        student = student.to(device=student_device, dtype=student_dtype)
        student.gradient_checkpointing_enable()

        optimizer = D16AdamW(
            student.d16_parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            indiff_eps=args.optimizer_indiff_eps,
            indiff_rel=args.optimizer_indiff_rel,
        )

        train_blocks, train_hash = sample_blocks(
            args.train_text, tokenizer, args.train_blocks, args.sequence_length, args.seed
        )
        validation_blocks, validation_hash = sample_blocks(
            args.validation_text, tokenizer, args.validation_blocks, args.sequence_length, args.seed + 1
        )
        macro_batch_blocks = args.batch_size * args.grad_accum_steps
        steps_per_epoch = math.ceil(len(train_blocks) / macro_batch_blocks)
        total_steps = args.epochs * steps_per_epoch

        metadata = {
            "config": asdict(config),
            "weight_storage": "d16",
            "teacher_source": teacher_source,
            "teacher_revision": args.teacher_revision,
            "teacher_device": str(teacher_device),
            "teacher_dtype": str(teacher_dtype),
            "student_dtype": str(student_dtype),
            "train_sha256": train_hash,
            "validation_sha256": validation_hash,
            "train_blocks": len(train_blocks),
            "validation_blocks": len(validation_blocks),
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "temperature": args.temperature,
            "distillation_weight": args.distillation_weight,
            "distill_top_k": args.distill_top_k,
            "optimizer_indiff_eps": args.optimizer_indiff_eps,
            "optimizer_indiff_rel": args.optimizer_indiff_rel,
            "tf32": args.tf32,
        }
        write_event(
            log,
            "dataset",
            **metadata,
            tokenizer_vocab=len(tokenizer),
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
            student_d16_elements=sum(m.x.numel() for m in student.d16_parameters()),
            student_float_parameters=sum(p.numel() for p in student.parameters()),
            teacher_parameters=sum(p.numel() for p in teacher.parameters()),
        )

        if args.resume:
            state = load_training_checkpoint(latest, student, optimizer, metadata)
            if state.batch > steps_per_epoch or state.step != (state.epoch - 1) * steps_per_epoch + state.batch:
                raise ValueError("Checkpoint cursor does not match this dataset")
            write_event(log, "resume", epoch=state.epoch, batch=state.batch, step=state.step)
        else:
            state = TrainingState()
            state.baseline = evaluate(student, teacher, validation_blocks, args, student_device, teacher_device)
            write_event(log, "epoch", epoch=0, step=0, validation=state.baseline)

        if student_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(student_device)

        while state.epoch <= args.epochs:
            epoch_start = perf_counter()
            student.train()
            order = torch.Generator().manual_seed(args.seed + state.epoch + 1)
            macro_batches = torch.randperm(len(train_blocks), generator=order).split(macro_batch_blocks)

            for macro_indices in macro_batches[state.batch:]:
                if args.stop_after_step is not None and state.step >= args.stop_after_step:
                    save_training_checkpoint(latest, student, optimizer, state, metadata)
                    write_event(log, "paused", epoch=state.epoch, batch=state.batch, step=state.step)
                    return

                step_start = perf_counter()
                cur_lr = _scheduled_lr(
                    state.step, total_steps, args.learning_rate, args.warmup_steps, args.min_lr_ratio
                )
                micro_list = [train_blocks[idx] for idx in macro_indices.split(args.batch_size)]
                metrics, tokens = _update_accumulated(
                    student,
                    teacher,
                    optimizer,
                    micro_list,
                    args,
                    student_device,
                    teacher_device,
                    cur_lr,
                )
                state.tokens += tokens
                state.step += 1
                state.batch += 1
                for name in ("student_nll", "distillation_kl", "objective"):
                    state.totals[name] = state.totals.get(name, 0.0) + tokens * metrics[name]
                write_event(
                    log,
                    "step",
                    epoch=state.epoch,
                    step=state.step,
                    batch=state.batch,
                    tokens=tokens,
                    seconds=perf_counter() - step_start,
                    **metrics,
                )
                if state.batch == steps_per_epoch:
                    break
                if state.step % args.evaluate_every == 0:
                    validation = _validate_and_save_best(
                        student, teacher, validation_blocks, args, student_device, teacher_device, state, metadata
                    )
                    write_event(log, "validation", epoch=state.epoch, step=state.step, validation=validation)
                    student.train()
                if state.step == 1 or state.step % args.checkpoint_every == 0:
                    save_training_checkpoint(latest, student, optimizer, state, metadata)
                    write_event(log, "checkpoint", epoch=state.epoch, batch=state.batch, step=state.step)

            validation = _validate_and_save_best(
                student, teacher, validation_blocks, args, student_device, teacher_device, state, metadata
            )
            training = {key: total / max(state.tokens, 1) for key, total in state.totals.items()}
            training["student_perplexity"] = _safe_perplexity(float(training.get("student_nll", float("nan"))))
            training["tokens"] = state.tokens
            write_event(
                log,
                "epoch",
                epoch=state.epoch,
                step=state.step,
                validation=validation,
                training=training,
                seconds=perf_counter() - epoch_start,
            )
            state.epoch += 1
            state.batch = 0
            state.tokens = 0
            state.totals.clear()
            save_training_checkpoint(latest, student, optimizer, state, metadata)

        peak = torch.cuda.max_memory_allocated(student_device) if student_device.type == "cuda" else None
        write_event(
            log,
            "complete",
            epochs=args.epochs,
            steps=state.step,
            best_epoch=state.best_epoch,
            baseline=state.baseline,
            best_validation=state.best_validation,
            seconds=perf_counter() - started,
            peak_cuda_bytes=peak,
        )


if __name__ == "__main__":
    train(parse_args())