"""Distill a frozen local Qwen2.5 checkpoint into a D16 model on text blocks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from d16_adamw import D16AdamW
from d16_parameter import D16Parameter
from d16_qwen_model2 import (
    D16ComplexTensor,
    D16QwenConfig,
    D16QwenForCausalLM,
    _REAL_NEGATIVE,
    _from_real,
    _real_add,
    _real_cross_entropy,
    _scale_wedge,
    _shifted_exp,
    _unpack_complex,
    encode_token_ids,
)
from d16_qwen_training_state import (
    TrainingState,
    load_training_checkpoint,
    read_training_metadata,
    save_inference_checkpoint,
    save_training_checkpoint,
)


def distillation_kl(
    student_scores: D16ComplexTensor, teacher_logits: torch.Tensor, temperature: float
) -> D16ComplexTensor:
    """Mean token-level KL(teacher || student) on matching, aligned vocabularies."""
    if student_scores.shape != teacher_logits.shape or teacher_logits.ndim < 2 or not teacher_logits.numel():
        raise ValueError("Distillation requires nonempty, matching [..., vocabulary] logits")
    if not 0.0 < temperature < float("inf"):
        raise ValueError("Distillation temperature must be positive and finite")

    dtype = torch.float64 if torch.float64 in (student_scores.dtype, teacher_logits.dtype) else torch.float32
    student_scores = student_scores.to(dtype)
    student_scores = _scale_wedge(student_scores, student_scores.x.new_tensor(1.0 / temperature))
    undefined = student_scores.is_nan()
    maximum, shifted_exp = _shifted_exp(student_scores, ~undefined)
    log_normalizer = shifted_exp.sum(dim=-1).log()

    with torch.no_grad():
        teacher_log_probs = (teacher_logits.to(dtype) / temperature).log_softmax(dim=-1)
        teacher_probs = _from_real(teacher_log_probs.exp())
        teacher_log_probs = teacher_log_probs.masked_fill(teacher_probs.x == 0, 0)

    # The distance from the row maximum uses the two rays of the real D16 axis.
    distance = torch.where(
        student_scores.rotation == _REAL_NEGATIVE, maximum + student_scores.x, maximum - student_scores.x
    )
    weighted_distance = (teacher_probs.x * (teacher_log_probs + distance)).sum(dim=-1)
    mean = (log_normalizer + weighted_distance).mean().clamp_min(0)
    return _from_real(mean.masked_fill(undefined.any() | teacher_probs.is_nan().any(), torch.nan))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--teacher-revision")
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--validation-text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-blocks", type=int, help="Limit the training split to a seeded sample")
    parser.add_argument("--validation-blocks", type=int, default=24, help="Fixed held-out sample; zero uses the full split")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save resumable state every N updates")
    parser.add_argument("--evaluate-every", type=int, default=500, help="Validate every N updates and at each epoch end")
    parser.add_argument("--resume", action="store_true", help="Restore an existing run from latest.pt")
    parser.add_argument("--weight-storage", choices=("d16", "cartesian"),
                        help="New runs default to D16; resumes use their recorded format")
    parser.add_argument("--stop-after-step", type=int, help="Save and pause after this absolute optimizer step")
    parser.add_argument("--tf32", action="store_true", help="Permit TF32 for FP32 coordinate matrix products on CUDA")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.checkpoint_every, args.evaluate_every) <= 0:
        parser.error("epochs, batch size and intervals must be positive")
    if args.validation_blocks < 0 or (args.train_blocks is not None and args.train_blocks <= 0):
        parser.error("invalid block count")
    if args.stop_after_step is not None and args.stop_after_step <= 0:
        parser.error("stop-after-step must be positive")
    if args.sequence_length < 2:
        parser.error("sequence length must be at least two")
    if not 0.0 <= args.distillation_weight <= 1.0:
        parser.error("distillation weight must be between zero and one")
    if not 0 < args.learning_rate < math.inf or not 0 < args.temperature < math.inf:
        parser.error("learning rate and temperature must be positive and finite")
    if args.train_text.resolve() == args.validation_text.resolve():
        parser.error("provide separate training and validation text files")
    return args


def sample_blocks(path, tokenizer, count, length, seed):
    source = path.read_bytes()
    ids = tokenizer(source.decode("utf-8"), add_special_tokens=False, verbose=False).input_ids
    available = len(ids) // length
    if not available or (count and count > available):
        raise ValueError(f"{path} has {available} blocks, but {count} were requested")
    blocks = torch.tensor(ids[:available * length], dtype=torch.long).reshape(available, length)
    if count:
        generator = torch.Generator().manual_seed(seed)
        blocks = blocks[torch.randperm(available, generator=generator)[:count]]
    return blocks, hashlib.sha256(source).hexdigest()


def _student_losses(logits, targets, teacher_logits, temperature):
    real_scores, _ = _unpack_complex(logits)
    ce = _real_cross_entropy(real_scores, targets)
    return ce, distillation_kl(real_scores, teacher_logits, temperature)


def batch_losses(student, teacher, ids, temperature):
    with torch.no_grad():
        teacher_logits = teacher(input_ids=ids, use_cache=False).logits[:, :-1]
    logits = student(encode_token_ids(ids)).logits[:, :-1]
    operands = (logits, ids[:, 1:], teacher_logits, temperature)
    if torch.is_grad_enabled():
        ce, kl = checkpoint(_student_losses, *operands, use_reentrant=False, preserve_rng_state=False)
    else:
        ce, kl = _student_losses(*operands)
    return ce, kl, teacher_logits


@torch.no_grad()
def evaluate(student, teacher, blocks, args, device):
    student.eval()
    totals = {"student_nll": 0.0, "teacher_nll": 0.0, "distillation_kl": 0.0}
    token_count = 0
    for block in blocks.split(args.batch_size):
        ids = block.to(device)
        ce, kl, teacher_logits = batch_losses(student, teacher, ids, args.temperature)
        targets = ids[:, 1:]
        teacher_nll = F.cross_entropy(teacher_logits.float().flatten(0, 1), targets.reshape(-1))
        values = torch.stack((ce.x.masked_fill(ce.is_nan(), torch.nan),
                              teacher_nll, kl.x.masked_fill(kl.is_nan(), torch.nan))).tolist()
        for name, value in zip(totals, values):
            totals[name] += value * targets.numel()
        token_count += targets.numel()

    metrics = {name: total / token_count for name, total in totals.items()}
    metrics["student_perplexity"] = math.exp(metrics["student_nll"])
    metrics["teacher_perplexity"] = math.exp(metrics["teacher_nll"])
    metrics["tokens"] = token_count
    return metrics


def write_event(log, event, **fields):
    encoded = json.dumps({"event": event, "time": datetime.now(timezone.utc).isoformat(), **fields},
                         ensure_ascii=False, allow_nan=False)
    print(encoded, file=log, flush=True)
    print(encoded, flush=True)


def student_config(teacher_config) -> D16QwenConfig:
    return D16QwenConfig(
        vocab_size=teacher_config.vocab_size,
        hidden_size=teacher_config.hidden_size,
        intermediate_size=teacher_config.intermediate_size,
        num_hidden_layers=teacher_config.num_hidden_layers,
        num_attention_heads=teacher_config.num_attention_heads,
        num_key_value_heads=teacher_config.num_key_value_heads,
        max_position_embeddings=teacher_config.max_position_embeddings,
        rms_norm_eps=teacher_config.rms_norm_eps,
        rope_theta=teacher_config.rope_theta,
        tie_word_embeddings=teacher_config.tie_word_embeddings,
        attention_bias=getattr(teacher_config, "attention_bias", True),
        mlp_bias=getattr(teacher_config, "mlp_bias", False),
    )


def _update(student, teacher, optimizer, ids, args) -> tuple[dict[str, float], int]:
    optimizer.zero_grad(set_to_none=True)
    ce, kl, _ = batch_losses(student, teacher, ids, args.temperature)
    kd_scale = args.distillation_weight * args.temperature ** 2
    objective = _real_add(
        _scale_wedge(ce, ce.x.new_tensor(1 - args.distillation_weight)),
        _scale_wedge(kl, kl.x.new_tensor(kd_scale)),
    )
    values = torch.stack((ce.x, kl.x, objective.x)).tolist()
    if bool(objective.is_nan()) or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Undefined loss")
    objective.x.backward()
    if isinstance(optimizer, D16AdamW):
        grad_norm = optimizer.step(max_grad_norm=1.0)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "student_nll": values[0],
        "distillation_kl": values[1],
        "objective": values[2],
        "grad_norm": float(grad_norm),
    }, ids.shape[0] * (ids.shape[1] - 1)


def _validate_and_save_best(student, teacher, validation_blocks, args, device, state, metadata):
    metrics = evaluate(student, teacher, validation_blocks, args, device)
    if state.best_validation is None or metrics["student_nll"] < state.best_validation["student_nll"]:
        save_inference_checkpoint(
            args.output_dir / "best.pt", student, config=metadata["config"], epoch=state.epoch,
            step=state.step, validation=metrics, teacher_revision=args.teacher_revision,
            weight_storage=metadata["weight_storage"],
        )
        state.best_epoch = state.epoch
        state.best_validation = metrics
    return metrics


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    started = perf_counter()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.tf32 and device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    log_path = args.output_dir / "training_log.jsonl"
    latest = args.output_dir / "latest.pt"
    if args.resume and not latest.is_file():
        raise FileNotFoundError(f"Missing resumable checkpoint {latest}")
    resume_metadata = read_training_metadata(latest) if args.resume else None
    if resume_metadata is None:
        weight_storage = args.weight_storage or "d16"
    else:
        weight_storage = resume_metadata.get("weight_storage", "cartesian")
        if args.weight_storage is not None and args.weight_storage != weight_storage:
            raise ValueError(f"The saved run uses weight storage {weight_storage!r}")

    with log_path.open("a" if args.resume else "x", encoding="utf-8") as log:
        write_event(
            log, "setup", resumed=args.resume,
            arguments={key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
            pytorch=torch.__version__, transformers=transformers_version,
            device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        )
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_dir, local_files_only=True)
        teacher_dtype = torch.float32
        if device.type == "cuda":
            teacher_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_dir, dtype=teacher_dtype, local_files_only=True,
        ).to(device).eval().requires_grad_(False)

        config = student_config(teacher.config)
        student = D16QwenForCausalLM(config, weight_storage=weight_storage).to(device)
        student.gradient_checkpointing_enable()
        if weight_storage == "d16":
            native_parameters = (module for module in student.modules() if isinstance(module, D16Parameter))
            optimizer = D16AdamW(native_parameters, lr=args.learning_rate)
        else:
            optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, fused=device.type == "cuda")
        train_blocks, train_hash = sample_blocks(
            args.train_text, tokenizer, args.train_blocks, args.sequence_length, args.seed,
        )
        validation_blocks, validation_hash = sample_blocks(
            args.validation_text, tokenizer, args.validation_blocks, args.sequence_length, args.seed + 1,
        )
        metadata = {
            "config": asdict(config),
            "weight_storage": weight_storage,
            "teacher_revision": args.teacher_revision,
            "teacher_dtype": str(teacher_dtype),
            "train_sha256": train_hash,
            "validation_sha256": validation_hash,
            "train_blocks": len(train_blocks),
            "validation_blocks": len(validation_blocks),
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "distillation_weight": args.distillation_weight,
            "tf32": args.tf32,
        }
        steps_per_epoch = math.ceil(len(train_blocks) / args.batch_size)
        write_event(
            log, "dataset", **metadata, tokenizer_vocab=len(tokenizer), steps_per_epoch=steps_per_epoch,
            student_parameters=sum(p.numel() for p in student.parameters()),
            teacher_parameters=sum(p.numel() for p in teacher.parameters()),
        )

        if args.resume:
            expected_metadata = metadata.copy()
            if "weight_storage" not in resume_metadata:
                del expected_metadata["weight_storage"]
            state = load_training_checkpoint(latest, student, optimizer, expected_metadata)
            if state.batch > steps_per_epoch or state.step != (state.epoch - 1) * steps_per_epoch + state.batch:
                raise ValueError("Checkpoint cursor does not match this dataset")
            write_event(log, "resume", epoch=state.epoch, batch=state.batch, step=state.step)
        else:
            state = TrainingState()
            state.baseline = evaluate(student, teacher, validation_blocks, args, device)
            write_event(log, "epoch", epoch=0, step=0, validation=state.baseline)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        while state.epoch <= args.epochs:
            epoch_start = perf_counter()
            student.train()
            order = torch.Generator().manual_seed(args.seed + state.epoch + 1)
            batches = torch.randperm(len(train_blocks), generator=order).split(args.batch_size)
            for indices in batches[state.batch:]:
                if args.stop_after_step is not None and state.step >= args.stop_after_step:
                    save_training_checkpoint(latest, student, optimizer, state, metadata)
                    write_event(log, "paused", epoch=state.epoch, batch=state.batch, step=state.step)
                    return
                step_start = perf_counter()
                metrics, tokens = _update(student, teacher, optimizer, train_blocks[indices].to(device), args)
                state.tokens += tokens
                state.step += 1
                state.batch += 1
                for name in ("student_nll", "distillation_kl", "objective"):
                    state.totals[name] = state.totals.get(name, 0.0) + tokens * metrics[name]
                write_event(
                    log, "step", epoch=state.epoch, step=state.step, batch=state.batch,
                    tokens=tokens, seconds=perf_counter() - step_start, **metrics,
                )
                if state.batch == steps_per_epoch:
                    break
                if state.step % args.evaluate_every == 0:
                    validation = _validate_and_save_best(
                        student, teacher, validation_blocks, args, device, state, metadata,
                    )
                    write_event(log, "validation", epoch=state.epoch, step=state.step, validation=validation)
                    student.train()
                if state.step == 1 or state.step % args.checkpoint_every == 0:
                    save_training_checkpoint(latest, student, optimizer, state, metadata)
                    write_event(log, "checkpoint", epoch=state.epoch, batch=state.batch, step=state.step)

            validation = _validate_and_save_best(
                student, teacher, validation_blocks, args, device, state, metadata,
            )
            training = {key: total / state.tokens for key, total in state.totals.items()}
            training["student_perplexity"] = math.exp(training["student_nll"])
            training["tokens"] = state.tokens
            write_event(
                log, "epoch", epoch=state.epoch, step=state.step,
                validation=validation, training=training, seconds=perf_counter() - epoch_start,
            )
            state.epoch += 1
            state.batch = 0
            state.tokens = 0
            state.totals.clear()
            save_training_checkpoint(latest, student, optimizer, state, metadata)

        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        write_event(
            log, "complete", epochs=args.epochs, steps=state.step, best_epoch=state.best_epoch,
            baseline=state.baseline, best_validation=state.best_validation,
            seconds=perf_counter() - started, peak_cuda_bytes=peak,
        )


if __name__ == "__main__":
    train(parse_args())
