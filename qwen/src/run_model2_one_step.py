"""Import a local Qwen2.5 checkpoint into model2 and measure one D16AdamW update."""
import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from d16_adamw import D16AdamW
from d16_complex import D16ComplexTensor
from d16_parameter import D16Parameter
from d16_qwen_importer import import_qwen_weights, load_qwen_checkpoint, load_d16_checkpoint, save_d16_checkpoint
from d16_qwen_model2 import encode_token_ids


TEXTS = {
    "train": [
        "The train leaves the station every morning and crosses a bridge over the river.",
        "When the rain stopped the children ran outside to play in the garden again.",
        "A computer program can read a file and write a new list of numbers.",
        "At the end of the road there was a small wooden house with a red roof.",
    ],
    "held_out": [
        "The capital of France is Paris. The capital of Italy is Rome.",
        "A triangle has three sides and the sum of its interior angles is always the same.",
        "Water freezes at zero degrees Celsius and boils at one hundred degrees Celsius.",
        "Вода замерзает при нуле градусов. Столица Франции называется Париж.",
    ],
}


@torch.no_grad()
def reference_metrics(model, batches):
    result = {}
    for split, ids in batches.items():
        logits = model(ids, use_cache=False).logits[:, :-1].float()
        targets = ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
        hits = (logits.argmax(-1) == targets).sum().item()
        result[split] = dict(nll=nll, ppl=math.exp(nll), correct=hits, predictions=targets.numel())
    return result


@torch.no_grad()
def d16_metrics(model, batches):
    model.eval()
    result = {}
    for split, ids in batches.items():
        tokens = encode_token_ids(ids)
        out = model(tokens, labels=tokens, sampling_temperature=1.0)
        hits = (out.logits[:, :-1].wedge_argmax(eps=0).squeeze(-1) == ids[:, 1:]).sum().item()
        result[split] = dict(nll=out.nll.item(), ppl=out.perplexity.item(),
                             train_objective=out.loss.x.item(), correct=hits,
                             predictions=ids[:, 1:].numel())
    return result


def tiles(shape):
    rows = max(1, (1 << 18) // math.prod(shape[1:]))
    for start in range(0, shape[0], rows):
        yield slice(start, start + rows)


@torch.no_grad()
def verify_import(model, report, source):
    state, _ = load_qwen_checkpoint(source)
    pairs = dict(model.named_modules())
    checked = nonzero_im = 0
    max_error = 0.0
    for name, weight in state.items():
        parent, _, field = name.rpartition(".")
        if field == "weight" and parent.endswith("norm"):
            field = "gain"
        target = pairs[f"{parent}.{field}"].value()
        if target.shape != weight.shape:
            raise ValueError(f"Shape mismatch for {name}")
        for idx in tiles(weight.shape):
            real, imag = target[idx].unpack()
            ref = weight[idx].float()
            if not torch.allclose(real, ref, rtol=2e-6, atol=1e-6) or not torch.isfinite(imag).all():
                raise ValueError(f"Coordinate check failed for {name} at {idx}")
            max_error = max(max_error, (real - ref).abs().max().item())
            nonzero_im += imag.count_nonzero().item()
        checked += weight.numel()
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    if len(state) != report.total_imported_tensors or checked != report.total_weights:
        raise ValueError("Import counts differ from source checkpoint")
    if tied != model.config.tie_word_embeddings or not nonzero_im:
        raise ValueError("Unexpected output weight tying or empty imaginary component")
    return dict(source_tensors=len(state), checked_values=checked, max_abs_re_error=max_error,
                nonzero_im=nonzero_im, tied_embeddings=tied)


@torch.no_grad()
def update_metrics(checkpoint, named):
    """Measure Cartesian changes by layer against the checkpoint on disk."""
    state = torch.load(checkpoint, weights_only=True, mmap=True)["state_dict"]
    groups = {}
    for name, param in named.items():
        group = ".".join(name.split(".")[:3]) if name.startswith("model.layers.") else name
        stats = groups.setdefault(group, dict(weights=0, changed=0, mask_changes=0,
                                              squared_delta=0.0, squared_weight=0.0))
        old = D16ComplexTensor(*(state[name + "." + k] for k in ("x", "y", "rotation")))
        stats["weights"] += old.x.numel()
        for idx in tiles(old.shape):
            a, b = old[idx].unpack()
            current = param.value()[idx]
            c, d = current.unpack()
            stats["changed"] += ((a != c) | (b != d)).sum().item()
            stats["mask_changes"] += (old.rotation[idx] != current.rotation).sum().item()
            stats["squared_delta"] += ((c-a).square() + (d-b).square()).sum(dtype=torch.float64).item()
            stats["squared_weight"] += (a.square() + b.square()).sum(dtype=torch.float64).item()
    totals = {k: sum(group[k] for group in groups.values()) for k in next(iter(groups.values()))}
    for group in [*groups.values(), totals]:
        group["relative_l2"] = math.sqrt(group.pop("squared_delta") / group.pop("squared_weight"))
    return {**totals, "by_group": groups}


def train_full_step(model, report, batches, metrics, run, previous_step):
    if any(layer.mlp.mlp_activation != metrics["mlp_activation"] for layer in model.model.layers):
        raise ValueError("Checkpoint MLP activation differs from experiment configuration")
    step = previous_step + 1
    previous_weights = run / (f"step-{previous_step:04d}.pt" if previous_step else "imported.pt")
    model_path = run / f"step-{step:04d}.pt"
    moments_path = run / f"optimizer-{step:04d}.pt"
    result_path = run / (f"metrics-step-{step:04d}.json" if previous_step else "metrics.json")
    if any(p.exists() for p in (model_path, moments_path, result_path)):
        raise FileExistsError(f"Output already exists for step {step} in {run}")
    model.requires_grad_(True)
    named = {n: p for n, p in model.named_modules() if isinstance(p, D16Parameter)}
    if {id(p) for p in model.parameters()} != {id(c) for p in named.values() for c in (p.x, p.y)}:
        raise RuntimeError("Optimizer does not cover all learned coordinates")
    optimizer = D16AdamW(named.values(), lr=metrics["lr"], betas=tuple(metrics["betas"]),
                         weight_decay=metrics["weight_decay"], indiff_eps=0.0, tile_size=1 << 18)
    if previous_step:
        old = torch.load(run / f"optimizer-{previous_step:04d}.pt", weights_only=True, mmap=True)
        if old["global_step"] != previous_step or old["optimizer"]["lr"] != metrics["lr"]:
            raise ValueError("Optimizer checkpoint does not match the requested step or learning rate")
        moments = dict(zip(old["parameter_names"], old["optimizer"]["moments"], strict=True))
        if not moments.keys() <= named.keys():
            raise ValueError("Unknown parameter in saved optimizer")
        # Parameters frozen in earlier steps start their Adam history on this step.
        optimizer.load_state_dict({**old["optimizer"], "moments": [moments.get(n) for n in named]})
        del old, moments
    metrics["trainable_layers"] = list(range(len(model.model.layers)))
    metrics["trainable_complex_weights"] = sum(p.x.numel() for p in named.values())
    metrics["trainable_tensors"] = len(named)
    model.gradient_checkpointing_enable()
    model.train()
    optimizer.zero_grad()
    tokens = encode_token_ids(batches["train"])
    out = model(tokens, labels=tokens)
    print(f"Step {step}, full-network backward, loss {out.loss.x.item():.8f}", flush=True)
    out.loss.x.backward()
    del out
    if any(p.x.grad is None or p.y.grad is None for p in named.values()):
        raise RuntimeError("Some coordinates were disconnected from the loss")
    metrics["unclipped_grad_norm"] = optimizer.step(
        max_grad_norm=metrics["max_grad_norm"], release_grads=True).item()
    metrics["optimizer_steps"] = step
    print("Saving optimizer moments and full model", flush=True)
    torch.save(dict(parameter_names=list(named), optimizer=optimizer.state_dict(),
                    global_step=step), moments_path)
    del optimizer
    save_d16_checkpoint(model, model_path, report=report)
    metrics["update"] = update_metrics(previous_weights, named)
    if not (0 < metrics["update"]["relative_l2"] < math.inf):
        raise RuntimeError("The update was zero or nonfinite")
    print("Actual changed values:", metrics["update"]["changed"], flush=True)
    metrics["after"] = d16_metrics(model, batches)
    print("After", metrics["after"], flush=True)
    result_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local Qwen2.5-0.5B Hugging Face snapshot")
    parser.add_argument("output", type=Path, help="New directory for two checkpoints and metrics.json")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mlp-activation", choices=("bilinear", "gate"),
                        help="MLP mode for full import (default: bilinear; resume: saved mode)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--from-step", type=int, default=0,
                        help="Resume from a saved step in the existing output directory")
    args = parser.parse_args()
    if not args.source.is_dir() or not (0 < args.lr < math.inf):
        parser.error("Supply a local checkpoint directory and a positive finite learning rate")
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    if args.from_step < 0:
        parser.error("--from-step must be nonnegative")
    if args.from_step:
        filename = "metrics.json" if args.from_step == 1 else f"metrics-step-{args.from_step:04d}.json"
        metrics = json.loads((args.output / filename).read_text(encoding="utf-8"))
        if (metrics["optimizer_steps"] != args.from_step or metrics["lr"] != args.lr
                or Path(metrics["source"]).resolve() != args.source.resolve()):
            parser.error("Source, learning rate or step differs from the recorded run")
        if args.mlp_activation is not None and args.mlp_activation != metrics["mlp_activation"]:
            parser.error("MLP activation differs from the saved run")
        batches = {k: torch.tensor(v) for k, v in metrics["tokens"].items()}
        metrics["before"] = metrics["after"]
        metrics["threads"] = args.threads
        model, report = load_d16_checkpoint(args.output / f"step-{args.from_step:04d}.pt",
                                            return_report=True)
        train_full_step(model, report, batches, metrics, args.output, args.from_step)
        return
    activation = args.mlp_activation or "bilinear"
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    batches = {}
    for split, texts in TEXTS.items():
        rows = [tokenizer.encode(s, add_special_tokens=False)[:12] for s in texts]
        if any(len(row) != 12 for row in rows):
            raise ValueError("All prompts must supply 12 tokens")
        batches[split] = torch.tensor(rows)
    metrics = dict(source=str(args.source.resolve()), torch=str(torch.__version__),
                   transformers=transformers.__version__, device="cpu", dtype="float32",
                   seed=42, threads=args.threads, lr=args.lr, optimizer="D16AdamW",
                   betas=[0.8, 0.99], weight_decay=0, max_grad_norm=1, temperature=1,
                   indiff_eps=0, mlp_activation=activation, second_coord_mode="d16_sat_rope_gate",
                   tokens={k: v.tolist() for k, v in batches.items()})
    print("Evaluating original Qwen", flush=True)
    original = AutoModelForCausalLM.from_pretrained(
        args.source, local_files_only=True, dtype=torch.float32, attn_implementation="eager")
    metrics["original"] = reference_metrics(original.eval(), batches)
    print("Original", metrics["original"], flush=True)
    del original
    gc.collect()
    model, report = import_qwen_weights(args.source, dtype=torch.float32, device="cpu",
                                        indiff_eps=0.0, mlp_activation=activation,
                                        second_coord_mode="d16_sat_rope_gate")
    layers = model.model.layers
    metrics["layers"] = len(layers)
    metrics["import"] = verify_import(model, report, args.source)
    print("Import verified", metrics["import"], flush=True)
    save_d16_checkpoint(model, args.output / "imported.pt", report=report)
    metrics["before"] = d16_metrics(model, batches)
    print("Before", metrics["before"], flush=True)

    train_full_step(model, report, batches, metrics, args.output, 0)


if __name__ == "__main__":
    main()
