"""Import all 24 Qwen2.5-0.5B layers into model2 and measure one D16AdamW update.

Run the phases in order: import, before, step, after, verify, report.
Use the same --source <local Hugging Face snapshot> and --run <output directory>
for each phase. Only "step" trains. The evaluation and verification phases may
be repeated against the saved checkpoints. The data files are read from data/.
Choose --activation on import. Subsequent phases read the saved configuration.
Use --reference-run on the report phase to compare activations on matching tokens.
"""

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "qwen" / "src"))
from d16_qwen_importer import (
    WEDGE_HALF_SAT, WEDGE_SAT_SLOPE, import_qwen_weights,
    load_d16_checkpoint, save_d16_checkpoint,
)
from d16_qwen_model2 import encode_token_ids
from d16_adamw import D16AdamW

LAYERS = 24
MODE = "d16_sat_rope_gate"
LR = 1e-5
CLIP = 1.0
MLP_OPERATIONS = {"bilinear": "up × gate", "gate": "up × sigmoid(gate)"}


def write_json(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_stage(model, run, name, report=None):
    directory = run / name
    if directory.exists():
        raise FileExistsError(directory)
    pending = run / (name + ".pending")
    save_d16_checkpoint(model, pending, report=report, use_safetensors=False)
    pending.rename(directory)


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(source, run, activation="bilinear"):
    if (run / "step0").exists() or (run / "step1").exists():
        raise FileExistsError("Use an unused run directory for a fresh import.")
    start = time.monotonic()
    config = read_json(source / "config.json")
    assert config["num_hidden_layers"] == LAYERS
    weights = source / "model.safetensors"
    with safe_open(str(weights), framework="pt") as raw:
        names = raw.keys()
        count = sum(math.prod(raw.get_slice(name).get_shape()) for name in names)
    model, report = import_qwen_weights(
        source, num_hidden_layers=LAYERS, mlp_activation=activation,
        second_coord_mode=MODE, c_sat=WEDGE_SAT_SLOPE, c_w0=WEDGE_HALF_SAT,
        indiff_eps="auto", indiff_rel=0.25, device="cuda", dtype=torch.float32,
    )
    assert sorted(report.layer_reports) == list(range(LAYERS))
    assert all(layer.mlp.mlp_activation == activation for layer in model.model.layers)
    assert report.total_weights == count and report.total_imported_tensors == len(names)
    assert report.total_re_weights == count and report.total_im_weights == 0
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert len(model.d16_parameters()) == len(names)
    print(f"Saving imported checkpoint ({count:,} weights)", flush=True)
    save_stage(model, run, "step0", report=report)
    write_json(run / "import.json", {
        "source": str(source), "revision": source.name, "source_sha256": file_hash(weights),
        "layers": LAYERS, "source_scalars": count, "unique_tensors": len(names),
        "source_in_real_axis": report.total_re_weights, "tied_embeddings": True,
        "second_coord_mode": MODE, "c_sat": WEDGE_SAT_SLOPE, "c_w0": WEDGE_HALF_SAT,
        "sign": 1.0, "mlp_activation": activation, "dtype": "float32",
        "indiff_eps": "auto", "indiff_rel": 0.25, "seconds": time.monotonic() - start,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
    })
    print("step0 saved", flush=True)


def data_manifest(source, run):
    from transformers import AutoTokenizer

    dest = run / "data.json"
    if dest.exists():
        return read_json(dest)["examples"]
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    train_file = ROOT / "data" / "train_en_ru_pl.txt"
    train_text = train_file.read_text(encoding="utf-8")
    training = train_text.split("\n\n")
    examples = []
    hashes = {train_file.name: file_hash(train_file)}
    calibration = (ROOT / "data" / "val_en_ru_pl.txt").read_text(encoding="utf-8")
    for index, (lang, heldout_index) in enumerate((("en", 1), ("ru", 2), ("pl", 2))):
        val_file = ROOT / "data" / f"val_{lang}.txt"
        hashes[val_file.name] = file_hash(val_file)
        validation = val_file.read_text(encoding="utf-8").strip().split("\n\n")[heldout_index]
        for split, path, doc, content, length in (
            ("train", train_file, index, training[index], 32),
            ("validation", val_file, heldout_index, validation, 128),
        ):
            ids = tokenizer.encode(content, add_special_tokens=False)[:length]
            assert len(ids) == length
            if split == "validation":
                assert content[:128] not in train_text and content[:128] not in calibration
            examples.append(dict(split=split, language=lang, filename=path.name,
                                 paragraph_index=doc, ids=ids))
    write_json(dest, {"source_revision": source.name, "sha256": hashes, "examples": examples})
    return examples


@torch.no_grad()
def teacher_probabilities(source, examples):
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, dtype=torch.float32)
    teacher.to("cuda").eval()
    reference = []
    for ex in examples:
        ids = torch.tensor([ex["ids"]], device="cuda")
        logits = teacher(input_ids=ids, use_cache=False).logits[0, :-1]
        log_p = logits.log_softmax(dim=-1)
        nll = -log_p.gather(-1, ids[0, 1:, None]).sum(dtype=torch.float64).item()
        reference.append((log_p.exp().cpu(), nll))
    return reference


def reduce_metrics(rows):
    n = sum(row["tokens"] for row in rows)
    def total(key):
        return sum(row[key] for row in rows)
    zero = total("zero_probability_targets")
    nll = math.inf if zero else total("finite_nll_sum") / n
    teacher_nll = total("qwen_nll_sum") / n
    return {
        "tokens": n, "loss": total("loss_sum") / n, "sampling_nll": "inf" if zero else nll,
        "sampling_perplexity": "inf" if zero else math.exp(nll),
        "zero_probability_targets": zero, "cutoff_targets": total("cutoff_targets"),
        "nll_without_readout_cutoff": total("no_cutoff_nll_sum") / n,
        "perplexity_without_readout_cutoff": math.exp(total("no_cutoff_nll_sum") / n),
        "top1_correct": total("top1_correct"), "top1_accuracy": total("top1_correct") / n,
        "top5_correct": total("top5_correct"), "top10_correct": total("top10_correct"),
        "top1_matches_qwen": total("top1_matches_qwen"), "qwen_top1_correct": total("qwen_top1_correct"),
        "qwen_top5_correct": total("qwen_top5_correct"), "qwen_top10_correct": total("qwen_top10_correct"),
        "qwen_nll": teacher_nll, "qwen_perplexity": math.exp(teacher_nll),
        "js_divergence_to_qwen_nats": total("js_sum") / n,
    }


@torch.no_grad()
def evaluate(source, run, stage):
    from transformers import AutoTokenizer

    examples = data_manifest(source, run)
    reference = teacher_probabilities(source, examples)
    gc.collect()
    torch.cuda.empty_cache()
    model = load_d16_checkpoint(run / stage, device="cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    eps, rel = model.config.indiff_eps, model.config.indiff_rel
    rows = []
    predictions = []
    for ex, (qwen_probs, teacher_nll) in zip(examples, reference, strict=True):
        ids = torch.tensor([ex["ids"]], device="cuda")
        output = model(encode_token_ids(ids), use_cache=False)
        n = ids.shape[1] - 1
        row = dict(language=ex["language"], split=ex["split"], tokens=n, loss_sum=0.,
                   no_cutoff_nll_sum=0., finite_nll_sum=0., zero_probability_targets=0,
                   cutoff_targets=0, top1_correct=0, top1_matches_qwen=0,
                   qwen_top1_correct=0, top5_correct=0, top10_correct=0,
                   qwen_top5_correct=0, qwen_top10_correct=0, js_sum=0., qwen_nll_sum=teacher_nll)
        for start in range(0, n, 8):
            stop = min(start + 8, n)
            z = output.logits[0, start:stop]
            target = ids[0, start + 1:stop + 1]
            size = target.numel()
            row["loss_sum"] += z.wedge_cross_entropy(target, eps=eps, rel=rel).unpack()[0].item() * size
            row["no_cutoff_nll_sum"] += z.wedge_cross_entropy(target, eps=0.).unpack()[0].item() * size
            p = z.sampling_probabilities(1., eps=eps, rel=rel).unpack()[0]
            if not torch.allclose(p.sum(dim=-1), torch.ones_like(target, dtype=p.dtype), atol=2e-5):
                raise ValueError("Invalid D16 sampling distribution")
            p_target = p.gather(-1, target[:, None]).flatten().double()
            row["zero_probability_targets"] += (p_target == 0).sum().item()
            row["finite_nll_sum"] -= p_target[p_target > 0].log().sum().item()
            strength = z.wedge_strength(eps=eps, rel=rel)
            cut = (strength.amax(-1) > 0) & (strength.gather(-1, target[:, None]).flatten() == 0)
            row["cutoff_targets"] += cut.sum().item()
            top = z.wedge_argmax(eps=eps, rel=rel).flatten()
            q = qwen_probs[start:stop].to(p.device)
            qtop = q.argmax(-1)
            row["top1_correct"] += (top == target).sum().item()
            row["top1_matches_qwen"] += (top == qtop).sum().item()
            row["qwen_top1_correct"] += (qtop == target).sum().item()
            for probs, prefix in ((p, ""), (q, "qwen_")):
                top10 = probs.topk(10, dim=-1).indices
                hit = top10 == target[:, None]
                row[prefix + "top5_correct"] += hit[:, :5].any(-1).sum().item()
                row[prefix + "top10_correct"] += hit.any(-1).sum().item()
            q_target = q.gather(-1, target[:, None]).flatten()
            for j, (d16_id, qwen_id, prob, qprob) in enumerate(zip(
                    top.tolist(), qtop.tolist(), p_target.tolist(), q_target.tolist(), strict=True)):
                pos = start + j + 1
                actual = ex["ids"][pos]
                predictions.append({
                    "split": ex["split"], "language": ex["language"], "position": pos,
                    "context": tokenizer.decode(ex["ids"][max(0, pos - 12):pos]),
                    "target_id": actual, "target": tokenizer.decode([actual]),
                    "d16_top1_id": d16_id, "d16_top1": tokenizer.decode([d16_id]),
                    "qwen_top1_id": qwen_id, "qwen_top1": tokenizer.decode([qwen_id]),
                    "d16_target_probability": prob, "qwen_target_probability": qprob,
                })
            mixture = (p + q) * .5
            js = torch.xlogy(p, p) - torch.xlogy(p, mixture) + torch.xlogy(q, q) - torch.xlogy(q, mixture)
            row["js_sum"] += js.sum(dtype=torch.float64).item() * .5
        rows.append(row)
        del output
        print(stage, ex["split"], ex["language"], reduce_metrics([row]), flush=True)
    metrics = {split: reduce_metrics([r for r in rows if r["split"] == split])
               for split in ("train", "validation")}
    write_json(run / f"predictions_{stage}.json", predictions)
    write_json(run / f"metrics_{stage}.json", {"temperature": 1., **metrics, "per_text": rows})


def train_once(source, run):
    if any((run / name).exists() for name in ("update_started.json", "step1", "step1.pending")):
        raise FileExistsError("This run already started its single optimizer update.")
    before = read_json(run / "metrics_step0.json")
    training = [e for e in data_manifest(source, run) if e["split"] == "train"]
    model = load_d16_checkpoint(run / "step0", device="cuda").train()
    model.gradient_checkpointing_enable()
    info = read_json(run / "import.json")
    assert len(model.model.layers) == LAYERS
    assert all(layer.mlp.mlp_activation == info["mlp_activation"] for layer in model.model.layers)
    assert all(p.requires_grad for p in model.parameters())
    assert len(model.d16_parameters()) == info["unique_tensors"]

    # Keep all layers trainable on a 12 GB GPU. Hooks only change the location
    # of accumulated gradients. Restore each one to its matching leaf before AdamW.
    host_layers = list(model.model.layers[:16])
    host_grads = {}
    def offload(name, param):
        grad = param.grad.detach().cpu()
        if name in host_grads:
            host_grads[name].add_(grad)
        else:
            host_grads[name] = grad
        param.grad = None

    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    targets = sum(len(e["ids"]) - 1 for e in training)
    measured_loss = 0.
    with ExitStack() as stack:
        for index, layer in enumerate(host_layers):
            for name, param in layer.named_parameters():
                hook = param.register_post_accumulate_grad_hook(partial(offload, (index, name)))
                stack.callback(hook.remove)
        for layer in model.model.layers:
            recompute = partial(checkpoint, layer.forward, use_reentrant=False, preserve_rng_state=False)
            stack.enter_context(patch.object(layer, "forward", recompute))
        for ex in training:
            inputs = encode_token_ids(torch.tensor([ex["ids"]], device="cuda"))
            output = model(inputs, labels=inputs, use_cache=False)
            fraction = (len(ex["ids"]) - 1) / targets
            loss = output.loss.unpack()[0] * fraction
            measured_loss += loss.detach().item()
            loss.backward()
            del output, loss
            print("Backward completed", ex["language"], flush=True)
    for index, layer in enumerate(host_layers):
        layer.to(device="cpu")
        for name, param in layer.named_parameters():
            param.grad = host_grads.pop((index, name))
    assert not host_grads
    gc.collect()
    torch.cuda.empty_cache()
    parameters = model.d16_parameters()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    if missing or not all(p.has_grad() for p in parameters):
        raise RuntimeError(f"Missing gradients before the only update: {missing}")
    if not math.isclose(measured_loss, before["train"]["loss"], rel_tol=2e-5, abs_tol=2e-5):
        raise RuntimeError(f"Training inputs/weights differ from baseline: {measured_loss}")
    options = dict(lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.,
                   indiff_eps=0., tile_size=1 << 19)
    optimizer = D16AdamW(parameters, **options)
    metadata = dict(optimizer="D16AdamW", options=options, max_grad_norm=CLIP, mlp_activation=info["mlp_activation"],
                    layers=LAYERS, unique_d16_tensors=len(parameters), frozen_tensors=0,
                    tensors_with_gradients=len(parameters), train_targets=targets,
                    microbatches=len(training), train_loss_before_backward=measured_loss,
                    gradient_offload_decoder_layers=len(host_layers))
    with (run / "update_started.json").open("x", encoding="utf-8") as guard:
        json.dump(metadata, guard, indent=2)
    print(f"One D16AdamW update, all {len(parameters)} tensors", flush=True)
    grad_norm = optimizer.step(max_grad_norm=CLIP).item()
    steps = [m["step"] if m is not None else 0 for m in optimizer.state_dict()["moments"]]
    assert steps == [1] * len(parameters)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    save_stage(model, run, "step1")
    write_json(run / "update.json", {
        **metadata, "completed_optimizer_steps": 1, "unique_tensors_at_step_one": len(steps),
        "global_grad_norm_before_clip": grad_norm, "clip_factor": min(1., CLIP / (grad_norm + 1e-6)),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "seconds_including_save": time.monotonic() - start,
    })
    print("step1 saved", flush=True)


@torch.no_grad()
def verify_weights(source, run):
    """Compare source coordinates and the completed update element by element."""
    from torch.utils.serialization import config as serialization

    info = read_json(run / "import.json")
    with patch.object(serialization.load, "mmap", True):
        after = load_d16_checkpoint(run / "step1", device="cuda").eval()
        before = load_d16_checkpoint(run / "step0", device="cpu").eval()
    head_dim = before.config.head_dim
    tensors = {}
    with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as raw:
        sigmas = {}
        for key in raw.keys():
            source_tensor = raw.get_tensor(key)
            sum_sq = torch.zeros((), device="cuda", dtype=torch.float64)
            for part in source_tensor.reshape(-1).split(1 << 20):
                sum_sq += part.to("cuda", torch.float32).square().sum(dtype=torch.float64)
            sigmas[key] = (sum_sq / source_tensor.numel()).sqrt().float().clamp_min(1e-8)
        for key in raw.keys():
            source_tensor = raw.get_tensor(key)
            gain = key.endswith("norm.weight")
            parameter_name = key.removesuffix("weight") + "gain" if gain else key
            old = before.get_submodule(parameter_name).value()
            new = after.get_submodule(parameter_name).value()
            assert old.shape == new.shape == source_tensor.shape
            real_only = gain or key.endswith("v_proj.bias")
            is_qk = ".q_proj." in key or ".k_proj." in key
            cross_key = None
            if ".up_proj." in key:
                cross_key = key.replace(".up_proj.", ".gate_proj.")
            elif ".gate_proj." in key:
                cross_key = key.replace(".gate_proj.", ".up_proj.")
            cross = raw.get_tensor(cross_key) if cross_key else None
            sigma = sigmas[key]
            partner_sigma = sigmas[cross_key] if cross_key else sigma
            row_size = max(1, (1 << 19) // math.prod(old.shape[1:]))
            stats = dict(elements=source_tensor.numel(), changed=0, state_changed=0,
                         nonzero_second_coordinate=0, mismatched_elements=0, nonfinite_after=0,
                         max_source_error=0., max_formula_error=0., norm_sq=0., delta_sq=0.)
            for start in range(0, old.shape[0], row_size):
                stop = min(start + row_size, old.shape[0])
                rows = slice(start, stop)
                original = source_tensor[rows].to("cuda", torch.float32)
                if real_only:
                    expected = torch.zeros_like(original)
                else:
                    if is_qk:
                        indices = torch.arange(start, stop)
                        other = (indices // head_dim) * head_dim + (indices + head_dim // 2) % head_dim
                        sign = torch.where(indices % head_dim < head_dim // 2, 1., -1.).to("cuda")
                        partner = source_tensor[other].to("cuda", torch.float32)
                        partner = partner * (sign if original.ndim == 1 else sign[:, None])
                    elif cross is not None:
                        partner = cross[rows].to("cuda", torch.float32)
                    else:
                        partner = torch.empty_like(original)
                        partner[..., 0::2] = -original[..., 1::2]
                        partner[..., 1::2] = original[..., 0::2]
                    amplitude = info["c_sat"] * sigma * (original.abs() / (original.abs() + info["c_w0"] * sigma))
                    expected = amplitude * (partner / partner_sigma).clamp(-3., 3.)
                base = old[rows].to("cuda")
                trained = new[rows]
                r0, i0 = base.unpack()
                r1, i1 = trained.unpack()
                stats["max_source_error"] = max(stats["max_source_error"], (r0 - original).abs().max().item())
                stats["max_formula_error"] = max(stats["max_formula_error"], (i0 - expected).abs().max().item())
                matches = torch.isclose(r0, original, rtol=2e-5, atol=1e-6) & torch.isclose(i0, expected, rtol=2e-5, atol=1e-6)
                stats["mismatched_elements"] += (~matches | base.is_nan()).sum().item()
                stats["nonfinite_after"] += (trained.is_nan() | ~torch.isfinite(r1) | ~torch.isfinite(i1)).sum().item()
                stats["changed"] += ((r0 != r1) | (i0 != i1)).sum().item()
                stats["state_changed"] += (base.state() != trained.state()).sum().item()
                stats["nonzero_second_coordinate"] += (i0 != 0.).sum().item()
                stats["norm_sq"] += base.norm_squared().sum(dtype=torch.float64).item()
                stats["delta_sq"] += ((r1.double() - r0).square() + (i1.double() - i0).square()).sum().item()
            stats["real_only_by_design"] = real_only
            tensors[key] = stats
    groups = {}
    for key, stats in tensors.items():
        group = "layer_" + key.split(".")[2] if key.startswith("model.layers.") else key
        groups.setdefault(group, []).append(stats)
    def summarize(records):
        summed = {k: sum(r[k] for r in records) for k in (
            "elements", "changed", "state_changed", "mismatched_elements", "nonfinite_after",
            "nonzero_second_coordinate", "norm_sq", "delta_sq")}
        summed["relative_l2_change"] = math.sqrt(summed["delta_sq"] / summed["norm_sq"])
        summed["tensors_with_change"] = sum(r["changed"] > 0 for r in records)
        summed["max_source_error"] = max(r["max_source_error"] for r in records)
        summed["max_formula_error"] = max(r["max_formula_error"] for r in records)
        return summed
    all_weights = summarize(list(tensors.values()))
    write_json(run / "weights.json", {
        "tolerance": {"atol": 1e-6, "rtol": 2e-5}, "total": all_weights,
        "per_group": {k: summarize(v) for k, v in groups.items()}, "per_tensor": tensors,
    })
    assert all_weights["elements"] == info["source_scalars"]
    assert not all_weights["mismatched_elements"] and not all_weights["nonfinite_after"]
    print("Weight verification", all_weights, flush=True)


def compare_activations(run, reference, current_metrics):
    if read_json(run / "data.json") != read_json(reference / "data.json"):
        raise ValueError("Token sequences or source data differ between runs.")
    configurations = [read_json(p / "step0" / "d16_config.json") for p in (reference, run)]
    modes = [cfg.pop("mlp_activation") for cfg in configurations]
    if configurations[0] != configurations[1]:
        raise ValueError("Model settings other than the MLP activation differ between runs.")
    for filename, keys in (
        ("import.json", ("source_sha256", "second_coord_mode", "c_sat", "c_w0", "sign", "unique_tensors", "source_scalars")),
        ("update.json", ("options", "max_grad_norm", "frozen_tensors", "train_targets", "completed_optimizer_steps")),
    ):
        reference_meta, current_meta = (read_json(p / filename) for p in (reference, run))
        for key in keys:
            if reference_meta[key] != current_meta[key]:
                raise ValueError(f"Incompatible experiments: {key} differs.")
    previous = [read_json(reference / f"metrics_step{i}.json") for i in (0, 1)]
    lines = ["", f"Сравнение с {reference.name}"]
    for mode, directory in zip(modes, (reference, run), strict=True):
        digest = file_hash(directory / "step0" / "d16_model.pt")
        lines.append(f"SHA-256 файла весов после импорта, {mode:<8}  {digest}")
    lines.append("Режим       Шаг   Выборка         Loss    Top-1  Top-5  Top-10  Цели с P=0    PPL")
    for split in ("validation", "train"):
        for mode, stages in zip(modes, (previous, current_metrics), strict=True):
            for step, stage in enumerate(stages):
                m = stage[split]
                lines.append(f"{mode:<11} {step:>2}   {split:<10}  {m['loss']:8.4f}  {m['top1_correct']:5}  "
                             f"{m['top5_correct']:5}  {m['top10_correct']:6}  {m['zero_probability_targets']:10}    {m['sampling_perplexity']}")
    return lines


def report_results(run, reference=None):
    data = [read_json(run / f"metrics_step{i}.json") for i in (0, 1)]
    pred = [read_json(run / f"predictions_step{i}.json") for i in (0, 1)]
    update = read_json(run / "update.json")
    weights = read_json(run / "weights.json")
    imp = read_json(run / "import.json")
    lines = [
        f"Qwen2.5-0.5B → D16 model2 | {imp['mlp_activation']} ({MLP_OPERATIONS[imp['mlp_activation']]}) | один шаг по всей сети", "",
        f"Ревизия Qwen   {imp['revision']}", f"SHA-256 весов  {imp['source_sha256']}",
        f"Импорт         {imp['source_scalars']:,} весов D16, {imp['unique_tensors']} тензоров, {imp['layers']} слоя",
        f"Формула        {imp['second_coord_mode']}, c_sat={imp['c_sat']:.10f}, c_w0={imp['c_w0']:.10f}, sign=+1",
        f"Оптимизатор    D16AdamW, lr={update['options']['lr']}, decay=0, норма до клиппинга {update['global_grad_norm_before_clip']:.6f}",
        f"Обновлений     {update['completed_optimizer_steps']}, счётчик шага равен 1 у {update['unique_tensors_at_step_one']} тензоров",
        "Данные         Один обучающий батч из трёх текстов, 93 целевых токена. Валидация EN/RU/PL по 127 позиций.",
        "Оценка         Предсказание следующего токена с исходным контекстом, температура 1, indiff_eps=auto, indiff_rel=0.25",
    ]
    if reference is not None:
        lines.extend(compare_activations(run, reference, data))
    for split in ("validation", "train"):
        a, b = (s[split] for s in data)
        n = a["tokens"]
        lines += ["", f"{split}  ({n} позиций)", "Показатель                                   D16 до          D16 после        Qwen",
                  f"Loss D16 (wedge_cross_entropy)           {a['loss']:10.6f}         {b['loss']:10.6f}",
                  f"Генеративная NLL                         {str(a['sampling_nll']):>10}         {str(b['sampling_nll']):>10}     {a['qwen_nll']:9.6f}",
                  f"Генеративная perplexity                  {str(a['sampling_perplexity']):>10}         {str(b['sampling_perplexity']):>10}     {a['qwen_perplexity']:9.4f}",
                  f"Правильные токены с нулевой вероятностью {a['zero_probability_targets']:10d}         {b['zero_probability_targets']:10d}",
                  f"NLL без выходной отсечки                 {a['nll_without_readout_cutoff']:10.6f}         {b['nll_without_readout_cutoff']:10.6f}",
                  f"JS до распределения Qwen, нат            {a['js_divergence_to_qwen_nats']:10.6f}         {b['js_divergence_to_qwen_nats']:10.6f}"]
        for k in (1, 5, 10):
            key = f"top{k}_correct"
            lines.append(f"Правильный токен в top-{k:<2}               {a[key]:12d}         {b[key]:10d}     {a['qwen_'+key]:9d}")
        pairs = [(old, new) for old, new in zip(*pred, strict=True) if old["split"] == split]
        for old, new in pairs:
            assert all(old[k] == new[k] for k in ("split", "language", "position", "target_id", "qwen_top1_id"))
        changed = sum(old["d16_top1_id"] != new["d16_top1_id"] for old, new in pairs)
        improved = sum(old["d16_target_probability"] < new["d16_target_probability"] for old, new in pairs)
        lines += [f"Предсказание top-1 изменилось на {changed}/{n} позициях. Вероятность правильного токена выросла на {improved}/{n}.",
                  f"Совпадение top-1 с Qwen до и после шага    {a['top1_matches_qwen']}/{n} и {b['top1_matches_qwen']}/{n}."]
        if split == "validation":
            lines.append("Примеры отдельных токенов")
            for old, new in pairs:
                if old["position"] == {"en": 33, "ru": 9, "pl": 4}[old["language"]]:
                    lines.append(f"  {old['language']} {old['context']!r} → цель {old['target']!r}, Qwen {old['qwen_top1']!r}, D16 {old['d16_top1']!r} → {new['d16_top1']!r}")
    overall = weights["total"]
    lines += ["", "Проверка весов", f"Проверены {overall['elements']:,} объектов. Расхождений с формулой вне допуска {overall['mismatched_elements']}.",
              f"Изменилось {overall['changed']:,} весов в {overall['tensors_with_change']} тензорах. Относительный сдвиг L2 {overall['relative_l2_change']:.8%}.",
              "Слой или группа             Всего весов     Изменились     Относительный сдвиг L2"]
    groups = weights["per_group"]
    for name in ["model.embed_tokens.weight", *(f"layer_{n}" for n in range(LAYERS)), "model.norm.weight"]:
        g = groups[name]
        lines.append(f"{name:<27} {g['elements']:12,d} {g['changed']:14,d}       {g['relative_l2_change']:.8%}")
    (run / "comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["import", "before", "step", "after", "verify", "report"])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--activation", choices=list(MLP_OPERATIONS),
                        help="MLP mode for the import; other phases use the saved mode")
    parser.add_argument("--reference-run", type=Path, help="completed one-step run to compare in the report")
    args = parser.parse_args()
    if args.reference_run is not None and args.phase != "report":
        parser.error("--reference-run applies to the report phase")
    args.run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(torch.get_num_threads(), 6))
    torch.manual_seed(20261125)
    source, run = args.source.resolve(), args.run.resolve()
    if args.phase != "import":
        info = read_json(run / "import.json")
        if file_hash(source / "model.safetensors") != info["source_sha256"]:
            raise ValueError("Source checkpoint differs from the one used for the import.")
        if args.activation and args.activation != info["mlp_activation"]:
            parser.error("The specified activation differs from the saved import.")
    if args.phase == "import":
        prepare(source, run, args.activation or "bilinear")
    elif args.phase == "step":
        train_once(source, run)
    elif args.phase == "verify":
        verify_weights(source, run)
    elif args.phase == "report":
        report_results(run, args.reference_run)
    else:
        evaluate(source, run, {"before": "step0", "after": "step1"}[args.phase])
