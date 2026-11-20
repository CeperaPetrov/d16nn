import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


SOURCES = [
    # (ключ, язык, repo, config, доля_в_итоговом_train)
    ("en_fineweb", "en", "HuggingFaceFW/fineweb-edu", "sample-10BT", 0.35 * 0.65),
    ("en_wiki",    "en", "wikimedia/wikipedia",       "20231101.en", 0.35 * 0.35),
    ("ru_fineweb", "ru", "HuggingFaceFW/fineweb-2",   "rus_Cyrl",    0.35 * 0.65),
    ("ru_wiki",    "ru", "wikimedia/wikipedia",       "20231101.ru", 0.35 * 0.35),
    ("pl_fineweb", "pl", "HuggingFaceFW/fineweb-2",   "pol_Latn",    0.30 * 0.65),
    ("pl_wiki",    "pl", "wikimedia/wikipedia",       "20231101.pl", 0.30 * 0.35),
]


def load_qwen_tokenizer():
    """Берёт локальный кеш Qwen2.5-0.5B, если он есть, иначе качает с HF."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = hf_home / "hub" / "models--Qwen--Qwen2.5-0.5B" / "snapshots"
    if hub_dir.is_dir():
        for snap in hub_dir.glob("*"):
            if (snap / "tokenizer_config.json").is_file():
                print(f"[tokenizer] Использую локальный кеш: {snap}")
                return AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    print("[tokenizer] Локальный снапшот не найден, загружаю Qwen/Qwen2.5-0.5B...")
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


def clean_text(text: str, title: str | None = None) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if title:
        title = title.strip()
        if title and not text.startswith(title):
            text = f"{title}\n{text}"
    return text


def trim_to_max_tokens(text: str, tokenizer, max_tokens: int) -> tuple[str, int]:
    """Обрезает слишком длинные статьи по абзацам, не рвя слова и предложения."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text, len(ids)

    paragraphs = text.split("\n\n")
    kept = []
    total = 0
    for p in paragraphs:
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        if total + len(p_ids) > max_tokens:
            if not kept:
                clipped = tokenizer.decode(p_ids[:max_tokens], skip_special_tokens=True).strip()
                return clipped, len(tokenizer.encode(clipped, add_special_tokens=False))
            break
        kept.append(p)
        total += len(p_ids) + 1  # запас на "\n\n"
    result = "\n\n".join(kept).strip()
    return result, len(tokenizer.encode(result, add_special_tokens=False))


def collect_from_source(
    key: str,
    repo: str,
    config: str,
    tokenizer,
    train_target: int,
    val_target: int,
    skip_docs: int = 0,
    min_chars: int = 200,
    max_doc_tokens: int = 2500,
):
    print(f"\n[{key}] Стрим {repo} ({config}) | skip={skip_docs} | target: val={val_target:,}, train={train_target:,} токенов")
    ds = load_dataset(repo, name=config, split="train", streaming=True)
    if skip_docs > 0:
        ds = ds.skip(skip_docs)

    val_docs, train_docs = [], []
    val_toks, train_toks = 0, 0
    raw_seen = skip_docs
    last_report = time.time()

    for row in ds:
        raw_seen += 1
        raw_text = row.get("text") or ""
        title = row.get("title") if "wiki" in key else None
        text = clean_text(raw_text, title=title)

        if len(text) < min_chars:
            continue

        text, n_toks = trim_to_max_tokens(text, tokenizer, max_doc_tokens)
        if n_toks < 40:
            continue

        # Сначала набираем валидацию (чтобы train и val строго не пересекались)
        if val_toks < val_target:
            val_docs.append((text, n_toks))
            val_toks += n_toks
        elif train_toks < train_target:
            train_docs.append((text, n_toks))
            train_toks += n_toks
        else:
            break

        if time.time() - last_report >= 5.0:
            pct = 100.0 * (val_toks + train_toks) / max(val_target + train_target, 1)
            print(f"  [{key}] {pct:5.1f}% | val={val_toks:,} | train={train_toks:,} | просмотрено строк={raw_seen:,}")
            last_report = time.time()

    print(
        f"  [{key}] ГОТОВО: val={val_toks:,} ток. ({len(val_docs)} док.) | "
        f"train={train_toks:,} ток. ({len(train_docs)} док.) | смещение потока={raw_seen:,}"
    )
    return {
        "val_docs": val_docs,
        "train_docs": train_docs,
        "val_tokens": val_toks,
        "train_tokens": train_toks,
        "next_skip_docs": raw_seen,
    }


def write_docs(path: Path, docs: list[str]) -> int:
    content = "\n\n".join(docs) + "\n"
    path.write_text(content, encoding="utf-8")
    return len(content.encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Сборка сбалансированного EN/RU/PL корпуса для D16-Qwen")
    parser.add_argument("--output-dir", type=str, default="data", help="Общая папка data в корне проекта")
    parser.add_argument("--train-name", type=str, default="train_en_ru_pl.txt", help="Имя выходного train-файла")
    parser.add_argument("--manifest-name", type=str, default="manifest_en_ru_pl.json", help="Имя manifest-файла")
    parser.add_argument("--train-tokens", type=int, default=6_000_000, help="Целевое число токенов в train-файле")
    parser.add_argument("--val-tokens-per-lang", type=int, default=4_096, help="Токенов в отдельных val_en/ru/pl.txt")
    parser.add_argument("--fast-val-per-lang", type=int, default=1_100, help="Токенов на язык в быстром val_en_ru_pl.txt")
    parser.add_argument("--seed", type=int, default=20261120, help="Seed перемешивания")
    parser.add_argument(
        "--skip-from-manifest",
        type=str,
        default=None,
        help="Путь к прошлому manifest-файлу, чтобы собрать СЛЕДУЮЩИЙ непересекающийся срез",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skip_map = {}
    if args.skip_from_manifest:
        prev = json.loads(Path(args.skip_from_manifest).read_text(encoding="utf-8"))
        skip_map = {k: int(v["next_skip_docs"]) for k, v in prev["sources"].items()}
        print(f"[resume-slice] Пропускаю уже использованные документы из {args.skip_from_manifest}: {skip_map}")

    tokenizer = load_qwen_tokenizer()
    rng = random.Random(args.seed)

    # Распределяем val_tokens_per_lang по FineWeb (65%) и Wiki (35%) внутри каждого языка
    val_split_ratio = {"fineweb": 0.65, "wiki": 0.35}

    all_train_docs = []
    val_by_lang = {"en": [], "ru": [], "pl": []}
    fast_val_docs = []
    manifest = {
        "seed": args.seed,
        "target_train_tokens": args.train_tokens,
        "sources": {},
        "files": {},
    }

    t0 = time.time()
    for key, lang, repo, config, train_share in SOURCES:
        src_kind = "fineweb" if "fineweb" in key else "wiki"
        t_target = int(round(args.train_tokens * train_share))
        v_target = int(round(args.val_tokens_per_lang * val_split_ratio[src_kind]))
        # Если собираем следующий срез поверх старого manifest — валидацию заново не вырезаем
        if args.skip_from_manifest:
            v_target = 0

        res = collect_from_source(
            key=key,
            repo=repo,
            config=config,
            tokenizer=tokenizer,
            train_target=t_target,
            val_target=v_target,
            skip_docs=skip_map.get(key, 0),
        )
        all_train_docs.extend([txt for txt, _ in res["train_docs"]])
        val_by_lang[lang].extend(res["val_docs"])

        manifest["sources"][key] = {
            "lang": lang,
            "repo": repo,
            "config": config,
            "train_tokens": res["train_tokens"],
            "train_docs": len(res["train_docs"]),
            "val_tokens": res["val_tokens"],
            "val_docs": len(res["val_docs"]),
            "next_skip_docs": res["next_skip_docs"],
        }

    # Перемешиваем train-документы, чтобы EN / RU / PL шли вперемешку в каждом батче
    rng.shuffle(all_train_docs)
    train_path = out_dir / args.train_name
    train_bytes = write_docs(train_path, all_train_docs)
    manifest["files"][args.train_name] = {
        "docs": len(all_train_docs),
        "bytes": train_bytes,
        "approx_tokens": sum(s["train_tokens"] for s in manifest["sources"].values()),
    }

    # Сохраняем валидационные файлы (если это первый срез)
    if not args.skip_from_manifest:
        for lang in ("en", "ru", "pl"):
            lang_items = val_by_lang[lang]
            rng.shuffle(lang_items)
            v_path = out_dir / f"val_{lang}.txt"
            v_bytes = write_docs(v_path, [txt for txt, _ in lang_items])
            manifest["files"][v_path.name] = {
                "docs": len(lang_items),
                "bytes": v_bytes,
                "approx_tokens": sum(nt for _, nt in lang_items),
            }

            # Берём компактный поднабор (~1100 токенов/язык) для быстрой валидации каждые 50 шагов
            acc = 0
            for txt, nt in lang_items:
                fast_val_docs.append(txt)
                acc += nt
                if acc >= args.fast_val_per_lang:
                    break

        rng.shuffle(fast_val_docs)
        fast_val_path = out_dir / "val_en_ru_pl.txt"
        fv_bytes = write_docs(fast_val_path, fast_val_docs)
        fv_tokens = len(tokenizer.encode(fast_val_path.read_text(encoding="utf-8"), add_special_tokens=False))
        manifest["files"]["val_en_ru_pl.txt"] = {
            "docs": len(fast_val_docs),
            "bytes": fv_bytes,
            "exact_tokens": fv_tokens,
        }

    manifest["elapsed_seconds"] = round(time.time() - t0, 1)
    manifest_path = out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"ГОТОВО за {manifest['elapsed_seconds']} с. Результаты в папке: {out_dir.resolve()}")
    for fname, info in manifest["files"].items():
        toks = info.get("exact_tokens", info.get("approx_tokens"))
        print(f"  {fname:<24} | {info['bytes']/1024/1024:6.2f} МБ | ~{toks:,} токенов | {info['docs']:,} док.")
    print("=" * 76)


if __name__ == "__main__":
    main()