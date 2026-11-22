"""Run the five project variants on the released Spider 1.0 test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from src.data import db_path, make_records
from src.evaluate import (
    BATCH_SIZE, EVALUATION_VERSION, MAX_NEW_TOKENS, QUERY_TIMEOUT, _fewshot_prefix, _generate,
    _query_rows, _read_jsonl, _saved_generations, evaluate_records, metric_summary,
)

SPIDER_SHA = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
TEST_SHA = "29d5638dbe53dd36b24ddef2384b7a365f2af65c085d044ab7b032f4c1bdcf06"
FEWSHOT_SHA = "25114622a92023c89514da93aa362db7a2372f16cf8ca61ee096792a283d321e"
ADAPTERS = {
    "original": (
        "8d90298f9f5aee2a228ac5b0f8a740817903f07a9731818664afc01d36a95447",
        "7f35391b415472a483583a0b813ceea2dd6ee730c27b90dba1352ed69cb5ca30",
    ),
    "expanded": (
        "18f75b76c500086799309ac4032eeb95f3be527314bc9d5d8d8a10f1d00788e9",
        "bd31c75f5d4e9adb470b9d3eed60fc935269b98982264a4363934c9976a7b9cd",
    ),
}
BASE = "Qwen/Qwen2.5-0.5B"
INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"
REVISIONS = {
    BASE: "060db6499f32faf8b98477b0a26969ef7d8b9987",
    INSTRUCT: "7ae557604adf67be50417f59c2c2f167def9a775",
}
RUNS = {
    "base_0.5b_zero": (BASE, None, "zero"),
    "sft_0.5b_zero": (BASE, "original", "zero"),
    "sft_0.5b_expanded_zero": (BASE, "expanded", "zero"),
    "base_0.5b_few": (BASE, None, "few"),
    "instruct_0.5b_zero": (INSTRUCT, None, "zero"),
}
CHUNK = 128  # A multiple of batch size so resume preserves the batches.


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict], *, newline: str = "\n") -> None:
    path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + newline for row in rows).encode("utf-8"))


def prepare(archive: Path, folder: Path) -> tuple[list[dict], Path]:
    """Prepare the official test without mixing dev examples into the test."""
    if sha256(archive) != SPIDER_SHA:
        raise ValueError("Unexpected Spider archive SHA-256")
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        def read(name: str):
            return json.loads(z.read(f"spider_data/{name}"))

        test, train = read("test.json"), read("train_spider.json")
        dbs = {row["db_id"] for row in test}
        excluded = {row["db_id"] for row in train + read("train_others.json") + read("dev.json")}
        if len(test) != 2147 or len(dbs) != 40 or dbs & excluded:
            raise ValueError("Unexpected test rows, databases, or train/dev overlap")
        (folder / "tables.json").write_bytes(z.read("spider_data/test_tables.json"))
        for db in sorted(dbs):
            database = db_path(db, folder)
            database.parent.mkdir(parents=True, exist_ok=True)
            with z.open(f"spider_data/test_database/{db}/{db}.sqlite") as source, database.open("wb") as target:
                shutil.copyfileobj(source, target)

    records, _ = make_records(test, "test", folder, check_exec=False)
    examples, _ = make_records([train[i] for i in (18, 2749, 1835, 2897)], "fewshot", folder, check_exec=False)
    test_path, fewshot = folder / "test.jsonl", folder / "fewshot.jsonl"
    _write_jsonl(test_path, records)
    _write_jsonl(fewshot, examples, newline="\r\n")
    if sha256(test_path) != TEST_SHA or sha256(fewshot) != FEWSHOT_SHA:
        raise ValueError("Prepared prompts differ from the expected records")
    for row in records:
        _query_rows(db_path(row["db_id"], folder), row["gold_sql"])
    print(f"Validated {len(records)} gold queries in {len(dbs)} databases; {dict(Counter(r['difficulty'] for r in records))}", flush=True)
    return records, fewshot


def load_adapter(bundle: Path, folder: Path, adapter: str) -> Path:
    output = folder / adapter
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / name for name in ("adapter_config.json", "adapter_model.safetensors")]
    expected = ADAPTERS[adapter]
    if all(path.is_file() and sha256(path) == digest for path, digest in zip(paths, expected)):
        return output
    deadline = time.monotonic() + 900
    while not zipfile.is_zipfile(bundle):
        if time.monotonic() >= deadline:
            raise FileNotFoundError(f"Upload the two adapters to {bundle}")
        time.sleep(3)
    with zipfile.ZipFile(bundle) as z:
        for path, digest in zip(paths, expected):
            data = z.read(f"{adapter}/{path.name}")
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"{adapter}/{path.name} SHA-256 mismatch")
            path.write_bytes(data)
    return output


def run(name, records, fewshot, folder, adapter_bundle, results, source_commit):
    import torch
    from huggingface_hub import snapshot_download
    from src.utils import load_model, load_tokenizer, set_seed

    model_id, adapter, mode = RUNS[name]
    model_path = snapshot_download(
        model_id, revision=REVISIONS[model_id],
        allow_patterns=["*.json", "*.safetensors", "vocab.*", "merges.*", "*.jinja"],
    )
    adapter_path = load_adapter(adapter_bundle, folder / "adapters", adapter) if adapter else None
    metadata = {
        "run_name": name, "model": model_id, "model_revision": REVISIONS[model_id],
        "adapter": adapter, "adapter_sha256": ADAPTERS[adapter][1] if adapter else None,
        "mode": mode, "fewshot_sha256": FEWSHOT_SHA if mode == "few" else None,
        "chat_template": model_id == INSTRUCT, "spider_archive_sha256": SPIDER_SHA,
        "dataset_split": "Spider 1.0 test", "dataset_sha256": TEST_SHA,
        "source_commit": source_commit, "evaluation_version": EVALUATION_VERSION, "num_examples": len(records),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {p: version(p) for p in ("torch", "transformers", "peft", "accelerate")},
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS,
                       "batch_size": BATCH_SIZE, "padding_side": "left", "seed": 42},
        "query_timeout_seconds": QUERY_TIMEOUT,
    }
    chunks = results / "checkpoints" / name
    chunks.mkdir(parents=True, exist_ok=True)
    spec = chunks / "run.json"
    if spec.is_file() and json.loads(spec.read_text(encoding="utf-8")) != metadata:
        raise ValueError(f"Checkpoint specifications differ for {name}. Choose a new output directory.")
    _write_json(spec, metadata)
    prefix = _fewshot_prefix(fewshot) if mode == "few" else ""
    model = tokenizer = None
    combined = []
    started = time.monotonic()
    for start in range(0, len(records), CHUNK):
        batch = records[start : start + CHUNK]
        checkpoint = chunks / f"{start:05d}.jsonl"
        if checkpoint.is_file():
            _saved_generations(checkpoint, batch)
            combined.extend(_read_jsonl(checkpoint))
            continue
        if model is None:
            set_seed(42)
            tokenizer = load_tokenizer(model_path)
            model = load_model(model_path, adapter=str(adapter_path) if adapter_path else None)
        prompts = [prefix + row["prompt"] for row in batch]
        if model_id == INSTRUCT:
            prompts = [
                tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                              tokenize=False, add_generation_prompt=True)
                for prompt in prompts
            ]
        generations = _generate(model, tokenizer, prompts)
        predicted = evaluate_records(batch, generations, database_resolver=lambda db: db_path(db, folder))
        temporary = checkpoint.with_suffix(".tmp")
        _write_jsonl(temporary, predicted)
        temporary.replace(checkpoint)
        combined.extend(predicted)
        print(f"{name}: saved {len(combined)}/{len(records)} after {time.monotonic()-started:.0f}s", flush=True)
    predictions = results / f"{name}_eval.jsonl"
    _write_jsonl(predictions, combined)
    metrics = {**metadata, "predictions_sha256": sha256(predictions), **metric_summary(combined)}
    _write_json(results / f"{name}_eval_metrics.json", metrics)
    print(f"{name}: execution accuracy {metrics['overall']['execution_accuracy']:.2%}", flush=True)
    with zipfile.ZipFile(results / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in (predictions, results / f"{name}_eval_metrics.json"):
            z.write(path, path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--adapters", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--variants", nargs="+", choices=RUNS, default=list(RUNS))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    records, fewshot = prepare(args.archive, args.data_dir)
    if args.prepare_only:
        return
    if not (args.adapters and args.results and args.source_commit):
        parser.error("evaluation requires --adapters, --results, and --source-commit")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Select a GPU runtime for inference")
    args.results.mkdir(parents=True, exist_ok=True)
    for name in args.variants:
        run(name, records, fewshot, args.data_dir, args.adapters, args.results, args.source_commit)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
