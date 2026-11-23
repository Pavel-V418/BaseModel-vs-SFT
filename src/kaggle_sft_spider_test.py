"""Run the two expanded SFT adapters with the team's Spider 1.0 test evaluator."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import sqlite3
import time
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

from src.data import db_path
from src.evaluate import (
    BATCH_SIZE, EVALUATION_VERSION, MAX_NEW_TOKENS, QUERY_TIMEOUT, _generate, _read_jsonl, _saved_generations,
    evaluate_records, metric_summary,
)
from src.spider_test import ADAPTERS, CHUNK, REVISIONS, SPIDER_SHA, TEST_SHA, _write_json, _write_jsonl, prepare, sha256


SOURCE_COMMIT = "1bf6018f71a58551ca6f66077c7a1839a68f1f88"
RUN_SUFFIX = "_nfull_others_r16_lr1e-4_s42_2ep"
SMOKE_SIZE = 24
MODELS = {
    "0.5b": ("Qwen/Qwen2.5-0.5B", REVISIONS["Qwen/Qwen2.5-0.5B"], *ADAPTERS["expanded"]),
    "1.5b": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323",
             "ecfe74ab5af060c73f4e7baa5ae7937139805c79099c4bfda64c516aac1e66f1",
             "22274ed0c15d8ac73f1b4a03566579b63d3cc1da315fc1c09c321f95767e9517"),
}
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
GENERATION = {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS,
              "batch_size": BATCH_SIZE, "padding_side": "left", "seed": 42, "chunk_size": CHUNK}
REFERENCE = Path(__file__).resolve().parents[1] / "results/spider_test/sft_0.5b_expanded_zero"


def prepare_test(archive: Path, folder: Path) -> list[dict]:
    records, _ = prepare(archive, folder)
    databases = {row["db_id"] for row in records}
    for db_id in sorted(databases):
        uri = f"{db_path(db_id, folder).resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError(f"SQLite quick_check failed: {db_id}")
    print(f"Verified {len(records)} test questions and {len(databases)} SQLite databases", flush=True)
    return records


def find_adapters(input_root: Path, work: Path) -> dict[str, Path]:
    adapters = {}
    for size, (model_id, _, config_sha, weights_sha) in MODELS.items():
        run = f"sft_{size}{RUN_SUFFIX}"
        zips = sorted(input_root.rglob(f"results_{run}.zip"))
        folders = sorted(path / "final" for path in input_root.rglob(run)
                         if (path / "final" / ADAPTER_FILES[0]).is_file())
        if len(zips) + len(folders) != 1:
            raise ValueError(f"Expected one results_{run}.zip or one extracted {run}/final under {input_root}; "
                             f"found {len(zips)} ZIP(s) and {len(folders)} folder(s)")
        if folders:
            adapter = folders[0]
        else:
            adapter = work / "adapters" / run
            adapter.mkdir(parents=True, exist_ok=True)
            suffix = f"outputs/{run}/final/{ADAPTER_FILES[0]}"
            with ZipFile(zips[0]) as archive:
                matches = [name for name in archive.namelist() if name == suffix or name.endswith(f"/{suffix}")]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {suffix} in {zips[0]}")
                parent = matches[0].rsplit("/", 1)[0]
                for filename in ADAPTER_FILES:
                    with archive.open(f"{parent}/{filename}") as source, (adapter / filename).open("wb") as target:
                        shutil.copyfileobj(source, target)
        for filename, expected_sha in zip(ADAPTER_FILES, (config_sha, weights_sha)):
            if sha256(adapter / filename) != expected_sha:
                raise ValueError(f"Unexpected SHA-256 for {adapter / filename}")
        config = json.loads((adapter / ADAPTER_FILES[0]).read_text(encoding="utf-8"))
        expected = {"base_model_name_or_path": model_id, "peft_type": "LORA", "task_type": "CAUSAL_LM",
                    "r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"LoRA configuration for {run} does not match {expected}")
        adapters[size] = adapter
        print(f"Verified {model_id} LoRA configuration and weights", flush=True)
    return adapters


def score(records: list[dict], generations: list[tuple[str, bool]], folder: Path) -> list[dict]:
    rows = evaluate_records(records, generations, database_resolver=lambda db: db_path(db, folder))
    for record, row in zip(records, rows):
        row["db_id"] = record["db_id"]
        if not row["pred_sql"]:
            row["error_category"] = "empty_sql"
        elif not row["exec_ok"]:
            row["error_category"] = "sqlite_error_or_timeout"
        elif not row["exec_match"]:
            row["error_category"] = "result_mismatch"
        else:
            row["error_category"] = None
    return rows


def compare_reference(records: list[dict], rows: list[dict]) -> dict:
    published = _read_jsonl(REFERENCE.with_name(REFERENCE.name + "_eval.jsonl"))[:len(records)]
    if len(rows) != len(records) or len(published) != len(records):
        raise ValueError("Published predictions and evaluation records have different lengths")
    changed = {"raw_output": 0, "pred_sql": 0, "pred_sql_with_same_raw_output": 0,
               "exec_match": 0, "exec_match_with_same_sql": 0,
               "exec_ok": 0, "exact_match": 0, "stopped": 0}
    differing_ids = []
    for record, row, reference in zip(records, rows, published):
        if any(reference[key] != record[key] or row[key] != record[key] for key in ("id", "question", "gold_sql")):
            raise ValueError(f"Published prediction alignment failed at {record['id']}")
        for key in ("raw_output", "pred_sql", "exec_match", "exec_ok", "exact_match", "stopped"):
            changed[key] += row[key] != reference[key]
        if row["pred_sql"] != reference["pred_sql"] and row["raw_output"] == reference["raw_output"]:
            changed["pred_sql_with_same_raw_output"] += 1
        if row["exec_match"] != reference["exec_match"]:
            differing_ids.append(record["id"])
            changed["exec_match_with_same_sql"] += row["pred_sql"] == reference["pred_sql"]
    return {"published_correct": sum(row["exec_match"] for row in published),
            "published_execution_accuracy": metric_summary(published)["overall"]["execution_accuracy"],
            "changed_rows": changed, "execution_match_difference_ids": differing_ids}


def load_checkpoint(path: Path, records: list[dict]) -> tuple[list[dict], float, int]:
    state = json.loads(path.read_text(encoding="utf-8"))
    rows = state["rows"]
    if len(rows) != len(records) or any(
        any(row[key] != record[key] for key in ("id", "db_id", "question", "gold_sql"))
        for row, record in zip(rows, records)
    ):
        raise ValueError(f"Checkpoint dataset mismatch: {path}")
    return rows, state["inference_seconds"], state["peak_vram_bytes"]


def run_model(size: str, adapter: Path, records: list[dict], folder: Path, output: Path,
              phase: str, resume_from: Path | None, dataset_sha: str) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from src.utils import compute_dtype, load_model, load_tokenizer, set_seed

    model_id, revision, config_sha, adapter_sha = MODELS[size]
    metadata = {"model": model_id, "model_revision": revision, "adapter_sha256": adapter_sha,
                "adapter_config_sha256": config_sha, "dataset_split": "Spider 1.0 test", "phase": phase,
                "num_examples": len(records), "dataset_sha256": dataset_sha, "full_test_sha256": TEST_SHA,
                "spider_archive_sha256": SPIDER_SHA, "source_commit": SOURCE_COMMIT, "evaluation_version": EVALUATION_VERSION,
                "runner_sha256": sha256(Path(__file__)),
                "mode": "zero", "chat_template": False, "tokenizer_source": "base_snapshot",
                "dtype": str(compute_dtype()), "sqlite_version": sqlite3.sqlite_version,
                "gpu": torch.cuda.get_device_name(0), "generation": GENERATION,
                "query_timeout_seconds": QUERY_TIMEOUT,
                "versions": {pkg: version(pkg) for pkg in ("torch", "transformers", "peft", "accelerate")}}
    checkpoints = Path("checkpoints") / f"sft_{size}"
    local = output / checkpoints
    imported = resume_from / checkpoints if resume_from else None
    for source in (local, imported):
        if source is not None and source.exists():
            spec = source / "spec.json"
            if not spec.is_file() or json.loads(spec.read_text(encoding="utf-8")) != metadata:
                raise ValueError(f"Checkpoint runtime, model, or data differs: {source}")
    local.mkdir(parents=True, exist_ok=True)
    _write_json(local / "spec.json", metadata)

    model = tokenizer = None
    combined = []
    elapsed = 0.0
    peak_vram = 0
    try:
        for start in range(0, len(records), CHUNK):
            batch = records[start:start + CHUNK]
            filename = f"{start:05d}.json"
            chunk_path = local / filename
            if chunk_path.is_file():
                rows, seconds, peak = load_checkpoint(chunk_path, batch)
            elif imported is not None and (imported / filename).is_file():
                rows, seconds, peak = load_checkpoint(imported / filename, batch)
                _write_json(chunk_path, {"rows": rows, "inference_seconds": seconds, "peak_vram_bytes": peak})
            else:
                if model is None:
                    model_path = snapshot_download(model_id, revision=revision,
                                                   allow_patterns=["*.json", "*.safetensors", "vocab.*", "merges.*", "*.jinja"])
                    set_seed(42)
                    torch.cuda.reset_peak_memory_stats()
                    tokenizer = load_tokenizer(model_path)
                    model = load_model(model_path, adapter=str(adapter))
                    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
                        raise RuntimeError(f"{model_id} + LoRA did not fit entirely on the selected GPU")
                    print(f"{model_id} + LoRA loaded; CUDA allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
                torch.cuda.synchronize()
                began = time.perf_counter()
                generations = _generate(model, tokenizer, [row["prompt"] for row in batch])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - began
                peak = torch.cuda.max_memory_reserved()
                rows = score(batch, generations, folder)
                _write_json(chunk_path, {"rows": rows, "inference_seconds": seconds, "peak_vram_bytes": peak})
            combined.extend(rows)
            elapsed += seconds
            peak_vram = max(peak_vram, peak)
            print(f"{model_id}: saved {len(combined)}/{len(records)}", flush=True)
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    predictions = output / f"sft_{size}_predictions.jsonl"
    temporary = predictions.with_suffix(".tmp")
    _write_jsonl(temporary, combined)
    temporary.replace(predictions)
    metrics = {**metadata, **metric_summary(combined), "predictions_sha256": sha256(predictions),
               "execution_correct": sum(row["exec_match"] for row in combined), "inference_seconds": elapsed,
               "seconds_per_example": elapsed / len(combined), "peak_vram_reserved_bytes": peak_vram}
    if size == "0.5b" and phase == "full":
        metrics["published_reference"] = compare_reference(records, combined)
    _write_json(output / f"sft_{size}_metrics.json", metrics)
    return metrics


def write_comparison(output: Path, summaries: list[dict]) -> None:
    columns = ["model", "num_examples", "execution_correct", "execution_accuracy", "exact_match", "valid_sql_rate",
               "stop_rate", "easy", "medium", "hard", "extra", "inference_seconds", "seconds_per_example"]
    temporary = output / "comparison.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for summary in summaries:
            writer.writerow([summary["model"], summary["num_examples"], summary["execution_correct"],
                             *(summary["overall"][key] for key in columns[3:7]),
                             *(summary["by_difficulty"][key]["execution_accuracy"] for key in columns[7:11]),
                             summary["inference_seconds"], summary["seconds_per_example"]])
    temporary.replace(output / "comparison.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "audit", "smoke", "full"))
    parser.add_argument("--spider-archive", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume-from", type=Path, help="Previous full output folder containing checkpoints/")
    args = parser.parse_args()
    if args.phase in {"smoke", "full"} and (args.input_root is None or args.output_root is None):
        parser.error("GPU evaluation requires --input-root and --output-root")
    if args.resume_from is not None and (args.phase != "full" or not (args.resume_from / "checkpoints").is_dir()):
        parser.error("--resume-from must be a previous full output folder containing checkpoints/")

    folder = args.work_dir / "spider_test"
    records = prepare_test(args.spider_archive, folder)
    if args.phase == "prepare":
        return
    if args.phase == "audit":
        source = REFERENCE.with_name(REFERENCE.name + "_eval.jsonl")
        rows = score(records, _saved_generations(source, records), folder)
        print(json.dumps({**metric_summary(rows), **compare_reference(records, rows)}, indent=2), flush=True)
        return

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select a Kaggle GPU and expose one device (CUDA_VISIBLE_DEVICES=0)")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM free: {free/2**30:.2f}/{total/2**30:.2f} GiB", flush=True)
    adapters = find_adapters(args.input_root, args.work_dir)
    if args.phase == "smoke":
        records = records[:SMOKE_SIZE]
    output = args.output_root / args.phase
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "test.jsonl"
    _write_jsonl(dataset, records)
    dataset_sha = sha256(dataset)
    if args.phase == "full" and dataset_sha != TEST_SHA:
        raise ValueError("Prepared input differs from the published test dataset")
    summaries = [run_model(size, adapter, records, folder, output, args.phase, args.resume_from, dataset_sha)
                 for size, adapter in adapters.items()]
    write_comparison(output, summaries)
    for summary in summaries:
        print(f"{summary['model']}: {summary['overall']['execution_accuracy']:.2%} execution accuracy", flush=True)


if __name__ == "__main__":
    main()
