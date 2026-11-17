"""Deterministic SQL evaluation for base and adapter models.

Example:
    python -m src.evaluate --model Qwen/Qwen2.5-0.5B --mode zero --run_name base_0.5b_zero
    python -m src.evaluate --model Qwen/Qwen2.5-0.5B --adapter outputs/run/final --mode zero
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import Counter
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable

from src.data import clean_sql, db_path
from src.parse import extract_sql

ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 16
MAX_NEW_TOKENS = 256
QUERY_TIMEOUT = 5.0
DIFFICULTIES = ("easy", "medium", "hard", "extra")


def normalize_sql(sql: str) -> str:
    """Normalize case, whitespace, and terminal semicolons for exact match."""
    return re.sub(r"\s+", " ", clean_sql(sql)).strip().casefold()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        rounded = round(float(value), 6)
        return 0.0 if rounded == 0 else rounded
    return value


def _canonical_row(row: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(_canonical_value(value) for value in row)


def results_match(
    predicted: Iterable[Iterable[Any]],
    gold: Iterable[Iterable[Any]],
    *,
    ordered: bool,
) -> bool:
    """Compare rows while retaining duplicates; optionally preserve row order."""
    pred_rows = [_canonical_row(row) for row in predicted]
    gold_rows = [_canonical_row(row) for row in gold]
    if ordered:
        return pred_rows == gold_rows
    return Counter(pred_rows) == Counter(gold_rows)


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute execution accuracy, exact match, validity, and stop rates."""
    def rates(group: list[dict[str, Any]]) -> dict[str, float]:
        total = len(group)
        if not total:
            return {
                "execution_accuracy": 0.0,
                "exact_match": 0.0,
                "valid_sql_rate": 0.0,
                "stop_rate": 0.0,
            }
        return {
            "execution_accuracy": sum(bool(row["exec_match"]) for row in group) / total,
            "exact_match": sum(bool(row["exact_match"]) for row in group) / total,
            "valid_sql_rate": sum(bool(row["exec_ok"]) for row in group) / total,
            "stop_rate": sum(bool(row["stopped"]) for row in group) / total,
        }

    return {
        "overall": rates(rows),
        "by_difficulty": {
            difficulty: rates([row for row in rows if row.get("difficulty") == difficulty])
            for difficulty in DIFFICULTIES
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _query_rows(database: Path, sql: str, timeout: float = QUERY_TIMEOUT) -> list[tuple[Any, ...]]:
    """Execute one query using a read-only SQLite URI and an interrupt deadline."""
    uri = f"{database.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    # Sort/group temporary tables stay in memory, including restricted runtimes.
    connection.execute("PRAGMA temp_store=MEMORY")
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    connection.set_authorizer(
        lambda action, _a, _b, _database, _trigger:
        sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
    )
    deadline = time.monotonic() + timeout
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1_000)
    try:
        cursor = connection.execute(sql)
        if cursor.description is None:
            raise sqlite3.OperationalError("Expected a result-producing read query")
        return cursor.fetchall()
    finally:
        connection.close()


def _fewshot_prefix(path: Path) -> str:
    examples = _read_jsonl(path)
    rendered = []
    for example in examples:
        completion = example.get("completion")
        if completion is None:
            completion = " " + example.get("gold_sql", "")
        rendered.append(f"{example['prompt']}{completion}")
    return "\n\n".join(rendered) + ("\n\n" if rendered else "")


def _generation_eos_id(model: Any, tokenizer: Any) -> Any:
    """Use the model stop IDs, falling back to the tokenizer when unspecified."""
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    return tokenizer.eos_token_id if configured is None else configured


def _generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[tuple[str, bool]]:
    import torch

    tokenizer.padding_side = "left"
    eos_id = _generation_eos_id(model, tokenizer)
    eos_ids = set(eos_id if isinstance(eos_id, (tuple, list)) else [eos_id]) if eos_id is not None else set()
    pad_id = tokenizer.pad_token_id
    if pad_id is None and eos_id is not None:
        pad_id = eos_id[0] if isinstance(eos_id, (tuple, list)) else eos_id

    model.eval()
    generated_results: list[tuple[str, bool]] = []
    started = time.monotonic()
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        device = model.get_input_embeddings().weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        for sequence in output[:, prompt_width:]:
            token_ids = sequence.tolist()
            eos_positions = [i for i, token_id in enumerate(token_ids) if token_id in eos_ids]
            stopped = bool(eos_positions and eos_positions[0] < max_new_tokens - 1)
            content_ids = token_ids[:eos_positions[0]] if eos_positions else token_ids
            generated_results.append(
                (tokenizer.decode(content_ids, skip_special_tokens=True), stopped)
            )
        completed = min(start + batch_size, len(prompts))
        elapsed = time.monotonic() - started
        remaining = elapsed / completed * (len(prompts) - completed)
        print(f"Generated {completed}/{len(prompts)} | elapsed {elapsed:.1f}s | remaining ~{remaining:.1f}s", flush=True)
    return generated_results


def evaluate_records(
    records: list[dict[str, Any]],
    generations: list[tuple[str, bool]],
    *,
    timeout: float = QUERY_TIMEOUT,
    database_resolver=db_path,
) -> list[dict[str, Any]]:
    """Execute predicted/gold SQL and return the requested per-example fields."""
    if len(records) != len(generations):
        raise ValueError("The number of generations must match the number of records.")

    evaluated = []
    for index, (example, (raw_output, stopped)) in enumerate(zip(records, generations)):
        pred_sql = extract_sql(raw_output)
        gold_sql = example.get("gold_sql", "")
        database = Path(database_resolver(example["db_id"]))

        try:
            pred_rows = _query_rows(database, pred_sql, timeout)
            exec_ok = True
        except (sqlite3.Error, OSError, TimeoutError):
            pred_rows = []
            exec_ok = False

        try:
            gold_rows = _query_rows(database, gold_sql, timeout)
            gold_ok = True
        except (sqlite3.Error, OSError, TimeoutError):
            gold_rows = []
            gold_ok = False

        ordered = bool(re.search(r"\bORDER\s+BY\b", gold_sql, re.IGNORECASE))
        exec_match = bool(
            exec_ok and gold_ok and results_match(pred_rows, gold_rows, ordered=ordered)
        )
        evaluated.append({
            "id": example.get("id", index),
            "difficulty": example.get("difficulty", "unknown"),
            "question": example.get("question", ""),
            "raw_output": raw_output,
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "exec_ok": exec_ok,
            "exec_match": exec_match,
            "exact_match": normalize_sql(pred_sql) == normalize_sql(gold_sql),
            "stopped": bool(stopped),
        })
    return evaluated


def _saved_generations(path: Path, records: list[dict[str, Any]]) -> list[tuple[str, bool]]:
    """Rescore archived generations only after checking their dataset alignment."""
    predictions = _read_jsonl(path)
    if len(predictions) != len(records):
        raise ValueError("Saved prediction count does not match the test data")
    for index, (record, prediction) in enumerate(zip(records, predictions)):
        expected = {"id": record.get("id", index), "question": record.get("question", ""),
                    "gold_sql": record.get("gold_sql", "")}
        if any(prediction.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Saved prediction/data mismatch at example {index}")
        if not isinstance(prediction.get("raw_output"), str) or not isinstance(prediction.get("stopped"), bool):
            raise ValueError(f"Invalid saved generation at example {index}")
    return [(row["raw_output"], row["stopped"]) for row in predictions]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generated Spider SQL.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model path")
    parser.add_argument("--predictions", type=Path, help="Rescore a saved eval JSONL without loading the model")
    parser.add_argument("--adapter", help="Optional PEFT adapter path for SFT evaluation")
    parser.add_argument("--mode", choices=("zero", "few"), default="zero")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "test.jsonl")
    parser.add_argument("--fewshot", type=Path, default=ROOT / "data" / "fewshot.jsonl")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples")
    parser.add_argument("--run_name", required=True, help="Prefix for result files")
    parser.add_argument("--results_dir", type=Path, default=ROOT / "results")
    parser.add_argument("--timeout", type=float, default=QUERY_TIMEOUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if not args.data.is_file():
        raise SystemExit(f"Test data not found: {args.data}")

    records = _read_jsonl(args.data)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"No evaluation examples found in {args.data}")
    prefix = ""
    if args.mode == "few":
        if not args.fewshot.is_file():
            raise SystemExit(f"Few-shot examples not found: {args.fewshot}")
        prefix = _fewshot_prefix(args.fewshot)

    if args.predictions is not None:
        generations = _saved_generations(args.predictions, records)
    else:
        from src.utils import load_model, load_tokenizer, set_seed

        set_seed(42)
        tokenizer = load_tokenizer(args.model)
        tokenizer.padding_side = "left"
        model = load_model(args.model, adapter=args.adapter)
        prompts = [prefix + example["prompt"] for example in records]
        generations = _generate(model, tokenizer, prompts)
    results = evaluate_records(records, generations, timeout=args.timeout)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    eval_path = args.results_dir / f"{args.run_name}_eval.jsonl"
    metrics_path = args.results_dir / f"{args.run_name}_eval_metrics.json"
    with eval_path.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "evaluation_version": 2,
        "rescored_from": args.predictions.name if args.predictions is not None else None,
        "run_name": args.run_name,
        "model": args.model,
        "adapter": args.adapter,
        "mode": args.mode,
        "num_examples": len(results),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": BATCH_SIZE,
            "padding_side": "left",
        },
        "query_timeout_seconds": args.timeout,
        **metric_summary(results),
    }
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Per-example results: {eval_path}")
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
