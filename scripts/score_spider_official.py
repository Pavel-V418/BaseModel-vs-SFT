"""Compare saved project predictions with a local checkout of the Spider evaluator."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import random
import sqlite3
import subprocess
import sys
from collections import Counter
from contextlib import ExitStack, closing
from pathlib import Path
from types import SimpleNamespace

from src.data import db_path
from src.evaluate import QUERY_TIMEOUT, _read_connection, _read_jsonl, _saved_generations


def _schemas(upstream, databases):
    # Upstream get_schema does not close its SQLite connections.
    parser = importlib.import_module("process_sql")
    original_sqlite = parser.sqlite3
    with ExitStack() as connections:
        def connect(path):
            uri = f"{Path(path).resolve().as_uri()}?mode=ro"
            return connections.enter_context(closing(sqlite3.connect(uri, uri=True)))

        parser.sqlite3 = SimpleNamespace(connect=connect)
        try:
            return {db: upstream.Schema(upstream.get_schema(str(path)))
                    for db, path in databases.items()}
        finally:
            parser.sqlite3 = original_sqlite


def _legacy_execution(upstream, database, pred, gold, pred_ast, gold_ast):
    if pred_ast is None:
        return False
    original_sqlite = upstream.sqlite3
    with closing(_read_connection(database)) as connection:
        upstream.sqlite3 = SimpleNamespace(connect=lambda _: connection)
        try:
            return bool(upstream.eval_exec_match(str(database), pred, gold, pred_ast, gold_ast))
        finally:
            upstream.sqlite3 = original_sqlite


def _normalize(upstream, schema, key_map, parsed):
    columns = upstream.build_valid_col_units(parsed["from"]["table_units"], schema)
    return upstream.rebuild_sql_col(columns, upstream.rebuild_sql_val(parsed), key_map)


def score(args):
    records = _read_jsonl(args.data)
    if not records:
        raise ValueError("No examples to score")
    db_ids = sorted({record["db_id"] for record in records})
    databases = {db: db_path(db, args.spider_dir).resolve(strict=True) for db in db_ids}
    suite_counts = None
    if args.suite_dir is not None:
        suite_counts = {db: len(list((args.suite_dir / db).glob("*.sqlite"))) for db in db_ids}
        if any(count < 2 for count in suite_counts.values()):
            missing = [db for db, count in suite_counts.items() if count < 2]
            raise ValueError(f"Missing multiple test-suite databases for: {', '.join(missing)}")

    sys.path.insert(0, str(args.evaluator.resolve()))
    upstream = importlib.import_module("evaluation")
    if args.suite_dir is not None:
        executor = importlib.import_module("exec_eval")
        executor.get_cursor_from_path = lambda path: _read_connection(
            Path(path), text_errors="ignore",
        ).cursor()
    random.seed(42)
    key_maps = upstream.build_foreign_key_map_from_json(str(args.spider_dir / "tables.json"))
    schemas = _schemas(upstream, databases)
    gold_asts, gold_parse_errors = [], []
    for record in records:
        db = record["db_id"]
        try:
            parsed = upstream.get_sql(schemas[db], record["gold_sql"])
            gold_asts.append(_normalize(upstream, schemas[db], key_maps[db], parsed))
        except (AssertionError, IndexError, KeyError, ValueError) as error:
            gold_asts.append(None)
            gold_parse_errors.append({"id": record["id"], "db_id": db,
                                      "error": f"{type(error).__name__}: {error}"})

    metrics = {}
    comparator = upstream.Evaluator()
    has_execution = args.protocol == "legacy" or args.suite_dir is not None
    for predictions in args.predictions:
        _saved_generations(predictions, records)
        rows = _read_jsonl(predictions)
        exact_total = execution_total = 0
        exact_count = execution_count = 0
        errors = Counter()
        disagreements = {"exact": Counter(), "execution": Counter()}
        for record, row, gold_ast in zip(records, rows, gold_asts):
            db = record["db_id"]
            gold, pred = record["gold_sql"], row["pred_sql"]
            if args.protocol == "test-suite":
                pred = pred.replace("value", "1")  # Upstream CLI placeholder handling.
            pred_ast = None
            if gold_ast is not None:
                try:
                    parsed = upstream.get_sql(schemas[db], pred)
                    pred_ast = _normalize(upstream, schemas[db], key_maps[db], parsed)
                except (AssertionError, IndexError, KeyError, ValueError):
                    errors["prediction_parser"] += 1
                exact = bool(pred_ast is not None
                             and comparator.eval_exact_match(copy.deepcopy(pred_ast), copy.deepcopy(gold_ast)))
                exact_total += exact
                exact_count += 1
                disagreements["exact"][f"project_{int(row['exact_match'])}_spider_{int(exact)}"] += 1

            if args.protocol == "legacy" and gold_ast is not None:
                try:
                    executable = _legacy_execution(upstream, databases[db], pred, gold, pred_ast, gold_ast)
                except (sqlite3.Error, IndexError):
                    errors["legacy_execution_error"] += 1
                    executable = False
            elif args.suite_dir is not None:
                executable = bool(upstream.eval_exec_match(
                    db=str(args.suite_dir / db / f"{db}.sqlite"), p_str=pred, g_str=gold,
                    plug_value=False, keep_distinct=True, progress_bar_for_each_datapoint=False,
                ))
            else:
                continue
            execution_total += executable
            execution_count += 1
            disagreements["execution"][f"project_{int(row['exec_match'])}_spider_{int(executable)}"] += 1

        result = {
            "num_examples": len(records),
            "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            "exact_set_match": {"correct": exact_total, "evaluated": exact_count,
                                "accuracy": exact_total / exact_count if exact_count else None},
            "execution_accuracy": {"correct": execution_total, "evaluated": execution_count,
                                   "accuracy": execution_total / execution_count
                                   if execution_count else None} if has_execution else None,
            "errors": dict(errors),
            "agreement": {name: dict(counts) for name, counts in disagreements.items()},
        }
        metrics[predictions.stem.removesuffix("_eval")] = result
        print(predictions.stem, result["exact_set_match"], result["execution_accuracy"], flush=True)
    return {
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(args.evaluator), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "protocol": args.protocol,
        "sqlite_version": sqlite3.sqlite_version,
        "query_timeout_seconds": QUERY_TIMEOUT,
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "gold_parse_errors": gold_parse_errors,
        "test_suite_files_per_database": suite_counts,
        "runs": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Aligned Spider question JSONL")
    parser.add_argument("--spider-dir", type=Path, required=True, help="Directory with tables.json and database/")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--evaluator", type=Path, required=True, help="Upstream Git checkout")
    parser.add_argument("--protocol", choices=("legacy", "test-suite"), required=True)
    parser.add_argument("--suite-dir", type=Path,
                        help="Augmented databases. Omit for structural Exact Match only with test-suite.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.protocol == "legacy" and args.suite_dir is not None:
        parser.error("--suite-dir requires the test-suite protocol")
    result = score(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
