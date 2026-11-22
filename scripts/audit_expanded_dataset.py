#!/usr/bin/env python3
"""Audit and package the EXACT saved expanded Spider JSONL splits, not regenerate them."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

EXPECTED = {"train": 8222, "val": 300, "test": 1034}
KEYS = ("db_id", "question", "gold_sql", "prompt", "completion")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict) or any(not isinstance(obj.get(k), str) for k in KEYS):
                    raise ValueError(f"{path}:{number}: invalid required fields")
                rows.append(obj)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing saved train/val/test.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--archive", action="store_true", help="Create private upload ZIP containing the three JSONL files")
    parser.add_argument("--expected-train-sha256", help="Known SHA-256 from original training run (required to certify exact identity)")
    parser.add_argument("--expected-val-sha256", help="Known SHA-256 from original training run")
    args = parser.parse_args()

    paths = {split: args.data_dir / f"{split}.jsonl" for split in EXPECTED}
    for split, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Recover ORIGINAL training files first; don't silently recreate them.")
    rows = {split: load(path) for split, path in paths.items()}
    counts = {split: len(data) for split, data in rows.items()}
    if counts != EXPECTED:
        raise SystemExit(f"Unexpected counts: {counts}; expected {EXPECTED}")

    databases = {split: {r["db_id"] for r in data} for split, data in rows.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = databases[left] & databases[right]
        if shared:
            raise SystemExit(f"DB leakage {left}/{right}: {sorted(shared)}")

    def canonical(row: dict) -> tuple[str, str, str]:
        return (row["db_id"], " ".join(row["question"].split()).casefold(),
                " ".join(row["gold_sql"].split()).casefold())

    signatures = {split: [canonical(r) for r in data] for split, data in rows.items()}
    for split, values in signatures.items():
        duplicates = sum(n - 1 for n in Counter(values).values() if n > 1)
        if duplicates:
            raise SystemExit(f"{split}: {duplicates} duplicate (db_id, question, SQL) entries")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        common = set(signatures[left]) & set(signatures[right])
        if common:
            raise SystemExit(f"Exact-example leakage {left}/{right}: {len(common)}")

    hashes = {split: sha256(path) for split, path in paths.items()}
    verified = {}
    for split, expected in (("train", args.expected_train_sha256), ("val", args.expected_val_sha256)):
        if expected is None:
            verified[split] = "not_verified_against_training_artifact"
        else:
            verified[split] = hashes[split] == expected.lower()
            if not verified[split]:
                raise SystemExit(f"{split} hash mismatch: expected {expected}, got {hashes[split]}")

    report = {
        "name": "spider-expanded-v1-candidate",
        "status": "verified_original_training_data" if all(v is True for v in verified.values()) else "candidate_only",
        "counts": counts,
        "db_counts": {split: len(dbs) for split, dbs in databases.items()},
        "db_overlap": 0,
        "exact_duplicate_count": 0,
        "sha256": hashes,
        "training_artifact_hash_match": verified,
        "note": "JSONL provenance cannot be established from row counts alone. Original training file hashes are required."
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "expanded_v1_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.archive:
        with zipfile.ZipFile(args.out_dir / "spider_expanded_v1_private.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for split, path in paths.items():
                z.write(path, f"data/{split}.jsonl")
            z.write(args.out_dir / "expanded_v1_manifest.json", "manifest.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
