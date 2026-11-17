"""Подготовка Spider: сырые JSON и базы -> train / val / test / fewshot в едином формате.

Запуск:
    python -m src.data --spider_dir data/spider --out_dir data --val_size 400 --seed 42

Что делает:
  1. Читает train_spider.json (-> train + val) и dev.json (-> test).
  2. Делит train/val ПО БАЗАМ (db_id): val получает несколько баз целиком.
  3. Схему каждой базы превращает в текст CREATE TABLE ... (schema_to_text).
  4. Собирает prompt по зафиксированному шаблону PROMPT_TEMPLATE и completion.
  5. Считает сложность (easy/medium/hard/extra) официальной функцией Spider.
  6. Выполняет все эталонные SQL на базах и выкидывает упавшие.
  7. Считает длины в токенах (Qwen/Qwen2.5-0.5B) и пишет stats.json.

Импорты для других модулей:
    from src.data import PROMPT_TEMPLATE, build_prompt, schema_to_text, db_path
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from src.spider_hardness import DIFFICULTIES, eval_hardness

# ---------------------------------------------------------------------------
# Константы. Шаблон согласован с участниками 2 и 3, менять только вместе с ними.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = "Схема базы:\n{schema}\n\nВопрос: {question}\nSQL:"

DEFAULT_SPIDER_DIR = Path("data/spider")
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-0.5B"
DEFAULT_MAX_LENGTH = 1024  # как в train.py

# Типы колонок Spider -> SQL-типы для текста схемы.
TYPE_MAP = {"text": "TEXT", "number": "INT", "time": "DATETIME", "boolean": "BOOL", "others": "TEXT"}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# Пути и схемы
# ---------------------------------------------------------------------------
def db_path(db_id: str, spider_dir: str | Path = DEFAULT_SPIDER_DIR) -> Path:
    """Путь к SQLite-базе: <spider_dir>/database/<db_id>/<db_id>.sqlite"""
    return Path(spider_dir) / "database" / db_id / f"{db_id}.sqlite"


@lru_cache(maxsize=None)
def load_tables(spider_dir: str | Path = DEFAULT_SPIDER_DIR) -> dict[str, dict]:
    """tables.json -> {db_id: описание схемы}."""
    with open(Path(spider_dir) / "tables.json", encoding="utf-8") as f:
        return {t["db_id"]: t for t in json.load(f)}


def _q(name: str) -> str:
    """Имя как есть, а если в нём пробел или спецсимвол, то в двойных кавычках."""
    return name if _IDENT_RE.match(name) else '"' + name.replace('"', '""') + '"'


def _flatten(items) -> list[int]:
    out = []
    for x in items:
        out.extend(_flatten(x) if isinstance(x, (list, tuple)) else [x])
    return out


def _fmt_value(v, max_chars: int = 40) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    if isinstance(v, str):
        v = " ".join(v.split())
        if len(v) > max_chars:
            v = v[: max_chars - 3] + "..."
        return "'" + v.replace("'", "''") + "'"
    return str(v)


def _sample_rows(db_file: Path, table: str, n: int) -> list[tuple]:
    try:
        conn = sqlite3.connect(f"{Path(db_file).resolve().as_uri()}?mode=ro", uri=True)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        try:
            return conn.execute(f"SELECT * FROM {_q(table)} LIMIT {int(n)}").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def schema_to_text(
    db_id: str,
    spider_dir: str | Path = DEFAULT_SPIDER_DIR,
    sample_rows: int = 0,
) -> str:
    """Схема базы в виде CREATE TABLE-строк (одна таблица на строку).

    Используются *_original имена (они в базе и в эталонных SQL), типы колонок,
    первичные и внешние ключи. sample_rows > 0 добавляет после каждой таблицы
    комментарий с первыми N строками (для бонусного эксперимента, по умолчанию выкл).
    """
    t = load_tables(str(spider_dir))[db_id]
    tables = t["table_names_original"]
    cols = t["column_names_original"]  # [[table_idx, name], ...], cols[0] = [-1, "*"]
    types = t["column_types"]
    pks = set(_flatten(t["primary_keys"]))
    fks = t["foreign_keys"]  # [[col_idx, ref_col_idx], ...]

    lines = []
    for ti, table in enumerate(tables):
        col_ids = [ci for ci, (tj, _) in enumerate(cols) if tj == ti]
        table_pks = [ci for ci in col_ids if ci in pks]
        parts = []
        for ci in col_ids:
            part = f"{_q(cols[ci][1])} {TYPE_MAP.get(types[ci], 'TEXT')}"
            if len(table_pks) == 1 and ci in pks:
                part += " PRIMARY KEY"
            parts.append(part)
        if len(table_pks) > 1:
            parts.append("PRIMARY KEY (" + ", ".join(_q(cols[ci][1]) for ci in table_pks) + ")")
        for src, ref in fks:
            if cols[src][0] == ti:
                ref_table = tables[cols[ref][0]]
                parts.append(
                    f"FOREIGN KEY ({_q(cols[src][1])}) REFERENCES {_q(ref_table)}({_q(cols[ref][1])})"
                )
        lines.append(f"CREATE TABLE {_q(table)} (" + ", ".join(parts) + ");")

        if sample_rows > 0:
            rows = _sample_rows(db_path(db_id, spider_dir), table, sample_rows)
            if rows:
                shown = "; ".join("(" + ", ".join(_fmt_value(v) for v in r) + ")" for r in rows)
                lines.append(f"/* {len(rows)} example rows from {table}: {shown} */")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Промпт и очистка SQL
# ---------------------------------------------------------------------------
def clean_sql(sql: str) -> str:
    """Пробелы по краям и ; в конце убрать, пробельные последовательности схлопнуть.
    Регистр, кавычки и псевдонимы не трогаем."""
    sql = sql.strip().rstrip(";").strip()
    return re.sub(r"\s+", " ", sql)


def build_prompt(schema: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(schema=schema, question=question.strip())


def build_completion(gold_sql: str) -> str:
    # Пробел в начале обязателен (после "SQL:"). EOS добавит TRL.
    return " " + clean_sql(gold_sql)


# ---------------------------------------------------------------------------
# Выполнение SQL
# ---------------------------------------------------------------------------
def execute_sql(db_file: Path, sql: str, timeout: float = 5.0) -> tuple[bool, str | None]:
    """Выполнить запрос только на чтение. Возвращает (ok, текст ошибки)."""
    if not Path(db_file).exists():
        return False, f"database not found: {db_file}"
    try:
        conn = sqlite3.connect(f"{Path(db_file).resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as e:
        return False, f"cannot open db: {e}"
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        conn.execute(sql).fetchall()
        return True, None
    except sqlite3.OperationalError as e:
        msg = str(e)
        return False, f"timeout > {timeout}s" if msg == "interrupted" else msg
    except Exception as e:  # noqa: BLE001 - любые ошибки = битый пример
        return False, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Сплиты
# ---------------------------------------------------------------------------
def split_by_db(examples: list[dict], val_size: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Делит примеры на train/val целыми базами.

    Базы перемешиваются с фиксированным seed и по очереди отдаются в val, пока в нём
    не наберётся ~val_size примеров. Базу, с которой val вышел бы больше 120% от
    val_size, пропускаем и берём следующую.
    """
    by_db = Counter(ex["db_id"] for ex in examples)
    db_ids = sorted(by_db)
    random.Random(seed).shuffle(db_ids)

    val_dbs, total = set(), 0
    for db in db_ids:
        if total >= val_size:
            break
        if total + by_db[db] <= val_size * 1.2:
            val_dbs.add(db)
            total += by_db[db]

    train = [ex for ex in examples if ex["db_id"] not in val_dbs]
    val = [ex for ex in examples if ex["db_id"] in val_dbs]
    return train, val


def assert_no_db_overlap(splits: dict[str, list[dict]]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = {e["db_id"] for e in splits[a]} & {e["db_id"] for e in splits[b]}
            if common:
                raise ValueError(f"db_id пересекаются между {a} и {b}: {sorted(common)[:10]}")


# ---------------------------------------------------------------------------
# Сборка записей
# ---------------------------------------------------------------------------
def make_records(
    raw: list[dict],
    split: str,
    spider_dir: Path,
    sample_rows: int = 0,
    check_exec: bool = True,
    timeout: float = 5.0,
) -> tuple[list[dict], list[dict]]:
    """Сырые примеры Spider -> записи нашего формата. Возвращает (records, dropped)."""
    records, dropped = [], []
    schema_cache: dict[str, str] = {}
    for ex in raw:
        db_id = ex["db_id"]
        gold_sql = clean_sql(ex["query"])
        if check_exec:
            ok, err = execute_sql(db_path(db_id, spider_dir), gold_sql, timeout)
            if not ok:
                dropped.append({"split": split, "db_id": db_id, "question": ex["question"],
                                "gold_sql": gold_sql, "error": err})
                continue
        if db_id not in schema_cache:
            schema_cache[db_id] = schema_to_text(db_id, spider_dir, sample_rows)
        try:
            difficulty = eval_hardness(ex["sql"]) if "sql" in ex else None
        except Exception:  # noqa: BLE001 - на всякий случай не роняем весь пайплайн
            difficulty = None
        records.append({
            "id": f"{split}_{len(records) + 1:04d}",
            "db_id": db_id,
            "difficulty": difficulty,
            "question": ex["question"].strip(),
            "gold_sql": gold_sql,
            "prompt": build_prompt(schema_cache[db_id], ex["question"]),
            "completion": build_completion(gold_sql),
        })
    return records, dropped


def pick_fewshot(train: list[dict], n: int = 4, seed: int = 42) -> list[dict]:
    """n примеров из train: разные базы, по возможности разная сложность, короткие промпты
    (few-shot промпт содержит схему каждого примера, поэтому длинные схемы не берём)."""
    rng = random.Random(seed)
    median_len = statistics.median(len(r["prompt"]) for r in train)
    pool = [r for r in train if len(r["prompt"]) <= median_len]
    rng.shuffle(pool)
    chosen, used_dbs = [], set()
    for diff in list(DIFFICULTIES) * 2:  # второй круг добирает, если какой-то сложности нет
        if len(chosen) >= n:
            break
        for r in pool:
            if r["difficulty"] == diff and r["db_id"] not in used_dbs and r not in chosen:
                chosen.append(r)
                used_dbs.add(r["db_id"])
                break
    for r in pool:  # добор, если сложностей не хватило
        if len(chosen) >= n:
            break
        if r["db_id"] not in used_dbs:
            chosen.append(r)
            used_dbs.add(r["db_id"])
    return chosen


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------
def _percentile(values: list[int], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _length_summary(values: list[int], max_length: int | None = None) -> dict:
    if not values:
        return {}
    out = {
        "mean": round(statistics.mean(values), 1),
        "p95": round(_percentile(values, 0.95), 1),
        "max": max(values),
    }
    if max_length is not None:
        out[f"n_over_{max_length}"] = sum(v > max_length for v in values)
    return out


def token_lengths(records: list[dict], tokenizer) -> list[int]:
    """Длина prompt + completion + EOS в токенах (так считает TRL при обучении)."""
    p = tokenizer([r["prompt"] for r in records], add_special_tokens=False)["input_ids"]
    c = tokenizer([r["completion"] for r in records], add_special_tokens=False)["input_ids"]
    return [len(a) + len(b) + 1 for a, b in zip(p, c)]


def load_tokenizer(name: str | None):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: токенизатор {name} не загрузился ({e}); длины в токенах не посчитаны.",
              file=sys.stderr)
        return None


def build_stats(splits: dict[str, list[dict]], dropped: list[dict], tokenizer,
                tokenizer_name: str | None, max_length: int, args: dict) -> dict:
    stats: dict = {"args": args, "splits": {}}
    for name, recs in splits.items():
        s = {
            "n_examples": len(recs),
            "n_databases": len({r["db_id"] for r in recs}),
            "difficulty": dict(Counter(r["difficulty"] for r in recs).most_common()),
            "n_dropped": sum(d["split"] == name for d in dropped),
            "chars_prompt_plus_completion": _length_summary(
                [len(r["prompt"]) + len(r["completion"]) for r in recs]),
        }
        if tokenizer is not None and recs:
            s["tokens_prompt_plus_completion"] = _length_summary(token_lengths(recs, tokenizer), max_length)
        stats["splits"][name] = s
    stats["tokenizer"] = tokenizer_name if tokenizer is not None else None
    stats["max_length"] = max_length
    stats["n_dropped_total"] = len(dropped)
    stats["dropped"] = dropped
    return stats


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Spider -> train/val/test/fewshot jsonl")
    p.add_argument("--spider_dir", type=Path, default=DEFAULT_SPIDER_DIR)
    p.add_argument("--out_dir", type=Path, default=Path("data"))
    p.add_argument("--val_size", type=int, default=400, help="примерное число примеров в val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rows", type=int, default=0,
                   help="добавлять N строк-примеров из каждой таблицы в схему (0 = выкл)")
    p.add_argument("--n_fewshot", type=int, default=4)
    p.add_argument("--timeout", type=float, default=5.0, help="таймаут выполнения SQL, сек")
    p.add_argument("--no_exec_check", action="store_true", help="не выполнять gold SQL (быстро, для отладки)")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="'' чтобы не считать токены")
    p.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--show", type=int, default=3, help="сколько случайных примеров напечатать")
    p.add_argument("--include_others", action="store_true", help="добавить train_others.json в обучающую выборку")
    return p.parse_args(argv)

def main(argv=None) -> dict:
    args = parse_args(argv)
    spider_dir: Path = args.spider_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(spider_dir / "train_spider.json", encoding="utf-8") as f:
        raw_train = json.load(f)
    with open(spider_dir / "dev.json", encoding="utf-8") as f:
        raw_test = json.load(f)

    raw_train_split, raw_val_split = split_by_db(raw_train, args.val_size, args.seed)

    if args.include_others:
        others_path = spider_dir / "train_others.json"
        if others_path.exists():
            with open(others_path, encoding="utf-8") as f:
                raw_others = json.load(f)
            raw_train_split.extend(raw_others)
            print(f"Добавлено {len(raw_others)} примеров из train_others.json")

    splits, dropped = {}, []
    for name, raw in [("train", raw_train_split), ("val", raw_val_split), ("test", raw_test)]:
        recs, drop = make_records(raw, name, spider_dir, args.sample_rows,
                                  check_exec=not args.no_exec_check, timeout=args.timeout)
        splits[name] = recs
        dropped += drop
        print(f"{name}: {len(recs)} примеров, {len({r['db_id'] for r in recs})} баз, выброшено {len(drop)}")

    assert_no_db_overlap(splits)

    fewshot = pick_fewshot(splits["train"], args.n_fewshot, args.seed)
    fewshot = [{**r, "id": f"fewshot_{i + 1:04d}"} for i, r in enumerate(fewshot)]

    for name, recs in splits.items():
        write_jsonl(args.out_dir / f"{name}.jsonl", recs)
    write_jsonl(args.out_dir / "fewshot.jsonl", fewshot)

    tokenizer = load_tokenizer(args.tokenizer)
    stats = build_stats(
        splits, dropped, tokenizer, args.tokenizer, args.max_length,
        args={k: str(v) for k, v in vars(args).items()},
    )
    stats["splits"]["fewshot"] = {
        "n_examples": len(fewshot),
        "db_ids": [r["db_id"] for r in fewshot],
        "difficulty": [r["difficulty"] for r in fewshot],
    }
    with open(args.out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    for name in ("train", "val", "test"):
        s = stats["splits"][name]
        tok = s.get("tokens_prompt_plus_completion")
        print(f"[{name}] сложность: {s['difficulty']}" + (f" | токены: {tok}" if tok else ""))
    if args.show and splits["train"]:
        rng = random.Random(args.seed)
        for r in rng.sample(splits["train"], min(args.show, len(splits["train"]))):
            print("\n" + "=" * 80 + f"\n{r['id']} | {r['db_id']} | {r['difficulty']}\n" + "-" * 80)
            print(r["prompt"] + r["completion"])
    print(f"\nГотово: {args.out_dir}/{{train,val,test,fewshot}}.jsonl + stats.json")
    return stats

if __name__ == "__main__":
    main()
