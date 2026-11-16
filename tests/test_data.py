import ast
import json
import re
from pathlib import Path

import pytest

from src import data
from src.spider_hardness import eval_hardness
from tests.conftest import EASY, EXTRA, HARD, MEDIUM

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- schema_to_text
def test_schema_has_all_tables_and_foreign_keys(fake_spider):
    s = data.schema_to_text("concert_singer", fake_spider)
    for t in ["stadium", "singer", "concert", "singer_in_concert"]:
        assert f"CREATE TABLE {t} (" in s
    assert "FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID)" in s
    assert "FOREIGN KEY (Singer_ID) REFERENCES singer(Singer_ID)" in s
    assert "FOREIGN KEY (concert_ID) REFERENCES concert(concert_ID)" in s
    assert "Singer_ID INT PRIMARY KEY" in s
    assert "*" not in s  # служебная колонка "*" в схему не попадает
    assert len(s.splitlines()) == 4


def test_schema_sample_rows_flag(fake_spider):
    plain = data.schema_to_text("concert_singer", fake_spider)
    with_rows = data.schema_to_text("concert_singer", fake_spider, sample_rows=2)
    assert "example rows" not in plain
    assert "'Netherlands'" in with_rows and "'France'" not in with_rows  # только 2 строки


# ---------------------------------------------------------------- prompt
def test_prompt_matches_template_exactly():
    p = data.build_prompt("CREATE TABLE t (a INT);", "How many?")
    assert p == "Схема базы:\nCREATE TABLE t (a INT);\n\nВопрос: How many?\nSQL:"
    assert data.PROMPT_TEMPLATE == "Схема базы:\n{schema}\n\nВопрос: {question}\nSQL:"
    assert data.build_completion("SELECT 1;") == " SELECT 1"


def test_prompt_with_braces_in_question():
    assert "Вопрос: what is {x}?" in data.build_prompt("s", "what is {x}?")


@pytest.mark.skipif(not (ROOT / "tests/fixtures/tiny_train.jsonl").exists(),
                    reason="нет tests/fixtures/tiny_train.jsonl")
def test_compatible_with_tiny_train_fixture():
    pattern = re.compile(r"^Схема базы:\n.+\n\nВопрос: .+\nSQL:$", re.S)
    for line in (ROOT / "tests/fixtures/tiny_train.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        assert pattern.match(r["prompt"]), r["prompt"]
        assert r["completion"].startswith(" ") and not r["completion"].startswith("  ")


# ---------------------------------------------------------------- clean_sql
@pytest.mark.parametrize("raw, expected", [
    ("  SELECT count(*) FROM singer ;  ", "SELECT count(*) FROM singer"),
    ("SELECT Name ,  Country\n\tFROM singer;", "SELECT Name , Country FROM singer"),
    ("SELECT T1.name FROM a AS T1 WHERE x = 'France';;", "SELECT T1.name FROM a AS T1 WHERE x = 'France'"),
    ('select "Name" from Singer', 'select "Name" from Singer'),  # регистр и кавычки не трогаем
])
def test_clean_sql(raw, expected):
    assert data.clean_sql(raw) == expected


# ---------------------------------------------------------------- splits
def test_split_by_db_no_overlap(fake_spider):
    raw = json.loads((fake_spider / "train_spider.json").read_text())
    train, val = data.split_by_db(raw, val_size=10, seed=42)
    assert {e["db_id"] for e in train}.isdisjoint({e["db_id"] for e in val})
    assert len(train) + len(val) == len(raw)
    assert 8 <= len(val) <= 12
    assert data.split_by_db(raw, 10, 42) == (train, val)  # детерминированно


def test_assert_no_db_overlap_raises():
    with pytest.raises(ValueError):
        data.assert_no_db_overlap({"a": [{"db_id": "x"}], "b": [{"db_id": "x"}]})


# ---------------------------------------------------------------- hardness / exec
def test_hardness_levels():
    assert [eval_hardness(s) for s in (EASY, MEDIUM, HARD, EXTRA)] == ["easy", "medium", "hard", "extra"]


def test_execute_sql(fake_spider):
    db = data.db_path("concert_singer", fake_spider)
    assert data.execute_sql(db, "SELECT count(*) FROM singer") == (True, None)
    ok, err = data.execute_sql(db, "SELECT salary FROM singer")
    assert not ok and "salary" in err
    ok, err = data.execute_sql(db, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
                                   "SELECT count(*) FROM c", timeout=0.2)
    assert not ok and "timeout" in err
    ok, _ = data.execute_sql(db, "DELETE FROM singer")  # база открыта только на чтение
    assert not ok


# ---------------------------------------------------------------- весь пайплайн
def test_main_end_to_end(fake_spider, tmp_path):
    stats = data.main(["--spider_dir", str(fake_spider), "--out_dir", str(tmp_path),
                       "--val_size", "10", "--tokenizer", "", "--show", "0"])
    files = {n: [json.loads(l) for l in (tmp_path / f"{n}.jsonl").read_text().splitlines()]
             for n in ("train", "val", "test", "fewshot")}
    assert (tmp_path / "stats.json").exists()

    keys = ["id", "db_id", "difficulty", "question", "gold_sql", "prompt", "completion"]
    for recs in files.values():
        for r in recs:
            assert list(r) == keys
            assert r["completion"] == " " + r["gold_sql"]
            assert r["prompt"].endswith(f"Вопрос: {r['question']}\nSQL:")

    assert files["train"][0]["id"].endswith("_0001")
    assert stats["n_dropped_total"] == 1 and stats["dropped"][0]["question"] == "Broken one"
    assert all("salary" not in r["gold_sql"] for r in files["train"] + files["val"])
    assert len(files["test"]) == 2 and {r["difficulty"] for r in files["test"]} == {"easy", "extra"}
    data.assert_no_db_overlap({k: files[k] for k in ("train", "val", "test")})
    assert len({r["db_id"] for r in files["fewshot"]}) == len(files["fewshot"]) >= 3


def test_token_stats_with_fake_tokenizer():
    class FakeTok:  # 1 токен = 1 слово
        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [t.split() for t in texts]}
    recs = [{"db_id": "a", "difficulty": "easy", "prompt": "w " * n, "completion": " SELECT 1"}
            for n in (10, 20, 2000)]
    assert data.token_lengths(recs, FakeTok()) == [13, 23, 2003]  # + 2 слова SQL + EOS
    s = data.build_stats({"train": recs}, [], FakeTok(), "fake", 1024, {})
    tok = s["splits"]["train"]["tokens_prompt_plus_completion"]
    assert tok["max"] == 2003 and tok["n_over_1024"] == 1


# ---------------------------------------------------------------- сверка с официальным eval_hardness
def _load_official_hardness():
    """Достаём из официального evaluation.py только нужные функции (без импорта nltk и т.п.)."""
    src_file = ROOT / "third_party/spider_evaluation.py"
    tree = ast.parse(src_file.read_text(encoding="utf-8"))
    wanted = {"has_agg", "count_agg", "get_nestedSQL", "count_component1",
              "count_component2", "count_others"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) in {"AGG_OPS", "WHERE_OPS"} for t in node.targets):
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "Evaluator":
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == "eval_hardness":
                    body.append(f)
    ns: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(src_file), "exec"), ns)
    return lambda sql: ns["eval_hardness"](None, sql)


@pytest.mark.skipif(not (ROOT / "third_party/spider_evaluation.py").exists(),
                    reason="нет third_party/spider_evaluation.py (качает scripts/download_spider.sh)")
def test_hardness_matches_official():
    official = _load_official_hardness()
    sqls = [EASY, MEDIUM, HARD, EXTRA]
    for name in ("train_spider.json", "dev.json"):
        f = ROOT / "data/spider" / name
        if f.exists():
            sqls += [e["sql"] for e in json.loads(f.read_text(encoding="utf-8"))]
    assert [eval_hardness(s) for s in sqls] == [official(s) for s in sqls]
