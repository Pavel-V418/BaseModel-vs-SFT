"""Evaluation helper tests."""
from src.evaluate import metric_summary, normalize_sql, results_match


def test_normalize_sql_ignores_case_and_whitespace():
    assert normalize_sql(" SELECT  A  FROM T ; ") == normalize_sql("select a from t")


def test_results_ignore_row_order_without_gold_order_by():
    assert results_match([(2,), (1,)], [(1,), (2,)], ordered=False)


def test_results_keep_duplicates_and_round_numbers():
    assert results_match([(1.2345671,), (1,)], [(1,), (1.2345674,)], ordered=False)
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)


def test_results_respect_order_when_gold_has_order_by():
    assert not results_match([(2,), (1,)], [(1,), (2,)], ordered=True)


def test_metric_summary_reports_overall_and_by_difficulty():
    rows = [
        {"difficulty": "easy", "exec_ok": True, "exec_match": True, "exact_match": True, "stopped": True},
        {"difficulty": "hard", "exec_ok": False, "exec_match": False, "exact_match": False, "stopped": False},
    ]
    summary = metric_summary(rows)
    assert summary["overall"]["execution_accuracy"] == 0.5
    assert summary["by_difficulty"]["easy"]["valid_sql_rate"] == 1.0
    assert summary["by_difficulty"]["hard"]["stop_rate"] == 0.0


def test_evaluate_records_executes_read_only_and_compares_rows(tmp_path):
    import sqlite3

    from src.evaluate import evaluate_records

    database = tmp_path / "tiny.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript("CREATE TABLE item (value REAL); INSERT INTO item VALUES (1.0000001); INSERT INTO item VALUES (2.0);")
    connection.commit()
    connection.close()

    records = [{"id": "x", "db_id": "tiny", "difficulty": "easy", "question": "values",
                "gold_sql": "SELECT value FROM item", "prompt": "SQL:"}]
    result = evaluate_records(records, [("SELECT value FROM item WHERE value > 0", True)],
                              database_resolver=lambda _: database)[0]

    assert result["exec_ok"] is True
    assert result["exec_match"] is True
    read_only = __import__("src.evaluate", fromlist=["_query_rows"])._query_rows
    try:
        read_only(database, "DELETE FROM item")
    except sqlite3.DatabaseError:
        pass
    else:
        raise AssertionError("SQLite evaluation connection allowed a write")
    assert sqlite3.connect(database).execute("SELECT count(*) FROM item").fetchone()[0] == 2


def test_result_comparison_preserves_integer_precision_and_blob_types():
    assert not results_match([(9007199254740992,)], [(9007199254740993,)], ordered=False)
    assert results_match([(1,)], [(1.0,)], ordered=False)
    assert not results_match([(b'x',)], [('x',)], ordered=False)
    assert not results_match([(b'\xff',)], [(b'\xfe',)], ordered=False)


def test_empty_and_administrative_sql_cannot_match_empty_gold(tmp_path):
    import sqlite3
    from src.evaluate import evaluate_records
    database = tmp_path / 'empty.sqlite'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE item (value INTEGER)')
    record = {'id': 'empty', 'db_id': 'empty', 'difficulty': 'easy', 'gold_sql': 'SELECT value FROM item'}
    for prediction in ('', '-- blank', 'BEGIN', 'PRAGMA user_version'):
        row = evaluate_records([record], [(prediction, True)], database_resolver=lambda _: database)[0]
        assert row['exec_ok'] is False, prediction
        assert row['exec_match'] is False, prediction


def test_invalid_utf8_in_sqlite_text_preserves_bytes_and_type(tmp_path):
    import sqlite3

    from src.evaluate import evaluate_records

    database = tmp_path / "invalid_text.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE item (value TEXT)")
        connection.execute("INSERT INTO item VALUES (CAST(X'FF' AS TEXT))")

    record = {"db_id": "invalid_text", "gold_sql": "SELECT value FROM item"}
    for sql, expected_match in (
        ("SELECT value FROM item", True),
        ("SELECT CAST(X'FE' AS TEXT)", False),
        ("SELECT X'FF'", False),
    ):
        row = evaluate_records([record], [(sql, True)], database_resolver=lambda _: database)[0]
        assert row["exec_ok"] is True
        assert row["exec_match"] is expected_match


def test_sql_cannot_create_external_files(tmp_path):
    import sqlite3
    from src.evaluate import _query_rows
    database = tmp_path / 'main.sqlite'
    sqlite3.connect(database).close()
    for index, operation in enumerate(('ATTACH DATABASE', 'VACUUM INTO')):
        target = tmp_path / f'external_{index}.sqlite'
        sql = f"{operation} '{target.as_posix()}'"
        if operation == 'ATTACH DATABASE':
            sql += ' AS external'
        try:
            _query_rows(database, sql)
        except sqlite3.DatabaseError:
            pass
        else:
            raise AssertionError(f'Allowed {operation}')
        assert not target.exists()


def test_query_timeout_interrupts_recursive_queries(tmp_path):
    import sqlite3
    from src.evaluate import _query_rows
    database = tmp_path / 'deadline.sqlite'
    sqlite3.connect(database).close()
    try:
        _query_rows(database, 'WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n', timeout=0.01)
    except sqlite3.OperationalError as exc:
        assert 'interrupted' in str(exc)
    else:
        raise AssertionError('Query did not time out')


def test_generation_uses_model_eos_configuration():
    from types import SimpleNamespace
    from src.evaluate import _generation_eos_id
    tokenizer = SimpleNamespace(eos_token_id=1)
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[1, 2]))
    assert _generation_eos_id(model, tokenizer) == [1, 2]
    model.generation_config.eos_token_id = None
    assert _generation_eos_id(model, tokenizer) == 1


def test_saved_predictions_must_match_test_records(tmp_path):
    import json
    import pytest
    from src.evaluate import _saved_generations
    records = [{'id': 0, 'question': 'q', 'gold_sql': 'SELECT 1'}]
    path = tmp_path / 'predictions.jsonl'
    saved = {'id': 0, 'question': 'q', 'gold_sql': 'SELECT 1', 'raw_output': 'SELECT 1', 'stopped': True}
    path.write_text(json.dumps(saved)+'\n', encoding='utf-8')
    assert _saved_generations(path, records) == [('SELECT 1', True)]
    saved['id'] = 1
    path.write_text(json.dumps(saved)+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='mismatch'):
        _saved_generations(path, records)
    path.write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='count'):
        _saved_generations(path, records)


def test_cli_rescores_saved_predictions_against_selected_spider_dir(tmp_path):
    import json
    import sqlite3

    from src.evaluate import main

    database = tmp_path / "database" / "tiny" / "tiny.sqlite"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE item (value TEXT)")
        connection.execute("INSERT INTO item VALUES ('x;y')")
    record = {"id": "x", "db_id": "tiny", "question": "value", "gold_sql": "SELECT value FROM item"}
    data = tmp_path / "questions.jsonl"
    predictions = tmp_path / "saved.jsonl"
    data.write_text(json.dumps(record) + "\n", encoding="utf-8")
    predictions.write_text(json.dumps({**record, "raw_output": "SELECT 'x;y';", "stopped": True}) + "\n",
                           encoding="utf-8")

    assert main(["--model", "unused", "--predictions", str(predictions), "--data", str(data),
                 "--spider-dir", str(tmp_path), "--run_name", "rescore", "--results_dir", str(tmp_path)]) == 0
    result = json.loads((tmp_path / "rescore_eval.jsonl").read_text(encoding="utf-8"))
    assert result["exec_match"] is True
