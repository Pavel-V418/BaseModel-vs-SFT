import pytest

from src.parse import extract_sql


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("SELECT count(*) FROM singer;\n\nВопрос: ...", "SELECT count(*) FROM singer"),
        ("SELECT 1\n\n explanation", "SELECT 1"),
        ("SELECT 1\nВопрос: next", "SELECT 1"),
        ("SELECT 1<|endoftext|>ignored", "SELECT 1"),
    ],
)
def test_stops_at_first_terminator(output, expected):
    assert extract_sql(output) == expected


def test_extracts_sql_fence_before_trailing_text():
    output = "Here is SQL:\n```sql\nSELECT name FROM singer;\n```\nExplanation"
    assert extract_sql(output) == "SELECT name FROM singer"


def test_returns_plain_sql_without_terminator():
    assert extract_sql("  SELECT 1  ") == "SELECT 1"


def test_empty_output_returns_empty_sql():
    assert extract_sql("") == ""
