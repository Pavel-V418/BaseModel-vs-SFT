"""Тесты парсера: если он сломается, сломаются все метрики."""
from src.parse import extract_sql


def test_cuts_trailing_text():
    out = "SELECT count(*) FROM singer;\n\nВопрос: ..."
    assert extract_sql(out) == "SELECT count(*) FROM singer"
