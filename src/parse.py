"""Этап 2. Парсер ответа модели (роль 2).

extract_sql(model_output: str) -> str
Вырезает SQL-запрос из сырого ответа модели (до первой ';' или пустой строки).
"""


def extract_sql(model_output: str) -> str:
    raise NotImplementedError
