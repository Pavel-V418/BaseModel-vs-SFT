# Данные: Spider → train / val / test (участник 1)

## Быстрый старт

```bash
pip install gdown transformers pytest
bash scripts/download_spider.sh            # Spider с базами -> data/spider/
python -m src.data --spider_dir data/spider --out_dir data --val_size 400 --seed 42
python -m pytest tests/test_data.py -q
```

## Что лежит в `data/` после запуска

| Файл | Что внутри |
|---|---|
| `train.jsonl` | train_spider.json без баз, ушедших в val (~6500) |
| `val.jsonl` | ~400 примеров из отдельных баз train (seed=42) |
| `test.jsonl` | dev.json целиком (~1000), базы не пересекаются с train |
| `fewshot.jsonl` | 4 коротких примера из train: разные базы, easy/medium/hard/extra |
| `stats.json` | размеры, число баз, сложность, длины в символах и токенах, выброшенные примеры с ошибками |

Формат строки одинаковый везде: `id, db_id, difficulty, question, gold_sql, prompt, completion`.

## Что импортировать в других модулях

```python
from src.data import PROMPT_TEMPLATE, build_prompt, schema_to_text, db_path, clean_sql, execute_sql
```

- `schema_to_text(db_id, spider_dir="data/spider", sample_rows=0)`: схема в виде CREATE TABLE с PK и FOREIGN KEY.
- `db_path(db_id)` возвращает `data/spider/database/<db_id>/<db_id>.sqlite`.
- `execute_sql(path, sql, timeout=5.0)` возвращает `(ok, error)`; база открывается только на чтение, по таймауту запрос прерывается.

## Решения, которые стоит знать

- **Сплит по базам.** Базы перемешиваются с seed и по одной уходят в val, пока там не наберётся ~`val_size` примеров (не больше 120%).
- **Типы колонок.** Типы Spider переводятся так: `number→INT`, `text→TEXT`, `time→DATETIME`, `boolean→BOOL`, `others→TEXT`.
- **Битые примеры.** Если эталонный SQL падает на своей базе, пример выбрасывается. Он попадает в `stats.json → dropped` вместе с текстом ошибки.
- **Длина в токенах.** Считается как `len(prompt) + len(completion) + 1 (EOS)`, токенизатор Qwen/Qwen2.5-0.5B. Поле `n_over_1024` показывает, сколько примеров обрежется в train.py.
- **Сложность.** Код `src/spider_hardness.py` перенесён из официального `evaluation.py` Spider. Тест `test_hardness_matches_official` сверяет его с оригиналом на всех примерах; оригинал скачивает `download_spider.sh` в `third_party/`.
- **Бонус: строки-примеры.** `--sample_rows 3` добавляет к схеме по 3 строки из каждой таблицы. По умолчанию выключено.
