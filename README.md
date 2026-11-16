# Base model vs SFT model: Text-to-SQL

Проектная практика, НИЯУ МИФИ.

**Цель:** оценить base-модель до fine-tuning, выполнить SFT и повторить тот же тест,
чтобы показать, что меняется после supervised instruction tuning.

- **Задача:** Text-to-SQL (вопрос на естественном языке + схема базы → SQL)
- **Датасет:** Spider
- **Модель:** Qwen/Qwen2.5-0.5B (base); бонусы: 1.5B, 3B, сравнение с -Instruct
- **Обучение:** TRL SFTTrainer + LoRA, Kaggle (T4)
- **Метрики:** validation loss, execution accuracy, exact match, valid SQL rate

## Структура

```
configs/      YAML-конфиги запусков
src/          data.py, parse.py, evaluate.py, train.py, utils.py
notebooks/    тонкие ноутбуки для Kaggle/Colab
scripts/      вспомогательные скрипты (скачивание Spider)
results/      метрики и ответы моделей (маленькие файлы)
tests/        тесты (парсер и проверка SQL)
data/         датасеты (не в git)
outputs/      адаптеры и чекпоинты (не в git)
```

## Форматы

Данные (`data/*.jsonl`):
```json
{"id": "train_0001", "db_id": "concert_singer", "difficulty": "easy", "prompt": "...", "completion": " SELECT ..."}
```

Результаты оценки (`results/*.jsonl`):
```json
{"id": "test_0001", "model": "sft_0.5b_r16", "raw_output": "...", "pred_sql": "...", "gold_sql": "...", "exec_ok": true, "exec_match": true, "exact_match": false}
```

## Роли

| # | Роль | Модуль |
|---|---|---|
| 1 | Данные | `src/data.py` |
| 2 | Оценка | `src/parse.py`, `src/evaluate.py` |
| 3 | Обучение | `src/train.py` |
| 4 | Инфраструктура и масштаб | `configs/`, `src/utils.py`, W&B, README |
| 5 | Анализ и демо | разбор ошибок, графики, Gradio |

## Главное правило

Base и SFT проверяются одинаково: один test, один промпт, один скрипт, одни настройки генерации.
