"""Этап 3. Скрипт оценки (роль 2) — один и тот же для base и SFT.

Запуск:
    python -m src.evaluate --model <HF id или путь> [--adapter <путь>] \
        --mode zero|few --out results/<имя_запуска>.jsonl

Метрики: execution accuracy (главная), exact match, valid SQL rate,
средняя длина ответа.
"""
