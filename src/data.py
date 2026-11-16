"""Этап 1. Данные (роль 1).

Загрузка Spider, разбиение на train / val / test,
превращение схемы базы в текст и сборка промптов формата prompt -> completion.

Выход: data/{train,val,test}.jsonl с полями
{"id", "db_id", "difficulty", "prompt", "completion"}
"""
