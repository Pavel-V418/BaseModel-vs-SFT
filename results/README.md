# Результаты

Полное описание методики, запуска и аудита: [docs/evaluation.md](../docs/evaluation.md).

| Модель | Execution accuracy | Exact match | Valid SQL | Stop rate |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B base | 24.47% | 7.93% | 61.61% | 76.98% |
| Qwen2.5-0.5B + LoRA | 49.23% | 31.62% | 73.40% | 99.90% |

1034 примера, zero-shot, Colab Tesla T4, 17.11.2026. Два полных запуска дали одинаковые JSONL. Опубликованные файлы повторно проверены evaluator v2 без изменения метрик.

- *_eval.jsonl — 1034 ответа и флаги на каждую модель; *_eval_metrics.json — общие и по сложности метрики.
- evaluation_manifest.json — происхождение, контрольные суммы, подтверждение повторяемости.
- environment_colab.json — исходное окружение замеров; requirements-colab-measured.txt — версии основных пакетов, а не универсальная команда установки CUDA.
- sft_0.5b_nall_r16_lr1e-4_s42_train.json и *_log_history.json — отчёты присланного обучения на Kaggle; это отдельное окружение от замеров Colab.

Большие архивы, датасет и веса в Git не включены.
