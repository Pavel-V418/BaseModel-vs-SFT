# Результаты обучения Qwen2.5 SFT на Spider

Оба запуска использовали расширенную обучающую выборку из 8 222 примеров, 300 примеров для валидации, LoRA (r=16, α=32), learning rate 0,0001, seed 42 и две эпохи на Tesla T4 в Kaggle. Оценка во время обучения выполнялась каждые 100 шагов для 0.5B и каждые 50 шагов для 1.5B. В обоих запусках лучший чекпоинт — `checkpoint-400`.

| Метрика обучения | Qwen2.5-0.5B SFT | Qwen2.5-1.5B SFT |
|---|---:|---:|
| Validation loss до → после SFT | 0,7540 → 0,2841 | 0,6558 → 0,2072 |
| Validation token accuracy до → после SFT | 80,25% → 91,71% | 83,18% → 93,61% |
| Validation perplexity после SFT | 1,3285 | 1,2302 |
| Обучаемые параметры | 8 798 208 | 18 464 768 |
| Время обучения | 54 мин 24 с | 1 ч 42 мин 47 с |

| Запуск | Исходный отчёт | История по шагам | Графики |
|---|---|---|---|
| 0.5B | [JSON](sft_0.5b_nfull_others_r16_lr1e-4_s42_2ep_train.json) | [JSON](sft_0.5b_nfull_others_r16_lr1e-4_s42_2ep_log_history.json) | [PNG](sft_0.5b_nfull_others_r16_lr1e-4_s42_2ep_curves.png) |
| 1.5B | [JSON](sft_1.5b_nfull_others_r16_lr1e-4_s42_2ep_train.json) | [JSON](sft_1.5b_nfull_others_r16_lr1e-4_s42_2ep_log_history.json) | [PNG](sft_1.5b_nfull_others_r16_lr1e-4_s42_2ep_curves.png) |

Validation token accuracy измеряет совпадение токенов в ходе валидации. Для сравнения моделей по выполнению сгенерированных SQL на Spider dev используйте [Kaggle-ноутбук](../notebooks/kaggle_evaluate_sft.ipynb) и [инструкцию запуска](../notebooks/kaggle_evaluate_sft.md). Веса адаптеров для него берутся из исходных ZIP.
