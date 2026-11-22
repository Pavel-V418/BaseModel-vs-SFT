# Spider 1.0 test

Пять конфигураций на 2147 вопросах из 40 баз. Проектные метрики пересчитаны оценщиком v3 из сохранённых ответов.

| Конфигурация | Проектная Execution Accuracy | Текстовый Exact Match | Valid SQL | Stop Rate |
|---|---:|---:|---:|---:|
| [Base zero-shot](base_0.5b_zero_eval_metrics.json) | 22,08% (474/2147) | 9,87% | 64,69% | 75,87% |
| [Base 4-shot](base_0.5b_few_eval_metrics.json) | 18,07% (388/2147) | 8,80% | 46,11% | 1,35% |
| [Исходный SFT zero-shot](sft_0.5b_zero_eval_metrics.json) | 50,95% (1094/2147) | 36,47% | 75,55% | 100,00% |
| [SFT на расширенном датасете, zero-shot](sft_0.5b_expanded_zero_eval_metrics.json) | 46,30% (994/2147) | 32,60% | 70,84% | 99,58% |
| [Instruct zero-shot](instruct_0.5b_zero_eval_metrics.json) | 24,64% (529/2147) | 4,94% | 52,54% | 80,16% |

На первых 16 вопросах два независимых запуска Base zero-shot и исходного SFT совпали с соответствующими строками полных прогонов. SQL всех 2147 ответов каждого варианта проверен повторно.

У Base 4-shot запрос `test_0447` выполнился в пределах лимита в Colab и превысил его при пересчёте на Windows. Это изменило Valid SQL на один пример.

[Официальные метрики и методика](../../docs/evaluation.md) · [JSON аудита](../evaluation_review.json) · [Colab](../../notebooks/colab_spider_test.ipynb)
