# Spider 1.0 test

All five variants use the 2,147 released test questions and 40 test SQLite databases. The four few-shot demonstrations come from training databases. Instruct wraps the same schema and question in its chat template.

| Variant | Execution accuracy | Exact match | Valid SQL | Stop rate |
|---|---:|---:|---:|---:|
| [Qwen2.5-0.5B base, zero-shot](base_0.5b_zero_eval_metrics.json) | 22.08% (474/2147) | 9.87% | 64.69% | 75.87% |
| [Qwen2.5-0.5B base, 4-shot](base_0.5b_few_eval_metrics.json) | 18.07% (388/2147) | 8.80% | 46.16% | 1.35% |

Completed variants: 2/5.

SQL answers are recorded in the adjacent JSONL files. Each metrics JSON contains the breakdown by difficulty, model revision, data hashes and runtime versions. The earlier [Spider dev results](../README.md) use a separate 1,034-question split.

The table reports the Colab scores. For base 4-shot, SQLite 3.45.1 on Colab executed prediction `test_0447` in 4.08 and 4.42 seconds on repeat runs. On Windows with SQLite 3.50.4 it reached the five-second timeout. This changes valid SQL by one example and preserves execution accuracy.

[Colab notebook](../../notebooks/colab_spider_test.ipynb) · [Official Spider release](https://yale-lily.github.io/spider/)
