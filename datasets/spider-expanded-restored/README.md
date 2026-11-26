# Spider Expanded — reconstructed training dataset

This directory tracks provenance of the reconstructed Spider 1.0 Expanded dataset used for the Qwen2.5 0.5B/1.5B/3B SFT experiments.

## Downloads

### Original reconstructed splits

- [Download `spider_expanded_restored.zip`](https://raw.githubusercontent.com/Pavel-V418/BaseModel-vs-SFT/refs/heads/dataset/spider-expanded-restored/datasets/spider-expanded-restored/spider_expanded_restored.zip) (592,428 bytes)
- ZIP SHA-256: `e73377bf3140867131efe475ede4ab7b2d0d001b61b0ab494752da5f6072cb2d`
- Contents: `train.jsonl` (8,222), `val.jsonl` (434), `val_used_300.jsonl` (300), `test.jsonl` (1,034 from **Spider dev**), `fewshot.jsonl` (4), `stats.json`, and `manifest.json`.
- The official Spider 1.0 final test set (2,147 examples) is separate. See [manifest.json](manifest.json) for the checksums of the original split files.

### Full reconstructed training pool

- [Download `spider_expanded_train_full.zip`](https://raw.githubusercontent.com/Pavel-V418/BaseModel-vs-SFT/refs/heads/dataset/spider-expanded-restored/datasets/spider-expanded-restored/spider_expanded_train_full.zip) (484,127 bytes)
- ZIP SHA-256: `4bddf8c97a68f76a66a9327c39077015406784611079928966ade6b10c178443`
- Contains `train_full.jsonl` (14,478,737 bytes, 8,656 examples), formed by appending `val.jsonl` to `train.jsonl` in their original row order.
- JSONL SHA-256: `6767e97ee9ba0a878c89334cdd859a09f90c449b06e705d3b3f9506f340e40cc`
- The source is the restored, preprocessed training pool. Spider dev and official test stay separate. The 300 selected validation examples are already part of the 434 validation examples. Few-shot examples are provided with the original archive.
- Use the original splits to reproduce the experiment's validation. Create a fresh holdout if you train from the full pool and need a validation set.

This dataset was reconstructed with `val_size=400`, `seed=42`, `include_others=true`, and 300 selected validation examples. Byte-for-byte identity to the original 2026 historic training files was not independently proven; see `manifest.json`.

## Reproduction from official Spider

```bash
pip install gdown transformers
bash scripts/download_spider.sh
python -m src.data --spider_dir data/spider --out_dir data/spider_others --val_size 400 --seed 42 --include_others
```

Use the pinned preprocessing commit from `manifest.json` when exact reproduction is required; current `master` might produce differences in record IDs or normalization.

Raw, *not additionally train/val split* Spider comes from `train_spider.json` plus `train_others.json` in the official downloaded Spider archive. Do not mix official dev/test into training.

## Provenance and licensing

Original dataset: [Spider 1.0, Yale LILY](https://yale-lily.github.io/spider) (CC BY-SA 4.0). This project-created transformed data preserves attribution and should retain the applicable share-alike licensing and credit to original authors. See the original dataset website for citation details. SQLite databases are **not included** here.
