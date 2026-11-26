# Spider Expanded — reconstructed training dataset

This directory tracks provenance of the reconstructed Spider 1.0 Expanded dataset used for the Qwen2.5 0.5B/1.5B/3B SFT experiments.

**Status: metadata published; dataset ZIP not yet committed.** The binary archive `spider_expanded_restored.zip` must be added separately before treating this directory as a download source.

## Expected archive

- Name: `spider_expanded_restored.zip`
- ZIP SHA-256: `e73377bf3140867131efe475ede4ab7b2d0d001b61b0ab494752da5f6072cb2d`
- Contents: `train.jsonl` (8222), `val.jsonl` (434), `val_used_300.jsonl` (300), `test.jsonl` (1034, **Spider dev**, NOT Spider 1.0 final test), `fewshot.jsonl` (4), `stats.json`, `manifest.json`.
- The **Spider 1.0 test set of 2147 questions is separate** and is not included in this ZIP.

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
