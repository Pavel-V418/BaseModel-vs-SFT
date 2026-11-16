"""Этап 5. SFT: SFTTrainer из TRL + LoRA / QLoRA / полный fine-tuning.

Запуск:
    python -m src.train --config configs/sft_0.5b.yaml
    python -m src.train --config configs/sft_0.5b.yaml --set train_size=2000 --set lora.r=32
    python -m src.train --config configs/sft_0.5b.yaml --smoke     # 10 шагов на игрушечных данных

Что делает:
    1. Загружает base-модель и prompt/completion-данные (data/*.jsonl).
    2. Считает validation loss ДО обучения (это loss base-модели:
       у свежего LoRA-адаптера матрица B = 0, поэтому модель = base).
    3. Обучает (loss только по completion, т.е. по SQL).
    4. Считает validation loss ПОСЛЕ обучения.
    5. Сохраняет адаптер в outputs/<run_name>/final и метрики в results/.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

from datasets import load_dataset

from src.utils import (
    count_parameters,
    env_info,
    gpu_supports_bf16,
    load_config,
    load_model,
    load_tokenizer,
    save_json,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
SMOKE_TRAIN = ROOT / "tests" / "fixtures" / "tiny_train.jsonl"
SMOKE_VAL = ROOT / "tests" / "fixtures" / "tiny_val.jsonl"


# ---------------------------------------------------------------- данные

def load_data(cfg: dict):
    files = {"train": cfg["data"]["train"], "validation": cfg["data"]["val"]}
    ds = load_dataset("json", data_files=files)
    train, val = ds["train"], ds["validation"]

    seed = cfg["seed"]
    if cfg.get("train_size"):
        n = min(cfg["train_size"], len(train))
        train = train.shuffle(seed=seed).select(range(n))
    if cfg.get("val_size"):
        n = min(cfg["val_size"], len(val))
        val = val.shuffle(seed=seed).select(range(n))

    # SFTTrainer понимает формат prompt/completion; остальные поля ему не нужны
    keep = {"prompt", "completion"}
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    val = val.remove_columns([c for c in val.column_names if c not in keep])
    return train, val


# ---------------------------------------------------------------- конфиг TRL

def build_sft_config(cfg: dict, out_dir: Path):
    from trl import SFTConfig

    t = cfg["training"]
    use_bf16 = gpu_supports_bf16()
    report_to = cfg.get("report_to", "none")
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg.get("wandb_project", "base-vs-sft"))

    return SFTConfig(
        output_dir=str(out_dir),
        run_name=cfg["run_name"],
        seed=cfg["seed"],
        # --- данные
        max_length=t.get("max_length", 1024),
        packing=False,
        completion_only_loss=cfg.get("completion_only_loss", True),
        # --- оптимизация
        learning_rate=float(t["learning_rate"]),
        num_train_epochs=t.get("num_train_epochs", 2),
        max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_steps=t.get("warmup_steps", 50),
        weight_decay=t.get("weight_decay", 0.0),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        # --- точность: T4 -> fp16, A100/H100 -> bf16
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # --- валидация и чекпоинты (лучший по eval_loss)
        eval_strategy="steps",
        eval_steps=t.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=t.get("eval_steps", 100),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # --- логи
        logging_steps=t.get("logging_steps", 10),
        report_to=report_to,
    )


def build_peft_config(cfg: dict):
    if cfg["method"] == "full":
        return None
    from peft import LoraConfig

    lc = cfg["lora"]
    return LoraConfig(
        r=lc["r"],
        lora_alpha=lc.get("alpha", 2 * lc["r"]),
        lora_dropout=lc.get("dropout", 0.05),
        target_modules=lc.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )


# ---------------------------------------------------------------- проверки

def sanity_check(trainer, tokenizer) -> None:
    """Проверяем, что completion заканчивается EOS — иначе модель не научится
    останавливаться после SQL, и метрики упадут из-за «болтовни»."""
    sample = trainer.train_dataset[0]
    ids = sample["input_ids"]
    print("\n=== Пример после токенизации ===")
    print(tokenizer.decode(ids))
    if "completion_mask" in sample:
        n = sum(sample["completion_mask"])
        print(f"=== Токенов, по которым считается loss: {n} из {len(ids)}")
        if n == 0:
            raise RuntimeError("completion_mask пустой: loss не считается ни по одному токену")
    if ids[-1] != tokenizer.eos_token_id:
        print(
            f"!!! ВНИМАНИЕ: последний токен не EOS ({tokenizer.eos_token!r}). "
            "Модель может не научиться останавливаться."
        )
    print()


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="переопределение: key.sub=value")
    ap.add_argument("--smoke", action="store_true", help="10 шагов на tests/fixtures, без W&B")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg["run_name"] = "smoke_" + cfg["run_name"]
        cfg["data"] = {"train": str(SMOKE_TRAIN), "val": str(SMOKE_VAL)}
        cfg["train_size"] = cfg["val_size"] = None
        cfg["report_to"] = "none"
        cfg["training"].update(max_steps=10, eval_steps=5, logging_steps=1,
                               warmup_steps=0, per_device_train_batch_size=2,
                               per_device_eval_batch_size=2, gradient_accumulation_steps=1)

    set_seed(cfg["seed"])
    run_name = cfg["run_name"]
    out_dir = ROOT / "outputs" / run_name
    res_dir = ROOT / "results"
    print(f"Run: {run_name}\nEnv: {env_info()}")

    # --- модель и данные
    tokenizer = load_tokenizer(cfg["model"])
    model = load_model(cfg["model"], method=cfg["method"])
    if cfg["method"] != "full" and cfg["training"].get("gradient_checkpointing", True):
        model.enable_input_require_grads()  # нужно для LoRA + gradient checkpointing
    train_ds, val_ds = load_data(cfg)
    print(f"train: {len(train_ds)}, val: {len(val_ds)}")

    # --- тренер
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(cfg, out_dir),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=build_peft_config(cfg),
    )
    params = count_parameters(trainer.model)
    print(f"Параметры: всего {params['total']:,}, обучаемых {params['trainable']:,} "
          f"({100 * params['trainable'] / params['total']:.2f}%)")
    sanity_check(trainer, tokenizer)

    # --- 1. validation loss ДО обучения (= base-модель)
    before = trainer.evaluate()
    print(f"[base] eval_loss = {before['eval_loss']:.4f}")

    # --- 2. обучение
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0

    # --- 3. validation loss ПОСЛЕ (лучший чекпоинт)
    after = trainer.evaluate()
    print(f"[sft]  eval_loss = {after['eval_loss']:.4f}")

    # --- 4. сохранение
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))  # для LoRA сохраняется только адаптер
    tokenizer.save_pretrained(str(final_dir))

    def ppl(x: float) -> float:
        return math.exp(x) if x < 50 else float("inf")

    summary = {
        "run_name": run_name,
        "config": cfg,
        "env": env_info(),
        "params": params,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "eval_before": {**before, "perplexity": ppl(before["eval_loss"])},
        "eval_after": {**after, "perplexity": ppl(after["eval_loss"])},
        "train_metrics": train_result.metrics,
        "train_time_sec": round(train_time, 1),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "adapter_dir": str(final_dir),
    }
    save_json(summary, res_dir / f"{run_name}_train.json")
    # полная история логов — чтобы строить кривые loss без W&B
    save_json(trainer.state.log_history, res_dir / f"{run_name}_log_history.json")

    if cfg.get("hub_repo") and not args.smoke:
        trainer.model.push_to_hub(cfg["hub_repo"])
        tokenizer.push_to_hub(cfg["hub_repo"])

    print(f"\nГотово. eval_loss: {before['eval_loss']:.4f} -> {after['eval_loss']:.4f}")
    print(f"Адаптер: {final_dir}\nМетрики: {res_dir / (run_name + '_train.json')}")


if __name__ == "__main__":
    main()
