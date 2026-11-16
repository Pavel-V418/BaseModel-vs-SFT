"""Общие утилиты: seed, конфиги, загрузка модели и токенизатора.

Используются и в train.py, и в evaluate.py — чтобы base и SFT
загружались одинаково.
"""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------- seed

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- конфиги

def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_value(raw: str) -> Any:
    """'1e-4' -> 1e-4, 'true' -> True, 'null' -> None, иначе строка."""
    return yaml.safe_load(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Читает YAML и применяет переопределения вида 'training.learning_rate=2e-4'."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node: dict = {}
        cur = node
        parts = key.split(".")
        for p in parts[:-1]:
            cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = _parse_value(raw)
        cfg = _deep_update(cfg, node)
    return cfg


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- GPU

def gpu_supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def compute_dtype() -> txorch.dtype:
    """T4 (Kaggle/Colab) не умеет bf16 -> fp16. A100/H100 -> bf16."""
    return torch.bfloat16 if gpu_supports_bf16() else torch.float16


# ---------------------------------------------------------------- модель

def _dtype_kwarg() -> str:
    """В новых transformers аргумент называется dtype, в старых — torch_dtype."""
    import transformers
    from packaging.version import Version

    return "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name: str, method: str = "lora", adapter: str | None = None):
    """Загружает модель.

    method:
      "lora"  — веса в fp16/bf16, адаптеры LoRA обучаются (PEFT держит их в fp32);
      "qlora" — веса в 4 бита (bitsandbytes), для 3B+ на T4;
      "full"  — полный fine-tuning: веса в fp32, обучение через AMP (fp16/bf16).
               Грузить fp16-веса для полного FT нельзя: AMP не умеет
               «unscale» fp16-градиенты.
    adapter: путь к обученному LoRA-адаптеру (для оценки SFT-модели).
    """
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"device_map": "auto"}
    dkey = _dtype_kwarg()
    if method == "full":
        kwargs[dkey] = torch.float32
    elif method == "qlora":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
        )
    elif method == "lora":
        kwargs[dkey] = compute_dtype()
    else:
        raise ValueError(f"Неизвестный method: {method}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model


def count_parameters(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["bf16"] = gpu_supports_bf16()
    for lib in ("torch", "transformers", "trl", "peft", "datasets"):
        try:
            info[lib] = __import__(lib).__version__
        except Exception:  # noqa: BLE001
            info[lib] = None
    info["kaggle"] = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    return info
