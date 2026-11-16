"""Этап 5. SFT (роль 3): SFTTrainer из TRL + LoRA.

Запуск:
    python -m src.train --config configs/sft_0.5b.yaml

На T4: bf16=False, fp16=True.
Validation loss base-модели: trainer.evaluate() до trainer.train().
"""
