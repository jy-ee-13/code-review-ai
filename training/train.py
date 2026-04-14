# training/train.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from severity_dataset import EXPANDED_DATA
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter
import random

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"      # ~2.7B — fits in CPU/low VRAM
OUTPUT_DIR = "./severity_model"
LABEL2ID = {"critical": 0, "warning": 1, "suggestion": 2}
ID2LABEL = {0: "critical", 1: "warning", 2: "suggestion"}

# ── Prepare dataset ───────────────────────────────────────────────────────────
random.shuffle(EXPANDED_DATA)
texts  = [item[0] for item in EXPANDED_DATA]
labels = [LABEL2ID[item[1]] for item in EXPANDED_DATA]

split = int(len(texts) * 0.85)
train_dataset = Dataset.from_dict({"text": texts[:split],  "label": labels[:split]})
eval_dataset  = Dataset.from_dict({"text": texts[split:],  "label": labels[split:]})

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
print(f"Label distribution (train): {Counter(labels[:split])}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset  = eval_dataset.map(tokenize,  batched=True)

# ── Model + LoRA ──────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.config.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                    # LoRA rank — higher = more capacity, more memory
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(
        labels, preds,
        target_names=["critical", "warning", "suggestion"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "f1_critical":   report["critical"]["f1-score"],
        "f1_warning":    report["warning"]["f1-score"],
        "f1_suggestion": report["suggestion"]["f1-score"],
        "f1_weighted":   report["weighted avg"]["f1-score"],
    }

# ── Training ──────────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,      # Reduce to 1 for CPU safety
    gradient_accumulation_steps=8,     # Keep effective batch size at 8
    gradient_checkpointing=True,       # <--- CRITICAL: Saves massive RAM
    learning_rate=2e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_steps=1,
    fp16=False,                        # Must be False for CPU
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

print("\nStarting fine-tuning...")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")
print("\nFinal eval metrics:")
print(trainer.evaluate())