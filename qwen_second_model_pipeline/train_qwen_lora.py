#!/usr/bin/env python3
import argparse, json, math, os, sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def load_any(path):
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in [".jsonl", ".json"]:
        return load_dataset("json", data_files=path, split="train")
    if suffix == ".csv":
        return load_dataset("csv", data_files=path, split="train")
    raise ValueError(f"Unsupported training file: {path}")

def infer_fields(cols):
    # Alpaca-GPT4 is normally instruction/input/output.
    out_candidates = ["output", "response", "answer", "completion", "target"]
    ins_candidates = ["instruction", "prompt", "question", "query"]
    out = next((x for x in out_candidates if x in cols), None)
    ins = next((x for x in ins_candidates if x in cols), None)
    if out is None or ins is None:
        raise ValueError(
            f"Could not infer fields. Columns={cols}. "
            "Expected instruction/prompt + output/response."
        )
    extra = "input" if "input" in cols and ins != "input" else None
    return ins, extra, out

class CausalCompletionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []

        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--model-name", default=MODEL_NAME)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    print("MODEL:", args.model_name)
    print("TRAIN FILE:", args.train_file)
    print("OUTPUT:", args.output_dir)
    print("SEED:", args.seed)
    print("torch:", torch.__version__)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = load_any(args.train_file)
    print("Rows:", len(ds))
    if len(ds) != 7800:
        print(
            f"WARNING: expected 7,800 rows for the 15% subset, found {len(ds)}.",
            file=sys.stderr,
        )

    ins_field, extra_field, out_field = infer_fields(ds.column_names)
    print("Fields:", ins_field, extra_field, out_field)

    def preprocess(ex):
        instruction = str(ex.get(ins_field, "") or "").strip()
        if extra_field:
            extra = str(ex.get(extra_field, "") or "").strip()
            if extra:
                instruction = instruction + "\n\n" + extra
        answer = str(ex.get(out_field, "") or "").strip()

        user_messages = [{"role": "user", "content": instruction}]
        full_messages = user_messages + [{"role": "assistant", "content": answer}]

        # Native Qwen chat template. This is the key difference from Mistral.
        prompt_text = tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        p = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )
        f = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )

        labels = list(f["input_ids"])
        prompt_len = min(len(p["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": f["input_ids"],
            "attention_mask": f["attention_mask"],
            "labels": labels,
            "has_target": int(any(x != -100 for x in labels)),
        }

    tokenized = ds.map(
        preprocess,
        remove_columns=ds.column_names,
        desc="Applying Qwen chat template",
    )
    before = len(tokenized)
    tokenized = tokenized.filter(lambda x: x["has_target"] == 1)
    tokenized = tokenized.remove_columns(["has_target"])
    print(f"Kept {len(tokenized)}/{before} examples after truncation check.")

    # Match the Mistral study as closely as practical.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenized,
        data_collator=CausalCompletionCollator(tokenizer),
    )
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(Path(args.output_dir) / "run_info.json", "w") as f:
        json.dump(
            {
                "base_model": args.model_name,
                "seed": args.seed,
                "train_file": str(Path(args.train_file).resolve()),
                "n_train": len(ds),
                "max_length": args.max_length,
                "epochs": 1.0,
                "lr": 2e-4,
                "per_device_batch": 1,
                "grad_accum": 8,
                "effective_batch": 8,
                "bf16": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "lora_targets": ["q_proj", "v_proj"],
                "completion_only_loss": True,
            },
            f,
            indent=2,
        )
    print("DONE:", args.output_dir)

if __name__ == "__main__":
    main()
