import json
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

DATA_PATH = "data/alpaca_gpt4_ifd_10pct.json"

OUTPUT_DIR = "models/mistral_lora_10pct_fixedsteps"


print("Loading selected 10% Alpaca-GPT4 data...")

with open(DATA_PATH) as f:
    raw_data = json.load(f)

print("Number of training examples:", len(raw_data))


def convert_example(example):

    instruction = example["instruction"].strip()
    input_text = example.get("input", "").strip()
    output_text = example["output"].strip()

    if input_text:
        user_content = (
            instruction
            + "\n\nInput:\n"
            + input_text
        )
    else:
        user_content = instruction

    return {
        "prompt": [
            {
                "role": "user",
                "content": user_content
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": output_text
            }
        ]
    }


dataset = Dataset.from_list(
    [convert_example(x) for x in raw_data]
)

print(dataset)

print("\nExample training item:")
print(dataset[0])


print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("\nLoading Mistral...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)

model.config.use_cache = False


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,

    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    bias="none"
)


training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    # Keep the first experiment manageable
    num_train_epochs=1,

    # Fixed optimizer-update control.
    # max_steps overrides num_train_epochs.
    max_steps=975,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    max_length=1024,

    bf16=True,

    gradient_checkpointing=True,

    logging_steps=10,

    save_strategy="epoch",

    seed=42,

    completion_only_loss=True,

    report_to="none"
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config
)


print("\nStarting 10% LoRA fine-tuning...")

trainer.train()


print("\nSaving LoRA adapter...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


print("\nTraining completed.")
print("Saved model to:", OUTPUT_DIR)

