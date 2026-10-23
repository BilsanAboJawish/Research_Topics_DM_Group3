import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

UA_PATH = "UAQ_Fact/data/1.0/task1/inter_ua_en.json"
AB_PATH = "UAQ_Fact/data/1.0/task1/inter_ab_en.json"

N = 100

print("Loading data...")

with open(UA_PATH) as f:
    unanswerable = json.load(f)[:N]

with open(AB_PATH) as f:
    answerable = json.load(f)[:N]

print("Unanswerable:", len(unanswerable))
print("Answerable:", len(answerable))

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()


def generate_answer(question):
    messages = [
        {
            "role": "user",
            "content": (
                "Answer the following factual question briefly. "
                "If the question cannot be answered or no such answer exists, "
                "say 'I don't know'.\n\n"
                f"Question: {question}"
            )
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()


results = []

for category, examples in [
    ("unanswerable", unanswerable),
    ("answerable", answerable)
]:
    print(f"\nEvaluating {category} questions...")

    for i, example in enumerate(examples):

        response = generate_answer(example["question"])

        results.append({
            "qid": example["qid"],
            "category": category,
            "question": example["question"],
            "reference_answer": example["answer"],
            "model_response": response
        })

        if i < 5:
            print("\nQuestion:", example["question"])
            print("Response:", response)


output_path = "results/mistral_baseline_uaqfact_200.json"

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print("\nFinished.")
print("Saved:", output_path)
