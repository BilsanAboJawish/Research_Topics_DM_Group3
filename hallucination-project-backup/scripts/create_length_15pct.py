import json
from transformers import AutoTokenizer

SOURCE = (
    "Superfiltering/data/data_with_ifd/"
    "alpaca_gpt4_data_gpt2_data.json"
)

OUTPUT = "data/alpaca_gpt4_length_15pct.json"

N = 7800
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

with open(SOURCE, encoding="utf-8") as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

rows = []

for i, x in enumerate(data):
    output = str(x.get("output", ""))

    n_tokens = len(
        tokenizer(
            output,
            add_special_tokens=False
        )["input_ids"]
    )

    rows.append((n_tokens, i, x))

# Longest first; original index gives deterministic tie breaking.
rows.sort(key=lambda z: (-z[0], z[1]))

selected = [x for _, _, x in rows[:N]]

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)

lengths = [n for n, _, _ in rows[:N]]

print("Saved:", OUTPUT)
print("N:", len(selected))
print("Mean completion tokens:", sum(lengths) / len(lengths))
print("Min selected tokens:", min(lengths))
print("Max selected tokens:", max(lengths))
