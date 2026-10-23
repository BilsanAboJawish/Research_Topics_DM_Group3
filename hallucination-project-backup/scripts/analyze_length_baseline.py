import json
import re
from collections import Counter
from statistics import mean, median
from transformers import AutoTokenizer


FILES = {
    "IFD15": "data/alpaca_gpt4_ifd_15pct.json",
    "Random15": "data/alpaca_gpt4_random_15pct_seed42.json",
    "Length15": "data/alpaca_gpt4_length_15pct.json",
}

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def key(x):
    return (
        str(x.get("instruction", "")).strip(),
        str(x.get("input", "")).strip(),
        str(x.get("output", "")).strip(),
    )


def wc(text):
    return len(str(text).split())


def classify_task(instruction):
    t = str(instruction).lower().strip()

    if re.search(r"\btranslate\b|\btranslation\b", t):
        return "translation"

    if re.search(r"\bsummarize\b|\bsummarise\b|\bsummary\b", t):
        return "summarization"

    if re.search(
        r"\brewrite\b|\brephrase\b|\bparaphrase\b|\bedit\b"
        r"|\bproofread\b|\bcorrect the grammar\b",
        t
    ):
        return "rewrite_edit"

    if re.search(
        r"\bpython\b|\bjava\b|\bc\+\+\b|\bjavascript\b|\bsql\b"
        r"|\bcode\b|\bprogram\b|\bfunction\b|\balgorithm\b",
        t
    ):
        return "coding"

    if re.search(
        r"\bcalculate\b|\bsolve\b|\bequation\b|\bderive\b"
        r"|\bproof\b|\bprobability\b|\bmathemat",
        t
    ):
        return "math_reasoning"

    if re.search(
        r"\bclassify\b|\bcategorize\b|\bcategorise\b"
        r"|\bextract\b|\blabel\b|\bidentify\b",
        t
    ):
        return "classification_extraction"

    if re.search(
        r"\bwrite\b|\bcreate\b|\bgenerate\b|\bcompose\b"
        r"|\bstory\b|\bpoem\b|\bessay\b|\bemail\b",
        t
    ):
        return "generation"

    if (
        t.startswith((
            "what ", "who ", "when ", "where ", "why ",
            "how ", "which ", "is ", "are ", "does ", "do "
        ))
        or "explain" in t
        or "describe" in t
    ):
        return "qa_explanation"

    return "other"


data = {name: load(path) for name, path in FILES.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("=" * 90)
print("15% BASELINE COMPOSITION")
print("=" * 90)

summaries = {}

for name, rows in data.items():

    words = [wc(x["output"]) for x in rows]

    tokens = [
        len(
            tokenizer(
                str(x["output"]),
                add_special_tokens=False
            )["input_ids"]
        )
        for x in rows
    ]

    tasks = Counter(
        classify_task(x["instruction"])
        for x in rows
    )

    summaries[name] = {
        "words": words,
        "tokens": tokens,
        "tasks": tasks,
    }

    print(f"\n{name}")
    print("-" * 50)
    print("N:", len(rows))
    print(f"Mean words:   {mean(words):.2f}")
    print(f"Median words: {median(words):.2f}")
    print(f"Mean tokens:  {mean(tokens):.2f}")
    print(f"<=10 words:   {100*sum(x<=10 for x in words)/len(words):.2f}%")
    print(f">=200 words:  {100*sum(x>=200 for x in words)/len(words):.2f}%")

    print("\nTask composition:")
    for task, n in sorted(tasks.items()):
        print(
            f"  {task:<28}"
            f"{n:>5} "
            f"({100*n/len(rows):6.2f}%)"
        )


print("\n")
print("=" * 90)
print("SUBSET OVERLAP")
print("=" * 90)

sets = {
    name: {key(x) for x in rows}
    for name, rows in data.items()
}

pairs = [
    ("Length15", "IFD15"),
    ("Length15", "Random15"),
    ("IFD15", "Random15"),
]

for a, b in pairs:

    overlap = len(
        sets[a] & sets[b]
    )

    pct = (
        100 * overlap / len(sets[a])
    )

    jaccard = (
        100 * overlap
        / len(sets[a] | sets[b])
    )

    print(
        f"{a:<10} vs {b:<10}: "
        f"overlap={overlap:>5} "
        f"({pct:6.2f}% of {a}), "
        f"Jaccard={jaccard:6.2f}%"
    )


print("\n")
print("=" * 90)
print("TASK-COMPOSITION DIFFERENCES: LENGTH15 - IFD15")
print("=" * 90)

all_tasks = sorted(
    set(
        summaries["Length15"]["tasks"]
    )
    |
    set(
        summaries["IFD15"]["tasks"]
    )
)

for task in all_tasks:

    lp = (
        100
        * summaries["Length15"]["tasks"].get(task, 0)
        / len(data["Length15"])
    )

    ip = (
        100
        * summaries["IFD15"]["tasks"].get(task, 0)
        / len(data["IFD15"])
    )

    print(
        f"{task:<28}"
        f" Length={lp:6.2f}%"
        f" IFD={ip:6.2f}%"
        f" delta={lp-ip:+6.2f} pp"
    )
