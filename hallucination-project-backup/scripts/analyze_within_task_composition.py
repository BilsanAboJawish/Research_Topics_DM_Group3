import json
import re
from collections import defaultdict
from statistics import mean, median


BUDGETS = ["05", "10", "15", "20", "25"]


def classify_task(instruction):
    t = instruction.lower().strip()

    if re.search(r"\btranslate\b|\btranslation\b", t):
        return "translation"

    if re.search(r"\bsummarize\b|\bsummarise\b|\bsummary\b", t):
        return "summarization"

    if re.search(
        r"\brewrite\b|\brephrase\b|\bparaphrase\b"
        r"|\bedit\b|\bproofread\b|\bcorrect the grammar\b",
        t
    ):
        return "rewrite_edit"

    if re.search(
        r"\bpython\b|\bjava\b|\bc\+\+\b|\bjavascript\b"
        r"|\bsql\b|\bcode\b|\bprogram\b|\bfunction\b"
        r"|\balgorithm\b",
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
            "what ", "who ", "when ", "where ",
            "why ", "how ", "which ", "is ",
            "are ", "does ", "do "
        ))
        or "explain" in t
        or "describe" in t
    ):
        return "qa_explanation"

    return "other"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wc(text):
    return len(str(text).split())


rows = defaultdict(list)

for budget in BUDGETS:

    files = {
        "IFD": f"data/alpaca_gpt4_ifd_{budget}pct.json",
        "Random": f"data/alpaca_gpt4_random_{budget}pct_seed42.json",
    }

    for method, path in files.items():

        data = load(path)

        for x in data:
            category = classify_task(
                str(x.get("instruction", ""))
            )

            rows[
                (budget, method, category)
            ].append(
                wc(x.get("output", ""))
            )


categories = sorted(
    set(k[2] for k in rows)
)


print()
print("=" * 110)
print("WITHIN-TASK RESPONSE LENGTH: IFD vs RANDOM")
print("=" * 110)

for budget in BUDGETS:

    print(f"\n========== {budget}% ==========")

    print(
        f"{'Category':<28}"
        f"{'IFD n':>8}"
        f"{'Rnd n':>8}"
        f"{'IFD mean':>11}"
        f"{'Rnd mean':>11}"
        f"{'Ratio':>9}"
        f"{'IFD med':>10}"
        f"{'Rnd med':>10}"
    )

    for cat in categories:

        a = rows.get(
            (budget, "IFD", cat), []
        )

        b = rows.get(
            (budget, "Random", cat), []
        )

        if not a or not b:
            continue

        ratio = mean(a) / mean(b)

        print(
            f"{cat:<28}"
            f"{len(a):>8}"
            f"{len(b):>8}"
            f"{mean(a):>11.1f}"
            f"{mean(b):>11.1f}"
            f"{ratio:>9.2f}"
            f"{median(a):>10.1f}"
            f"{median(b):>10.1f}"
        )
