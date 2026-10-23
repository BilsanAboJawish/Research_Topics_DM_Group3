import json
import re
import csv
from pathlib import Path
from statistics import mean, median


# ---------------------------------------------------------
# Files
# ---------------------------------------------------------

FILES = {}

for pct in ["05", "10", "15", "20", "25"]:
    FILES[f"IFD_{pct}"] = f"data/alpaca_gpt4_ifd_{pct}pct.json"
    FILES[f"RANDOM_{pct}"] = f"data/alpaca_gpt4_random_{pct}pct_seed42.json"


# ---------------------------------------------------------
# Lexical indicators
#
# These are intentionally transparent/simple.
# We will ALSO manually inspect examples, because lexical
# matching alone is not a perfect refusal classifier.
# ---------------------------------------------------------

REFUSAL_PATTERNS = [
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bi am unable\b",
    r"\bi'm unable\b",
    r"\bi cannot provide\b",
    r"\bi can't provide\b",
    r"\bi cannot assist\b",
    r"\bi can't assist\b",
    r"\bi cannot help\b",
    r"\bi can't help\b",
    r"\bi do not have access\b",
    r"\bi don't have access\b",
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\bunable to provide\b",
    r"\bnot able to provide\b",
    r"\bnot enough information\b",
    r"\binsufficient information\b",
    r"\bthere is no way to\b",
    r"\bas an ai\b",
    r"\bas a language model\b",
]

HEDGE_PATTERNS = [
    r"\bi am not sure\b",
    r"\bi'm not sure\b",
    r"\bit is unclear\b",
    r"\bit's unclear\b",
    r"\bmay be\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bit depends\b",
    r"\bI believe\b",
    r"\bit seems\b",
    r"\bit appears\b",
]


def matches_any(text, patterns):
    text = text.lower()
    return any(re.search(pattern, text, flags=re.IGNORECASE)
               for pattern in patterns)


def get_text(example):
    """
    Alpaca-GPT4 response field.
    """
    return str(example.get("output", "")).strip()


def analyse_file(label, path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    refusal_examples = []
    hedge_examples = []

    lengths_words = []
    lengths_chars = []

    refusal_count = 0
    hedge_count = 0
    either_count = 0

    for i, example in enumerate(data):
        response = get_text(example)

        refusal = matches_any(response, REFUSAL_PATTERNS)
        hedge = matches_any(response, HEDGE_PATTERNS)

        if refusal:
            refusal_count += 1
            refusal_examples.append({
                "index": i,
                "instruction": example.get("instruction", ""),
                "input": example.get("input", ""),
                "output": response,
            })

        if hedge:
            hedge_count += 1
            hedge_examples.append({
                "index": i,
                "instruction": example.get("instruction", ""),
                "input": example.get("input", ""),
                "output": response,
            })

        if refusal or hedge:
            either_count += 1

        lengths_words.append(len(response.split()))
        lengths_chars.append(len(response))

    n = len(data)

    result = {
        "condition": label,
        "n_examples": n,

        "refusal_count": refusal_count,
        "refusal_pct": 100 * refusal_count / n,

        "hedge_count": hedge_count,
        "hedge_pct": 100 * hedge_count / n,

        "refusal_or_hedge_count": either_count,
        "refusal_or_hedge_pct": 100 * either_count / n,

        "mean_response_words": mean(lengths_words),
        "median_response_words": median(lengths_words),

        "mean_response_chars": mean(lengths_chars),
        "median_response_chars": median(lengths_chars),
    }

    return result, refusal_examples, hedge_examples


def main():

    Path("results/training_subset_analysis").mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    for label, path in FILES.items():

        print("\n" + "=" * 70)
        print(label)
        print(path)
        print("=" * 70)

        result, refusals, hedges = analyse_file(label, path)

        all_results.append(result)

        for key, value in result.items():
            print(f"{key}: {value}")

        # Save matching examples for manual inspection
        with open(
            f"results/training_subset_analysis/{label}_refusal_examples.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(refusals, f, ensure_ascii=False, indent=2)

        with open(
            f"results/training_subset_analysis/{label}_hedge_examples.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(hedges, f, ensure_ascii=False, indent=2)

    # Save summary CSV
    output_csv = "results/training_subset_analysis/summary.csv"

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    header = (
        f"{'Condition':<12}"
        f"{'N':>8}"
        f"{'Refusal %':>12}"
        f"{'Hedge %':>12}"
        f"{'Either %':>12}"
        f"{'Mean words':>13}"
    )

    print(header)
    print("-" * len(header))

    for r in all_results:
        print(
            f"{r['condition']:<12}"
            f"{r['n_examples']:>8}"
            f"{r['refusal_pct']:>12.3f}"
            f"{r['hedge_pct']:>12.3f}"
            f"{r['refusal_or_hedge_pct']:>12.3f}"
            f"{r['mean_response_words']:>13.2f}"
        )

    print("\nSaved:")
    print(output_csv)


if __name__ == "__main__":
    main()

