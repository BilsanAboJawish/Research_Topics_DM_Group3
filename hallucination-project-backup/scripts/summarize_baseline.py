import csv
from collections import Counter

PATH = "results/baseline_manual_labels.csv"

rows = []

with open(PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("Total rows:", len(rows))

unlabeled = [
    r for r in rows
    if not r["manual_label"].strip()
]

if unlabeled:
    print("\nERROR:")
    print("There are still", len(unlabeled), "unlabelled responses.")
    print("Finish labeling before calculating final results.")
    raise SystemExit


unanswerable = [
    r for r in rows
    if r["category"] == "unanswerable"
]

answerable = [
    r for r in rows
    if r["category"] == "answerable"
]


ua_counts = Counter(
    r["manual_label"] for r in unanswerable
)

ab_counts = Counter(
    r["manual_label"] for r in answerable
)


n_ua = len(unanswerable)
n_ab = len(answerable)

fabrications = ua_counts["fabrication"]
abstentions = ua_counts["abstention"]
ua_ambiguous = ua_counts["ambiguous"]

correct = ab_counts["correct"]
incorrect = ab_counts["incorrect"]
ab_ambiguous = ab_counts["ambiguous"]


print("\n" + "=" * 60)
print("FINAL MISTRAL BASELINE EVALUATION")
print("=" * 60)

print("\nUNANSWERABLE QUESTIONS")
print("Total:", n_ua)

print(
    "Abstention:",
    abstentions,
    f"({abstentions / n_ua:.1%})"
)

print(
    "Fabrication:",
    fabrications,
    f"({fabrications / n_ua:.1%})"
)

print(
    "Ambiguous:",
    ua_ambiguous,
    f"({ua_ambiguous / n_ua:.1%})"
)


print("\nANSWERABLE QUESTIONS")
print("Total:", n_ab)

print(
    "Correct:",
    correct,
    f"({correct / n_ab:.1%})"
)

print(
    "Incorrect:",
    incorrect,
    f"({incorrect / n_ab:.1%})"
)

print(
    "Ambiguous:",
    ab_ambiguous,
    f"({ab_ambiguous / n_ab:.1%})"
)
