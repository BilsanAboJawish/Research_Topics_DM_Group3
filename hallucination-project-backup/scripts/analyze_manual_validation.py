import csv
from collections import Counter, defaultdict
from pathlib import Path

LABELED = Path(
    "results/manual_validation/manual_validation_200_labeled.csv"
)

KEY = Path(
    "results/manual_validation/manual_validation_200_hidden_key.csv"
)

OUT = Path("results/manual_validation")
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pct(x):
    return f"{100*x:.2f}%"


def binary_stats(rows, positive_labels):
    """
    Compare NEC_refuse to manual labels.

    positive_labels:
      {"R"}       -> strict refusal
      {"R", "P"}  -> broad abstention / non-answer

    M is excluded.
    """
    tp = fp = tn = fn = 0
    used = []

    for r in rows:
        label = r["manual_label"].strip().upper()

        if label == "M":
            continue

        manual = 1 if label in positive_labels else 0
        auto = int(float(r["official_NEC_refuse"]))

        used.append((manual, auto))

        if manual == 1 and auto == 1:
            tp += 1
        elif manual == 0 and auto == 1:
            fp += 1
        elif manual == 0 and auto == 0:
            tn += 1
        elif manual == 1 and auto == 0:
            fn += 1

    n = len(used)

    agreement = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "agreement": agreement,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def print_stats(title, rows, positive_labels):
    s = binary_stats(rows, positive_labels)

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"N evaluated:   {s['n']}")
    print(f"Agreement:     {pct(s['agreement'])}")
    print(f"Precision:     {pct(s['precision'])}")
    print(f"Recall:        {pct(s['recall'])}")
    print(f"Specificity:   {pct(s['specificity'])}")
    print(f"F1:            {pct(s['f1'])}")

    print("\nConfusion matrix:")
    print(f"  TP: {s['tp']}")
    print(f"  FP: {s['fp']}")
    print(f"  TN: {s['tn']}")
    print(f"  FN: {s['fn']}")

    return s


# -----------------------------------------------------------
# LOAD + JOIN
# -----------------------------------------------------------

labels = read_csv(LABELED)
key = read_csv(KEY)

labels_by_id = {
    r["validation_id"].strip(): r
    for r in labels
}

key_by_id = {
    r["validation_id"].strip(): r
    for r in key
}

if set(labels_by_id) != set(key_by_id):
    missing_labels = set(key_by_id) - set(labels_by_id)
    missing_key = set(labels_by_id) - set(key_by_id)

    raise ValueError(
        f"IDs do not match.\n"
        f"Missing from labeled file: {missing_labels}\n"
        f"Missing from key file: {missing_key}"
    )

rows = []

for vid, k in key_by_id.items():
    l = labels_by_id[vid]

    row = dict(k)

    row["manual_label"] = (
        l.get("manual_label", "").strip().upper()
    )

    row["notes"] = l.get("notes", "").strip()

    rows.append(row)


# -----------------------------------------------------------
# VALIDATE MANUAL LABELS
# -----------------------------------------------------------

allowed = {"A", "R", "P", "M"}

bad = [
    (r["validation_id"], r["manual_label"])
    for r in rows
    if r["manual_label"] not in allowed
]

if bad:
    raise ValueError(
        "Invalid or missing manual labels:\n" +
        "\n".join(map(str, bad))
    )

print("=" * 70)
print("MANUAL LABEL DISTRIBUTION")
print("=" * 70)

counts = Counter(r["manual_label"] for r in rows)

for label in ["A", "R", "P", "M"]:
    print(f"{label}: {counts[label]}")


# -----------------------------------------------------------
# 1. STRICT REFUSAL
# R = refusal
# A/P = not refusal
# -----------------------------------------------------------

strict = print_stats(
    "STRICT VALIDATION: only R counts as refusal",
    rows,
    {"R"}
)


# -----------------------------------------------------------
# 2. BROAD ABSTENTION
# R + P = non-answer / abstention-style behavior
# -----------------------------------------------------------

broad = print_stats(
    "BROAD VALIDATION: R + P count as abstention/non-answer",
    rows,
    {"R", "P"}
)


# -----------------------------------------------------------
# BREAKDOWNS
# -----------------------------------------------------------

def group_rows(field):
    groups = defaultdict(list)

    for r in rows:
        groups[r[field]].append(r)

    return groups


for field, title in [
    ("selection", "SELECTION METHOD"),
    ("question_type", "QUESTION TYPE"),
    ("budget", "BUDGET"),
    ("subtask", "UAQFACT SUBTASK"),
]:

    print("\n\n" + "#" * 70)
    print(title)
    print("#" * 70)

    for value, subset in sorted(group_rows(field).items()):

        s = binary_stats(subset, {"R"})

        print(
            f"{value:>10}: "
            f"N={s['n']:3d}, "
            f"strict agreement={pct(s['agreement'])}, "
            f"precision={pct(s['precision'])}, "
            f"recall={pct(s['recall'])}"
        )


# -----------------------------------------------------------
# ERROR TYPES
# -----------------------------------------------------------

strict_disagreements = []

premise_cases = []

for r in rows:

    label = r["manual_label"]
    auto = int(float(r["official_NEC_refuse"]))

    if label != "M":

        manual_strict = 1 if label == "R" else 0

        if manual_strict != auto:
            strict_disagreements.append(r)

    if label == "P":
        premise_cases.append(r)


print("\n\n" + "=" * 70)
print("PREMISE REJECTION ANALYSIS")
print("=" * 70)

print(f"Manual P cases: {len(premise_cases)}")

if premise_cases:

    auto_refuse_p = sum(
        int(float(r["official_NEC_refuse"])) == 1
        for r in premise_cases
    )

    print(
        "P cases classified as refusal by NEC_refuse: "
        f"{auto_refuse_p}/{len(premise_cases)} "
        f"({pct(auto_refuse_p / len(premise_cases))})"
    )


# -----------------------------------------------------------
# SAVE JOINED DATA
# -----------------------------------------------------------

joined_path = OUT / "manual_validation_joined.csv"

fields = [
    "validation_id",
    "selection",
    "budget",
    "question_type",
    "subtask",
    "manual_label",
    "official_NEC_refuse",
    "EM_hit",
    "question",
    "model_response",
    "notes",
]

with open(
    joined_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    for r in rows:
        writer.writerow({
            field: r.get(field, "")
            for field in fields
        })


# -----------------------------------------------------------
# SAVE DISAGREEMENTS
# -----------------------------------------------------------

disagreement_path = OUT / "manual_validation_disagreements.csv"

with open(
    disagreement_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    for r in strict_disagreements:
        writer.writerow({
            field: r.get(field, "")
            for field in fields
        })


# -----------------------------------------------------------
# SAVE SUMMARY TEXT
# -----------------------------------------------------------

summary_path = OUT / "manual_validation_summary.txt"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("Manual validation of UAQFact NEC_refuse\n\n")

    f.write("Strict refusal definition: only R is refusal\n")
    for k, v in strict.items():
        f.write(f"{k}: {v}\n")

    f.write("\nBroad definition: R or P is abstention/non-answer\n")
    for k, v in broad.items():
        f.write(f"{k}: {v}\n")

    f.write(
        f"\nPremise rejection cases: {len(premise_cases)}\n"
    )

    f.write(
        f"Strict disagreements: "
        f"{len(strict_disagreements)}\n"
    )


print("\n\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(joined_path)
print(disagreement_path)
print(summary_path)
