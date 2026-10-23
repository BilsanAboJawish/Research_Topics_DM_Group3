import json
import math
from pathlib import Path
from statistics import NormalDist


CONDITIONS = [
    # Original 15% / 1 epoch
    ("IFD 15% / 1 epoch",
     "results/official_15pct",
     "mistral_lora_15pct"),

    ("Random 15% / 1 epoch",
     "results/official_random_15pct",
     "mistral_lora_random_15pct"),

    # Fixed-budget / exposure control
    ("IFD 15% / 0.5 epoch",
     "results/official_15pct_halfepoch",
     "mistral_lora_15pct_halfepoch"),

    ("IFD 15% / 2 epochs",
     "results/official_15pct_2epoch",
     "mistral_lora_15pct_2epoch"),

    ("Random 15% / 0.5 epoch",
     "results/official_random_15pct_halfepoch",
     "mistral_lora_random_15pct_halfepoch"),

    ("Random 15% / 2 epochs",
     "results/official_random_15pct_2epoch",
     "mistral_lora_random_15pct_2epoch"),

    # Fixed 975-step IFD
    ("IFD 5% / 975 steps",
     "results/official_05pct_fixedsteps",
     "mistral_lora_05pct_fixedsteps"),

    ("IFD 10% / 975 steps",
     "results/official_10pct_fixedsteps",
     "mistral_lora_10pct_fixedsteps"),

    ("IFD 15% / 975 steps",
     "results/official_15pct_fixedsteps",
     "mistral_lora_15pct_fixedsteps"),

    ("IFD 20% / 975 steps",
     "results/official_20pct_fixedsteps",
     "mistral_lora_20pct_fixedsteps"),

    ("IFD 25% / 975 steps",
     "results/official_25pct_fixedsteps",
     "mistral_lora_25pct_fixedsteps"),

    # Fixed 975-step Random
    ("Random 5% / 975 steps",
     "results/official_random_05pct_fixedsteps",
     "mistral_lora_random_05pct_fixedsteps"),

    ("Random 10% / 975 steps",
     "results/official_random_10pct_fixedsteps",
     "mistral_lora_random_10pct_fixedsteps"),

    ("Random 15% / 975 steps",
     "results/official_random_15pct_fixedsteps",
     "mistral_lora_random_15pct_fixedsteps"),

    ("Random 20% / 975 steps",
     "results/official_random_20pct_fixedsteps",
     "mistral_lora_random_20pct_fixedsteps"),

    ("Random 25% / 975 steps",
     "results/official_random_25pct_fixedsteps",
     "mistral_lora_random_25pct_fixedsteps"),
]


SUBTASKS = ["dilemma", "inter", "time"]


def find_file(root, marker, task, qtype):
    root = Path(root)

    matches = [
        p for p in root.rglob(
            f"samples_task1_{task}_en_{qtype}_*.jsonl"
        )
        if marker in str(p)
    ]

    if not matches:
        raise FileNotFoundError

    matches.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return matches[0]


def read_rows(root, marker, qtype):
    rows = []

    for task in SUBTASKS:
        path = find_file(
            root,
            marker,
            task,
            qtype
        )

        with open(path, encoding="utf-8") as f:
            rows.extend(
                json.loads(line)
                for line in f
                if line.strip()
            )

    return rows


def get_metric(row, name):
    return float(row[name])


def mean_metric(rows, name):
    return (
        sum(get_metric(r, name) for r in rows)
        / len(rows)
    )


def dprime(hit, false_alarm, n_hit, n_fa):
    if hit <= 0:
        hit = 0.5 / n_hit
    elif hit >= 1:
        hit = (n_hit - 0.5) / n_hit

    if false_alarm <= 0:
        false_alarm = 0.5 / n_fa
    elif false_alarm >= 1:
        false_alarm = (n_fa - 0.5) / n_fa

    nd = NormalDist()

    return (
        nd.inv_cdf(hit)
        - nd.inv_cdf(false_alarm)
    )


results = []


for name, root, marker in CONDITIONS:

    try:
        abq = read_rows(root, marker, "ab")
        uaq = read_rows(root, marker, "ua")

    except FileNotFoundError:
        print(f"SKIPPING MISSING: {name}")
        continue

    abq_ref = mean_metric(
        abq,
        "NEC_refuse"
    )

    uaq_ref = mean_metric(
        uaq,
        "NEC_refuse"
    )

    em = mean_metric(
        abq,
        "EM_hit"
    )

    answered = [
        r for r in abq
        if get_metric(r, "NEC_refuse") == 0
    ]

    answered_correct = [
        r for r in answered
        if get_metric(r, "EM_hit") == 1
    ]

    ans_acc = (
        len(answered_correct) / len(answered)
        if answered
        else math.nan
    )

    d = dprime(
        uaq_ref,
        abq_ref,
        len(uaq),
        len(abq)
    )

    results.append({
        "name": name,
        "ABQ_ref": 100 * abq_ref,
        "UAQ_ref": 100 * uaq_ref,
        "gap": 100 * (uaq_ref - abq_ref),
        "dprime": d,
        "EM": 100 * em,
        "AnsAcc": 100 * ans_acc,
    })


lookup = {
    r["name"]: r
    for r in results
}


print()
print("=" * 115)
print("AVAILABLE CONTROL EXPERIMENT RESULTS")
print("=" * 115)

print(
    f"{'Condition':<31}"
    f"{'ABQ Ref':>10}"
    f"{'UAQ Ref':>10}"
    f"{'Gap':>9}"
    f"{'d-prime':>10}"
    f"{'EM':>9}"
    f"{'AnsAcc':>10}"
)

print("-" * 115)

for r in results:
    print(
        f"{r['name']:<31}"
        f"{r['ABQ_ref']:>9.2f}%"
        f"{r['UAQ_ref']:>9.2f}%"
        f"{r['gap']:>8.2f}"
        f"{r['dprime']:>10.3f}"
        f"{r['EM']:>8.2f}%"
        f"{r['AnsAcc']:>9.2f}%"
    )


print()
print("=" * 115)
print("FIXED 15% BUDGET — TRAINING EXPOSURE")
print("=" * 115)

order = [
    "IFD 15% / 0.5 epoch",
    "IFD 15% / 1 epoch",
    "IFD 15% / 2 epochs",

    "Random 15% / 0.5 epoch",
    "Random 15% / 1 epoch",
    "Random 15% / 2 epochs",
]

for name in order:

    if name not in lookup:
        print(f"{name:<31} MISSING")
        continue

    r = lookup[name]

    print(
        f"{name:<31}"
        f" ABQ={r['ABQ_ref']:6.2f}%"
        f" UAQ={r['UAQ_ref']:6.2f}%"
        f" gap={r['gap']:6.2f}"
        f" d'={r['dprime']:.3f}"
        f" AnsAcc={r['AnsAcc']:6.2f}%"
    )


print()
print("=" * 115)
print("FIXED 975 OPTIMIZER STEPS — BUDGET")
print("=" * 115)

for method in ["IFD", "Random"]:

    print(f"\n{method}")

    for pct in [5, 10, 15, 20, 25]:

        name = f"{method} {pct}% / 975 steps"

        if name not in lookup:
            print(f"{pct:>2}%: MISSING")
            continue

        r = lookup[name]

        print(
            f"{pct:>2}%:"
            f" ABQ={r['ABQ_ref']:6.2f}%"
            f" UAQ={r['UAQ_ref']:6.2f}%"
            f" gap={r['gap']:6.2f}"
            f" d'={r['dprime']:.3f}"
            f" AnsAcc={r['AnsAcc']:6.2f}%"
        )
