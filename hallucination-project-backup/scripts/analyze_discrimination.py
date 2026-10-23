import json
from pathlib import Path
from statistics import NormalDist


CONDITIONS = [
    ("BASE", "results/official_baseline",
     "mistralai__Mistral-7B-Instruct-v0.3"),

    ("IFD_05", "results/official_05pct",
     "mistral_lora_05pct"),
    ("IFD_10", "results/official_10pct",
     "mistral_lora_10pct"),
    ("IFD_15", "results/official_15pct",
     "mistral_lora_15pct"),
    ("IFD_20", "results/official_20pct",
     "mistral_lora_20pct"),
    ("IFD_25", "results/official_25pct",
     "mistral_lora_25pct"),

    ("Random_05", "results/official_random_05pct",
     "mistral_lora_random_05pct"),
    ("Random_10", "results/official_random_10pct",
     "mistral_lora_random_10pct"),
    ("Random_15", "results/official_random_15pct",
     "mistral_lora_random_15pct"),
    ("Random_20", "results/official_random_20pct",
     "mistral_lora_random_20pct"),
    ("Random_25", "results/official_random_25pct",
     "mistral_lora_random_25pct"),
]

SUBTASKS = ["dilemma", "inter", "time"]


def find_file(root, marker, task, qtype):
    files = [
        p for p in Path(root).rglob(
            f"samples_task1_{task}_en_{qtype}_*.jsonl"
        )
        if marker in str(p)
    ]

    if not files:
        raise FileNotFoundError(
            f"{root}: missing {task} {qtype}"
        )

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def get_all(root, marker, qtype):
    rows = []

    for task in SUBTASKS:
        rows.extend(
            read_jsonl(
                find_file(root, marker, task, qtype)
            )
        )

    return rows


def rate(rows, metric):
    return sum(float(x[metric]) for x in rows) / len(rows)


def dprime(hit, false_alarm, hit_n, fa_n):
    # Log-linear correction only if a rate reaches exactly 0 or 1.
    if hit <= 0:
        hit = 0.5 / hit_n
    elif hit >= 1:
        hit = (hit_n - 0.5) / hit_n

    if false_alarm <= 0:
        false_alarm = 0.5 / fa_n
    elif false_alarm >= 1:
        false_alarm = (fa_n - 0.5) / fa_n

    nd = NormalDist()

    return (
        nd.inv_cdf(hit)
        - nd.inv_cdf(false_alarm)
    )


print(
    f"{'Condition':<12}"
    f"{'ABQ N':>8}"
    f"{'UAQ N':>8}"
    f"{'ABQ Ref%':>11}"
    f"{'UAQ Ref%':>11}"
    f"{'Gap pp':>10}"
    f"{'d-prime':>10}"
    f"{'AnsAcc%':>11}"
)

print("-" * 81)


for name, root, marker in CONDITIONS:

    abq = get_all(root, marker, "ab")
    uaq = get_all(root, marker, "ua")

    ab_ref = rate(abq, "NEC_refuse")
    ua_ref = rate(uaq, "NEC_refuse")

    d = dprime(
        ua_ref,
        ab_ref,
        len(uaq),
        len(abq)
    )

    answered = [
        x for x in abq
        if int(float(x["NEC_refuse"])) == 0
    ]

    answered_correct = [
        x for x in answered
        if int(float(x["EM_hit"])) == 1
    ]

    ans_acc = (
        len(answered_correct) / len(answered)
        if answered else 0
    )

    print(
        f"{name:<12}"
        f"{len(abq):>8}"
        f"{len(uaq):>8}"
        f"{100*ab_ref:>11.2f}"
        f"{100*ua_ref:>11.2f}"
        f"{100*(ua_ref-ab_ref):>10.2f}"
        f"{d:>10.3f}"
        f"{100*ans_acc:>11.2f}"
    )
