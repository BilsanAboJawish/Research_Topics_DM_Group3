import json
from pathlib import Path
from statistics import NormalDist

CONDITIONS = [
    ("100% Alpaca-GPT4", "results/official_100pct"),
    ("Length 15%",       "results/official_length_15pct"),
]

TASKS = ["dilemma", "inter", "time"]


def find_latest(root, task, qtype):
    matches = list(
        Path(root).rglob(
            f"samples_task1_{task}_en_{qtype}_*.jsonl"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No file found: {root}, {task}, {qtype}"
        )

    matches.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return matches[0]


def read_rows(root, qtype):
    rows = []

    for task in TASKS:
        path = find_latest(root, task, qtype)

        with open(path, encoding="utf-8") as f:
            rows.extend(
                json.loads(line)
                for line in f
                if line.strip()
            )

    return rows


def rate(rows, metric):
    return sum(
        float(x[metric])
        for x in rows
    ) / len(rows)


def dprime(uaq_ref, abq_ref):
    nd = NormalDist()

    eps = 1e-7
    uaq_ref = min(max(uaq_ref, eps), 1 - eps)
    abq_ref = min(max(abq_ref, eps), 1 - eps)

    return (
        nd.inv_cdf(uaq_ref)
        - nd.inv_cdf(abq_ref)
    )


print()
print("=" * 95)
print("NEW EXPERIMENTAL CONDITIONS")
print("=" * 95)

print(
    f"{'Condition':<23}"
    f"{'ABQ Ref':>10}"
    f"{'UAQ Ref':>10}"
    f"{'Gap':>9}"
    f"{'d-prime':>10}"
    f"{'EM':>9}"
    f"{'AnsAcc':>10}"
)

print("-" * 95)


for name, root in CONDITIONS:

    abq = read_rows(root, "ab")
    uaq = read_rows(root, "ua")

    abq_ref = rate(abq, "NEC_refuse")
    uaq_ref = rate(uaq, "NEC_refuse")
    em = rate(abq, "EM_hit")

    answered = [
        x for x in abq
        if float(x["NEC_refuse"]) == 0
    ]

    answered_acc = (
        sum(float(x["EM_hit"]) for x in answered)
        / len(answered)
    )

    dp = dprime(
        uaq_ref,
        abq_ref
    )

    print(
        f"{name:<23}"
        f"{100*abq_ref:>9.2f}%"
        f"{100*uaq_ref:>9.2f}%"
        f"{100*(uaq_ref-abq_ref):>8.2f}"
        f"{dp:>10.3f}"
        f"{100*em:>8.2f}%"
        f"{100*answered_acc:>9.2f}%"
    )

    print(
        f"    N: ABQ={len(abq)}, "
        f"UAQ={len(uaq)}, "
        f"answered ABQ={len(answered)}"
    )
