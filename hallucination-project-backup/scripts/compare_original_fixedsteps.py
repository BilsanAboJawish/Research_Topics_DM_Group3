import json
from pathlib import Path
from statistics import NormalDist


SUBTASKS = ["dilemma", "inter", "time"]

CONDITIONS = []

for method in ["IFD", "Random"]:
    for pct in ["05", "10", "15", "20", "25"]:

        if method == "IFD":
            original_root = f"results/official_{pct}pct"
            original_marker = f"mistral_lora_{pct}pct"

            fixed_root = f"results/official_{pct}pct_fixedsteps"
            fixed_marker = f"mistral_lora_{pct}pct_fixedsteps"

        else:
            original_root = f"results/official_random_{pct}pct"
            original_marker = f"mistral_lora_random_{pct}pct"

            fixed_root = f"results/official_random_{pct}pct_fixedsteps"
            fixed_marker = f"mistral_lora_random_{pct}pct_fixedsteps"

        CONDITIONS.append({
            "method": method,
            "budget": pct,
            "original_root": original_root,
            "original_marker": original_marker,
            "fixed_root": fixed_root,
            "fixed_marker": fixed_marker,
        })


def find_file(root, marker, task, qtype):
    matches = [
        p for p in Path(root).rglob(
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
        p = find_file(root, marker, task, qtype)

        with open(p, encoding="utf-8") as f:
            rows.extend(
                json.loads(line)
                for line in f
                if line.strip()
            )

    return rows


def rate(rows, metric):
    return sum(
        float(r[metric]) for r in rows
    ) / len(rows)


def dprime(uaq_ref, abq_ref, n_uaq, n_abq):
    if uaq_ref <= 0:
        uaq_ref = 0.5 / n_uaq
    elif uaq_ref >= 1:
        uaq_ref = (n_uaq - 0.5) / n_uaq

    if abq_ref <= 0:
        abq_ref = 0.5 / n_abq
    elif abq_ref >= 1:
        abq_ref = (n_abq - 0.5) / n_abq

    nd = NormalDist()

    return (
        nd.inv_cdf(uaq_ref)
        - nd.inv_cdf(abq_ref)
    )


def evaluate(root, marker):
    abq = read_rows(root, marker, "ab")
    uaq = read_rows(root, marker, "ua")

    abr = rate(abq, "NEC_refuse")
    uar = rate(uaq, "NEC_refuse")

    return {
        "ABQ": 100 * abr,
        "UAQ": 100 * uar,
        "gap": 100 * (uar - abr),
        "dprime": dprime(
            uar, abr, len(uaq), len(abq)
        ),
    }


rows = []


for c in CONDITIONS:

    try:
        original = evaluate(
            c["original_root"],
            c["original_marker"]
        )

        fixed = evaluate(
            c["fixed_root"],
            c["fixed_marker"]
        )

    except FileNotFoundError:
        print(
            f"SKIPPING MISSING: "
            f"{c['method']} {c['budget']}%"
        )
        continue

    rows.append({
        "method": c["method"],
        "budget": c["budget"],

        "orig_ABQ": original["ABQ"],
        "fix_ABQ": fixed["ABQ"],
        "delta_ABQ":
            fixed["ABQ"] - original["ABQ"],

        "orig_UAQ": original["UAQ"],
        "fix_UAQ": fixed["UAQ"],
        "delta_UAQ":
            fixed["UAQ"] - original["UAQ"],

        "orig_gap": original["gap"],
        "fix_gap": fixed["gap"],
        "delta_gap":
            fixed["gap"] - original["gap"],

        "orig_dprime": original["dprime"],
        "fix_dprime": fixed["dprime"],
        "delta_dprime":
            fixed["dprime"] - original["dprime"],
    })


print()
print("=" * 128)
print("ORIGINAL 1-EPOCH vs FIXED 975-STEP CONTROL")
print("=" * 128)

print(
    f"{'Condition':<13}"
    f"{'Orig ABQ':>10}"
    f"{'Fix ABQ':>10}"
    f"{'ΔABQ':>9}"
    f"{'Orig UAQ':>10}"
    f"{'Fix UAQ':>10}"
    f"{'ΔUAQ':>9}"
    f"{'Orig gap':>10}"
    f"{'Fix gap':>9}"
    f"{'Orig d':>9}"
    f"{'Fix d':>9}"
)

print("-" * 128)

for r in rows:

    condition = (
        f"{r['method']} {r['budget']}%"
    )

    print(
        f"{condition:<13}"
        f"{r['orig_ABQ']:>9.2f}%"
        f"{r['fix_ABQ']:>9.2f}%"
        f"{r['delta_ABQ']:>+9.2f}"
        f"{r['orig_UAQ']:>9.2f}%"
        f"{r['fix_UAQ']:>9.2f}%"
        f"{r['delta_UAQ']:>+9.2f}"
        f"{r['orig_gap']:>10.2f}"
        f"{r['fix_gap']:>9.2f}"
        f"{r['orig_dprime']:>9.3f}"
        f"{r['fix_dprime']:>9.3f}"
    )


print()
print("INTERPRETATION HELP")
print("-------------------")

for method in ["IFD", "Random"]:

    subset = [
        r for r in rows
        if r["method"] == method
    ]

    mean_abs_abq = sum(
        abs(r["delta_ABQ"])
        for r in subset
    ) / len(subset)

    mean_abs_uaq = sum(
        abs(r["delta_UAQ"])
        for r in subset
    ) / len(subset)

    mean_abs_d = sum(
        abs(r["delta_dprime"])
        for r in subset
    ) / len(subset)

    print(
        f"{method}: "
        f"mean |ΔABQ|={mean_abs_abq:.2f} pp, "
        f"mean |ΔUAQ|={mean_abs_uaq:.2f} pp, "
        f"mean |Δd'|={mean_abs_d:.3f}"
    )
