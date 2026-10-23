import json
import math
import csv
from pathlib import Path
from statistics import mean, stdev


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

T_CRIT_DF2 = 4.302652729911275   # 95% CI, two-sided, df = 2

SUBTASKS = ["dilemma", "inter", "time"]

CONDITIONS = []

for selection in ["IFD", "Random"]:
    for pct in ["05", "10", "15", "20", "25"]:
        for seed in [42, 123, 2026]:

            if selection == "IFD":
                if seed == 42:
                    root = f"results/official_{pct}pct"
                    marker = f"mistral_lora_{pct}pct"
                else:
                    root = f"results/official_{pct}pct_seed{seed}"
                    marker = f"mistral_lora_{pct}pct_seed{seed}"

            else:
                if seed == 42:
                    root = f"results/official_random_{pct}pct"
                    marker = f"mistral_lora_random_{pct}pct"
                else:
                    root = f"results/official_random_{pct}pct_seed{seed}"
                    marker = f"mistral_lora_random_{pct}pct_seed{seed}"

            CONDITIONS.append({
                "selection": selection,
                "budget": pct,
                "seed": seed,
                "root": root,
                "marker": marker,
            })


OUT_DIR = Path("results/statistical_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def find_sample_file(root, marker, subtask, qtype):
    """
    qtype = 'ab' or 'ua'
    """

    root = Path(root)

    pattern = f"samples_task1_{subtask}_en_{qtype}_*.jsonl"

    files = [
        p for p in root.rglob(pattern)
        if marker in str(p)
    ]

    if not files:
        raise FileNotFoundError(
            f"No file found:\n"
            f"root={root}\n"
            f"marker={marker}\n"
            f"pattern={pattern}"
        )

    # Use newest matching file if evaluation was run multiple times.
    files.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return files[0]


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_condition(root, marker):

    abq = []
    uaq = []

    counts = {}

    for subtask in SUBTASKS:

        ab_path = find_sample_file(
            root, marker, subtask, "ab"
        )

        ua_path = find_sample_file(
            root, marker, subtask, "ua"
        )

        ab_rows = read_jsonl(ab_path)
        ua_rows = read_jsonl(ua_path)

        counts[f"{subtask}_ab"] = len(ab_rows)
        counts[f"{subtask}_ua"] = len(ua_rows)

        abq.extend(ab_rows)
        uaq.extend(ua_rows)

    return abq, uaq, counts


def metric_mean(rows, metric):
    vals = [
        float(r[metric])
        for r in rows
    ]

    return sum(vals) / len(vals)


def condition_metrics(abq, uaq):

    ab_refusal = metric_mean(abq, "NEC_refuse")
    ua_refusal = metric_mean(uaq, "NEC_refuse")
    ab_em = metric_mean(abq, "EM_hit")

    # Directly compute accuracy among NON-REFUSED ABQs
    answered_abq = [
        r for r in abq
        if int(float(r["NEC_refuse"])) == 0
    ]

    answered_correct = [
        r for r in answered_abq
        if int(float(r["EM_hit"])) == 1
    ]

    answered_accuracy = (
        len(answered_correct) / len(answered_abq)
        if answered_abq else float("nan")
    )

    refusal_gap = ua_refusal - ab_refusal

    return {
        "UAQ_refusal": 100 * ua_refusal,
        "ABQ_refusal": 100 * ab_refusal,
        "ABQ_EM_hit": 100 * ab_em,
        "ABQ_answered_accuracy": 100 * answered_accuracy,
        "refusal_gap": 100 * refusal_gap,
    }


def ci_summary(values):

    n = len(values)

    m = mean(values)

    if n < 2:
        return {
            "n": n,
            "mean": m,
            "sd": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "half_width": float("nan"),
        }

    s = stdev(values)

    half = T_CRIT_DF2 * s / math.sqrt(n)

    return {
        "n": n,
        "mean": m,
        "sd": s,
        "ci_low": m - half,
        "ci_high": m + half,
        "half_width": half,
    }


# ---------------------------------------------------------
# READ EVERY RUN
# ---------------------------------------------------------

run_results = []

reference_counts = None

for c in CONDITIONS:

    abq, uaq, counts = load_condition(
        c["root"],
        c["marker"]
    )

    metrics = condition_metrics(abq, uaq)

    row = {
        **c,
        "ABQ_n": len(abq),
        "UAQ_n": len(uaq),
        **counts,
        **metrics,
    }

    run_results.append(row)

    if reference_counts is None:
        reference_counts = {
            "ABQ_n": len(abq),
            "UAQ_n": len(uaq),
            **counts,
        }

    # Sanity check: every run should use exactly same benchmark size.
    current = {
        "ABQ_n": len(abq),
        "UAQ_n": len(uaq),
        **counts,
    }

    if current != reference_counts:
        raise ValueError(
            "Benchmark counts differ between runs!\n"
            f"Expected: {reference_counts}\n"
            f"Found: {current}\n"
            f"Condition: {c}"
        )


# ---------------------------------------------------------
# PRINT EXACT QUESTION COUNTS
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("EXACT UAQFACT TASK 1 EN QUESTION COUNTS")
print("=" * 80)

for subtask in SUBTASKS:

    print(
        f"{subtask:10s} "
        f"ABQ={reference_counts[f'{subtask}_ab']:5d}   "
        f"UAQ={reference_counts[f'{subtask}_ua']:5d}"
    )

print("-" * 80)

print(
    f"{'TOTAL':10s} "
    f"ABQ={reference_counts['ABQ_n']:5d}   "
    f"UAQ={reference_counts['UAQ_n']:5d}"
)

print(
    "Grand total:",
    reference_counts["ABQ_n"] + reference_counts["UAQ_n"]
)


# ---------------------------------------------------------
# SAVE PER-SEED RESULTS
# ---------------------------------------------------------

per_seed_csv = OUT_DIR / "per_seed_results.csv"

with open(
    per_seed_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=run_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(run_results)


# ---------------------------------------------------------
# AGGREGATE THE 3 SEEDS
# ---------------------------------------------------------

metrics_to_report = [
    "UAQ_refusal",
    "ABQ_refusal",
    "ABQ_EM_hit",
    "ABQ_answered_accuracy",
    "refusal_gap",
]

summary_rows = []

for selection in ["IFD", "Random"]:

    for pct in ["05", "10", "15", "20", "25"]:

        subset = [
            r for r in run_results
            if r["selection"] == selection
            and r["budget"] == pct
        ]

        if len(subset) != 3:
            raise ValueError(
                f"{selection} {pct}% has "
                f"{len(subset)} runs instead of 3."
            )

        for metric in metrics_to_report:

            vals = [
                r[metric]
                for r in subset
            ]

            s = ci_summary(vals)

            summary_rows.append({
                "selection": selection,
                "budget": pct,
                "metric": metric,

                "seed42": vals[0],
                "seed123": vals[1],
                "seed2026": vals[2],

                "mean": s["mean"],
                "sd": s["sd"],

                "ci95_low": s["ci_low"],
                "ci95_high": s["ci_high"],
                "ci95_half_width": s["half_width"],
            })


# ---------------------------------------------------------
# PRINT MAIN TABLE
# ---------------------------------------------------------

print("\n\n" + "=" * 130)
print("MULTI-SEED SUMMARY: MEAN ± SD AND 95% CI")
print("=" * 130)

for metric in metrics_to_report:

    print("\n" + metric)
    print("-" * 105)

    print(
        f"{'Condition':<12}"
        f"{'Mean':>10}"
        f"{'SD':>10}"
        f"{'95% CI low':>15}"
        f"{'95% CI high':>15}"
        f"{'Mean ± SD':>22}"
    )

    for row in summary_rows:

        if row["metric"] != metric:
            continue

        condition = (
            f"{row['selection']}_{row['budget']}"
        )

        print(
            f"{condition:<12}"
            f"{row['mean']:>10.2f}"
            f"{row['sd']:>10.2f}"
            f"{row['ci95_low']:>15.2f}"
            f"{row['ci95_high']:>15.2f}"
            f"{row['mean']:>10.2f} ± {row['sd']:<7.2f}"
        )


# ---------------------------------------------------------
# SAVE SUMMARY CSV
# ---------------------------------------------------------

summary_csv = OUT_DIR / "multiseed_mean_sd_ci95.csv"

with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ---------------------------------------------------------
# SAVE QUESTION COUNTS
# ---------------------------------------------------------

count_csv = OUT_DIR / "uaqfact_question_counts.csv"

count_rows = []

for subtask in SUBTASKS:

    count_rows.append({
        "subtask": subtask,
        "ABQ": reference_counts[f"{subtask}_ab"],
        "UAQ": reference_counts[f"{subtask}_ua"],
    })

count_rows.append({
    "subtask": "TOTAL",
    "ABQ": reference_counts["ABQ_n"],
    "UAQ": reference_counts["UAQ_n"],
})

with open(
    count_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["subtask", "ABQ", "UAQ"]
    )

    writer.writeheader()
    writer.writerows(count_rows)


print("\n\nSaved:")
print(per_seed_csv)
print(summary_csv)
print(count_csv)

print("\nNOTE:")
print(
    "95% confidence intervals use a Student-t interval "
    "across the 3 TRAINING seeds (df=2, t=4.303)."
)
print(
    "These CIs quantify training-run variability, not "
    "benchmark-item sampling uncertainty."
)
