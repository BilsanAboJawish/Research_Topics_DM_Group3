import json
import csv
from pathlib import Path
from statistics import mean, median


CONDITIONS = [
    (
        "BASE",
        "0",
        "results/official_baseline",
        "mistralai__Mistral-7B-Instruct-v0.3",
    ),

    ("IFD", "05", "results/official_05pct",
     "mistral_lora_05pct"),
    ("IFD", "10", "results/official_10pct",
     "mistral_lora_10pct"),
    ("IFD", "15", "results/official_15pct",
     "mistral_lora_15pct"),
    ("IFD", "20", "results/official_20pct",
     "mistral_lora_20pct"),
    ("IFD", "25", "results/official_25pct",
     "mistral_lora_25pct"),

    ("Random", "05", "results/official_random_05pct",
     "mistral_lora_random_05pct"),
    ("Random", "10", "results/official_random_10pct",
     "mistral_lora_random_10pct"),
    ("Random", "15", "results/official_random_15pct",
     "mistral_lora_random_15pct"),
    ("Random", "20", "results/official_random_20pct",
     "mistral_lora_random_20pct"),
    ("Random", "25", "results/official_random_25pct",
     "mistral_lora_random_25pct"),
]


SUBTASKS = ["dilemma", "inter", "time"]

OUT_DIR = Path("results/evaluation_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def get_response(record):

    filtered = record.get("filtered_resps")

    if isinstance(filtered, list) and filtered:
        value = filtered[0]

        while isinstance(value, list) and value:
            value = value[0]

        return str(value).strip()

    resps = record.get("resps")

    if isinstance(resps, list) and resps:
        value = resps[0]

        while isinstance(value, list) and value:
            value = value[0]

        return str(value).strip()

    return ""


def find_abq_file(root, marker, subtask):

    root = Path(root)

    pattern = f"samples_task1_{subtask}_en_ab_*.jsonl"

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

    # If the experiment was run more than once,
    # use the newest matching output.
    files.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return files[0]


def summarize(selection, budget, root, marker):

    all_rows = []

    for subtask in SUBTASKS:

        path = find_abq_file(
            root,
            marker,
            subtask
        )

        rows = read_jsonl(path)

        for row in rows:
            row["_subtask"] = subtask
            all_rows.append(row)

    n = len(all_rows)

    refusals = [
        r for r in all_rows
        if int(float(r["NEC_refuse"])) == 1
    ]

    answered = [
        r for r in all_rows
        if int(float(r["NEC_refuse"])) == 0
    ]

    correct = [
        r for r in all_rows
        if int(float(r["EM_hit"])) == 1
    ]

    answered_correct = [
        r for r in answered
        if int(float(r["EM_hit"])) == 1
    ]

    refused_but_em_hit = [
        r for r in refusals
        if int(float(r["EM_hit"])) == 1
    ]

    all_lengths = [
        len(get_response(r).split())
        for r in all_rows
    ]

    answered_lengths = [
        len(get_response(r).split())
        for r in answered
    ]

    correct_lengths = [
        len(get_response(r).split())
        for r in answered_correct
    ]

    refusal_rate = len(refusals) / n
    overall_em = len(correct) / n

    answered_accuracy = (
        len(answered_correct) / len(answered)
        if answered else 0
    )

    approx_formula = (
        overall_em / (1 - refusal_rate)
        if refusal_rate < 1 else 0
    )

    result = {
        "selection": selection,
        "budget": budget,

        "n_abq": n,

        "refusal_rate_pct":
            100 * refusal_rate,

        "EM_hit_pct":
            100 * overall_em,

        "answered_n":
            len(answered),

        "accuracy_on_answered_pct":
            100 * answered_accuracy,

        "EM_div_nonrefusal_pct":
            100 * approx_formula,

        "refused_but_EM_hit_n":
            len(refused_but_em_hit),

        "mean_response_words_all":
            mean(all_lengths),

        "median_response_words_all":
            median(all_lengths),

        "mean_response_words_answered":
            mean(answered_lengths)
            if answered_lengths else 0,

        "median_response_words_answered":
            median(answered_lengths)
            if answered_lengths else 0,

        "mean_response_words_correct_answered":
            mean(correct_lengths)
            if correct_lengths else 0,
    }

    return result


results = []

for condition in CONDITIONS:

    selection, budget, root, marker = condition

    result = summarize(
        selection,
        budget,
        root,
        marker
    )

    results.append(result)


print()
print("=" * 122)
print("ABQ EVALUATION ANALYSIS")
print("=" * 122)

header = (
    f"{'Condition':<12}"
    f"{'ABQ N':>8}"
    f"{'Refuse%':>10}"
    f"{'EM_hit%':>10}"
    f"{'AnsAcc%':>11}"
    f"{'EM/(1-R)%':>12}"
    f"{'EM&Ref':>9}"
    f"{'Words all':>12}"
    f"{'Words ans':>12}"
)

print(header)
print("-" * len(header))

for r in results:

    name = (
        "BASE"
        if r["selection"] == "BASE"
        else f'{r["selection"]}_{r["budget"]}'
    )

    print(
        f"{name:<12}"
        f"{r['n_abq']:>8}"
        f"{r['refusal_rate_pct']:>10.2f}"
        f"{r['EM_hit_pct']:>10.2f}"
        f"{r['accuracy_on_answered_pct']:>11.2f}"
        f"{r['EM_div_nonrefusal_pct']:>12.2f}"
        f"{r['refused_but_EM_hit_n']:>9}"
        f"{r['mean_response_words_all']:>12.2f}"
        f"{r['mean_response_words_answered']:>12.2f}"
    )


output_csv = OUT_DIR / "abq_metric_and_length_analysis.csv"

with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


print()
print("=" * 122)
print("IMPORTANT")
print("=" * 122)

print(
    "accuracy_on_answered_pct is calculated DIRECTLY "
    "from non-refused examples."
)

print(
    "EM_div_nonrefusal_pct is the professor's approximate "
    "EM_hit / (1 - refusal rate) calculation."
)

print(
    "If refused_but_EM_hit_n > 0, the approximation is not "
    "exactly equal to conditional accuracy."
)

print()
print("Saved:")
print(output_csv)

