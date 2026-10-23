import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


FULL_FILE = (
    "Superfiltering/data/data_with_ifd/"
    "alpaca_gpt4_data_gpt2_data.json"
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

BUDGETS = [5, 10, 15, 20, 25]

OUT_DIR = Path("results/full_ifd_pool_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def canonical_key(x):
    return (
        str(x.get("instruction", "")).strip(),
        str(x.get("input", "")).strip(),
        str(x.get("output", "")).strip(),
    )


def word_count(text):
    return len(str(text).split())


def classify_task(instruction):
    """
    Same broad heuristic categories used in our previous analysis.
    These are exploratory, not gold-standard task labels.
    """
    t = str(instruction).lower().strip()

    if re.search(r"\btranslate\b|\btranslation\b", t):
        return "translation"

    if re.search(
        r"\bsummarize\b|\bsummarise\b|\bsummary\b",
        t
    ):
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
        t.startswith(
            (
                "what ",
                "who ",
                "when ",
                "where ",
                "why ",
                "how ",
                "which ",
                "is ",
                "are ",
                "does ",
                "do ",
            )
        )
        or "explain" in t
        or "describe" in t
    ):
        return "qa_explanation"

    return "other"


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 2:
        return np.nan

    return np.corrcoef(
        x[mask],
        y[mask]
    )[0, 1]


def spearman(x, y):
    """
    Spearman = Pearson correlation between ranks.
    This avoids requiring scipy.
    """
    sx = pd.Series(x).rank(method="average")
    sy = pd.Series(y).rank(method="average")

    return sx.corr(sy, method="pearson")


# ============================================================
# LOAD FULL DATA
# ============================================================

print("Loading full Alpaca-GPT4 IFD-scored pool...")

data = load_json(FULL_FILE)

print("N =", len(data))

if len(data) != 52002:
    print(
        "WARNING: expected 52,002 examples, "
        f"found {len(data)}"
    )


# ============================================================
# BUILD DATAFRAME
# ============================================================

rows = []

for x in data:

    instruction = str(
        x.get("instruction", "")
    ).strip()

    inp = str(
        x.get("input", "")
    ).strip()

    output = str(
        x.get("output", "")
    ).strip()

    rows.append({
        "key": canonical_key(x),

        "ifd_ppl":
            float(x["ifd_ppl"]),

        "ppl_A_condition":
            float(x["ppl_A_condition"]),

        "ppl_A_direct":
            float(x["ppl_A_direct"]),

        "instruction_words":
            word_count(instruction),

        "input_words":
            word_count(inp),

        "response_words":
            word_count(output),

        "task":
            classify_task(instruction),

        "output_text":
            output,
    })


df = pd.DataFrame(rows)


# ============================================================
# MISTRAL OUTPUT TOKEN LENGTH
# ============================================================

print("\nLoading Mistral tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizing 52,002 outputs...")

texts = df["output_text"].tolist()

token_lengths = []

BATCH = 512

for start in range(0, len(texts), BATCH):

    batch = texts[start:start + BATCH]

    encoded = tokenizer(
        batch,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )

    token_lengths.extend(
        len(ids)
        for ids in encoded["input_ids"]
    )

    if start % 5000 < BATCH:
        print(
            f"  processed "
            f"{min(start + BATCH, len(texts))}"
            f"/{len(texts)}"
        )


df["response_tokens"] = token_lengths


# ============================================================
# BASIC FULL-POOL STATISTICS
# ============================================================

print("\n")
print("=" * 90)
print("FULL POOL")
print("=" * 90)

print("N:", len(df))

print(
    "IFD range:",
    df["ifd_ppl"].min(),
    "to",
    df["ifd_ppl"].max(),
)

print(
    "IFD mean:",
    df["ifd_ppl"].mean()
)

print(
    "IFD median:",
    df["ifd_ppl"].median()
)

print(
    "Response words mean:",
    df["response_words"].mean()
)

print(
    "Response words median:",
    df["response_words"].median()
)

print(
    "Response tokens mean:",
    df["response_tokens"].mean()
)

print(
    "Response tokens median:",
    df["response_tokens"].median()
)


# ============================================================
# CORRELATIONS
# ============================================================

correlation_rows = []

features = [
    "response_words",
    "response_tokens",
    "instruction_words",
    "input_words",
    "ppl_A_condition",
    "ppl_A_direct",
]

print("\n")
print("=" * 90)
print("FULL-POOL CORRELATIONS WITH IFD SCORE")
print("=" * 90)

for feature in features:

    p = pearson(
        df["ifd_ppl"],
        df[feature]
    )

    s = spearman(
        df["ifd_ppl"],
        df[feature]
    )

    correlation_rows.append({
        "feature": feature,
        "pearson": p,
        "spearman": s,
    })

    print(
        f"{feature:<24}"
        f" Pearson={p:+.4f}"
        f"  Spearman={s:+.4f}"
    )


# ============================================================
# VERIFY ACTUAL SELECTION DIRECTION
# ============================================================

print("\n")
print("=" * 90)
print("VERIFY IFD SUBSET RANKING DIRECTION")
print("=" * 90)

ranking_rows = []

for pct in BUDGETS:

    n = int(len(df) * pct / 100)

    subset_file = (
        f"data/alpaca_gpt4_ifd_{pct:02d}pct.json"
    )

    subset = load_json(subset_file)

    actual_keys = {
        canonical_key(x)
        for x in subset
    }

    ascending = (
        df.sort_values(
            "ifd_ppl",
            ascending=True
        )
        .head(n)
    )

    descending = (
        df.sort_values(
            "ifd_ppl",
            ascending=False
        )
        .head(n)
    )

    ascending_keys = set(
        ascending["key"]
    )

    descending_keys = set(
        descending["key"]
    )

    asc_overlap = (
        len(actual_keys & ascending_keys)
        / len(actual_keys)
    )

    desc_overlap = (
        len(actual_keys & descending_keys)
        / len(actual_keys)
    )

    ranking_rows.append({
        "budget_pct": pct,
        "n": len(actual_keys),
        "ascending_overlap_pct":
            100 * asc_overlap,
        "descending_overlap_pct":
            100 * desc_overlap,
    })

    print(
        f"{pct:>2}% "
        f"N={len(actual_keys):>5} | "
        f"lowest-IFD match={100*asc_overlap:6.2f}% | "
        f"highest-IFD match={100*desc_overlap:6.2f}%"
    )


# Decide direction using 5% subset
direction = (
    "highest"
    if ranking_rows[0][
        "descending_overlap_pct"
    ]
    >
    ranking_rows[0][
        "ascending_overlap_pct"
    ]
    else "lowest"
)

print(
    "\nDetected selection direction:",
    direction.upper(),
    "IFD scores"
)


# ============================================================
# SCORE DECILES
# ============================================================

# Rank from low score to high score.
df["ifd_decile"] = pd.qcut(
    df["ifd_ppl"],
    10,
    labels=False,
    duplicates="drop"
) + 1


decile_rows = []

print("\n")
print("=" * 90)
print("IFD SCORE DECILES")
print("Decile 1 = lowest scores; Decile 10 = highest scores")
print("=" * 90)

print(
    f"{'Decile':<8}"
    f"{'N':>7}"
    f"{'IFD mean':>12}"
    f"{'Words mean':>13}"
    f"{'Words med':>11}"
    f"{'Tokens mean':>13}"
    f"{'<=10w%':>10}"
    f"{'>=200w%':>11}"
)

for decile, g in df.groupby(
    "ifd_decile",
    observed=True
):

    row = {
        "decile": int(decile),
        "n": len(g),

        "ifd_mean":
            g["ifd_ppl"].mean(),

        "response_words_mean":
            g["response_words"].mean(),

        "response_words_median":
            g["response_words"].median(),

        "response_tokens_mean":
            g["response_tokens"].mean(),

        "response_le10_pct":
            100 * (
                g["response_words"] <= 10
            ).mean(),

        "response_ge200_pct":
            100 * (
                g["response_words"] >= 200
            ).mean(),
    }

    decile_rows.append(row)

    print(
        f"{row['decile']:<8}"
        f"{row['n']:>7}"
        f"{row['ifd_mean']:>12.4f}"
        f"{row['response_words_mean']:>13.1f}"
        f"{row['response_words_median']:>11.1f}"
        f"{row['response_tokens_mean']:>13.1f}"
        f"{row['response_le10_pct']:>10.2f}"
        f"{row['response_ge200_pct']:>11.2f}"
    )


# ============================================================
# TOP SELECTED BUDGETS VS FULL POOL
# ============================================================

print("\n")
print("=" * 90)
print("SELECTED IFD BUDGETS VS FULL POOL")
print("=" * 90)

budget_rows = []

full_stats = {
    "budget": "100/full",

    "n": len(df),

    "ifd_mean":
        df["ifd_ppl"].mean(),

    "words_mean":
        df["response_words"].mean(),

    "words_median":
        df["response_words"].median(),

    "tokens_mean":
        df["response_tokens"].mean(),

    "short_pct":
        100 * (
            df["response_words"] <= 10
        ).mean(),

    "long_pct":
        100 * (
            df["response_words"] >= 200
        ).mean(),
}

budget_rows.append(full_stats)


for pct in BUDGETS:

    n = int(len(df) * pct / 100)

    if direction == "highest":
        selected = df.nlargest(
            n,
            "ifd_ppl"
        )
    else:
        selected = df.nsmallest(
            n,
            "ifd_ppl"
        )

    budget_rows.append({
        "budget": f"{pct}%",

        "n": len(selected),

        "ifd_mean":
            selected["ifd_ppl"].mean(),

        "words_mean":
            selected["response_words"].mean(),

        "words_median":
            selected["response_words"].median(),

        "tokens_mean":
            selected["response_tokens"].mean(),

        "short_pct":
            100 * (
                selected["response_words"] <= 10
            ).mean(),

        "long_pct":
            100 * (
                selected["response_words"] >= 200
            ).mean(),
    })


print(
    f"{'Budget':<10}"
    f"{'N':>7}"
    f"{'IFD mean':>12}"
    f"{'Words':>11}"
    f"{'Median':>11}"
    f"{'Tokens':>11}"
    f"{'<=10w%':>10}"
    f"{'>=200w%':>11}"
)

for r in budget_rows:

    print(
        f"{r['budget']:<10}"
        f"{r['n']:>7}"
        f"{r['ifd_mean']:>12.4f}"
        f"{r['words_mean']:>11.1f}"
        f"{r['words_median']:>11.1f}"
        f"{r['tokens_mean']:>11.1f}"
        f"{r['short_pct']:>10.2f}"
        f"{r['long_pct']:>11.2f}"
    )


# ============================================================
# WITHIN-TASK CORRELATIONS
# ============================================================

print("\n")
print("=" * 90)
print("WITHIN-TASK: IFD SCORE vs RESPONSE LENGTH")
print("=" * 90)

task_rows = []

print(
    f"{'Task':<30}"
    f"{'N':>7}"
    f"{'Words mean':>13}"
    f"{'Pearson':>11}"
    f"{'Spearman':>11}"
)

for task, g in (
    df.groupby("task")
    .filter(lambda x: len(x) >= 30)
    .groupby("task")
):

    p = pearson(
        g["ifd_ppl"],
        g["response_words"]
    )

    s = spearman(
        g["ifd_ppl"],
        g["response_words"]
    )

    task_rows.append({
        "task": task,
        "n": len(g),
        "response_words_mean":
            g["response_words"].mean(),
        "pearson":
            p,
        "spearman":
            s,
    })

    print(
        f"{task:<30}"
        f"{len(g):>7}"
        f"{g['response_words'].mean():>13.1f}"
        f"{p:>11.4f}"
        f"{s:>11.4f}"
    )


# ============================================================
# TASK-CONTROLLED ASSOCIATION
# ============================================================

# Remove each task category's mean from both score
# and response length.
df["ifd_task_centered"] = (
    df["ifd_ppl"]
    - df.groupby("task")["ifd_ppl"]
        .transform("mean")
)

df["words_task_centered"] = (
    df["response_words"]
    - df.groupby("task")["response_words"]
        .transform("mean")
)

task_controlled_r = pearson(
    df["ifd_task_centered"],
    df["words_task_centered"]
)

print("\nTask-controlled Pearson association:")
print(
    "IFD score vs response words after "
    f"removing task-category means = "
    f"{task_controlled_r:+.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

pd.DataFrame(
    correlation_rows
).to_csv(
    OUT_DIR / "full_pool_correlations.csv",
    index=False
)

pd.DataFrame(
    ranking_rows
).to_csv(
    OUT_DIR / "selection_direction_check.csv",
    index=False
)

pd.DataFrame(
    decile_rows
).to_csv(
    OUT_DIR / "ifd_deciles.csv",
    index=False
)

pd.DataFrame(
    budget_rows
).to_csv(
    OUT_DIR / "selected_budget_statistics.csv",
    index=False
)

pd.DataFrame(
    task_rows
).to_csv(
    OUT_DIR / "within_task_correlations.csv",
    index=False
)

df[
    [
        "ifd_ppl",
        "response_words",
        "response_tokens",
        "instruction_words",
        "input_words",
        "task",
    ]
].to_csv(
    OUT_DIR / "full_pool_example_statistics.csv",
    index=False
)


print("\n")
print("=" * 90)
print("SAVED")
print("=" * 90)

print(
    OUT_DIR / "full_pool_correlations.csv"
)

print(
    OUT_DIR / "selection_direction_check.csv"
)

print(
    OUT_DIR / "ifd_deciles.csv"
)

print(
    OUT_DIR / "selected_budget_statistics.csv"
)

print(
    OUT_DIR / "within_task_correlations.csv"
)
