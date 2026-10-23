import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_LENGTH = 1024
EFFECTIVE_BATCH = 8
FIXED_STEPS = 975

BUDGETS = ["05", "10", "15", "20", "25"]

OUT = Path("results/ifd_random_deep_analysis")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {}

for pct in BUDGETS:
    FILES[("IFD", pct)] = (
        f"data/alpaca_gpt4_ifd_{pct}pct.json"
    )

    FILES[("Random", pct)] = (
        f"data/alpaca_gpt4_random_{pct}pct_seed42.json"
    )


# Seed-42 refusal results, already evaluated.
MODEL_RESULTS = {
    ("IFD", "05"): (24.65, 50.41),
    ("IFD", "10"): (39.16, 67.93),
    ("IFD", "15"): (41.02, 72.88),
    ("IFD", "20"): (40.64, 70.52),
    ("IFD", "25"): (46.27, 76.01),

    ("Random", "05"): (49.61, 80.42),
    ("Random", "10"): (49.49, 79.31),
    ("Random", "15"): (55.65, 84.68),
    ("Random", "20"): (59.48, 87.72),
    ("Random", "25"): (49.99, 77.80),
}


# ============================================================
# LEXICAL PATTERNS
# ============================================================

REFUSAL_PATTERNS = [
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bi am unable\b",
    r"\bi'm unable\b",
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\bnot enough information\b",
    r"\binsufficient information\b",
    r"\bas an ai\b",
    r"\bas a language model\b",
]

HEDGE_PATTERNS = [
    r"\bi am not sure\b",
    r"\bi'm not sure\b",
    r"\bit is unclear\b",
    r"\bit's unclear\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bprobably\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bit depends\b",
]

POLITE_STARTS = (
    "sure",
    "certainly",
    "of course",
    "here is",
    "here's",
    "absolutely",
)


# ============================================================
# HELPERS
# ============================================================

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def canonical_key(x):
    return (
        str(x.get("instruction", "")).strip(),
        str(x.get("input", "")).strip(),
        str(x.get("output", "")).strip(),
    )


def matches(text, patterns):
    return any(
        re.search(p, text, flags=re.IGNORECASE)
        for p in patterns
    )


def word_count(text):
    return len(str(text).split())


def sentence_count(text):
    chunks = re.split(r"[.!?]+", str(text))
    return sum(1 for x in chunks if x.strip())


def classify_task(instruction):
    """
    Exploratory heuristic task classification.
    This is NOT treated as a gold-standard label.
    """

    t = instruction.lower().strip()

    if re.search(
        r"\btranslate\b|\btranslation\b",
        t
    ):
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


def percentile(values, p):
    if not values:
        return float("nan")

    values = sorted(values)

    k = (len(values) - 1) * p

    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return values[int(k)]

    return (
        values[f] * (c - k)
        + values[c] * (k - f)
    )


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    mx = mean(xs)
    my = mean(ys)

    num = sum(
        (x - mx) * (y - my)
        for x, y in zip(xs, ys)
    )

    dx = math.sqrt(
        sum((x - mx) ** 2 for x in xs)
    )

    dy = math.sqrt(
        sum((y - my) ** 2 for y in ys)
    )

    if dx == 0 or dy == 0:
        return float("nan")

    return num / (dx * dy)


# ============================================================
# TOKENIZER
# ============================================================

print("Loading Mistral tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer truncation side:",
      tokenizer.truncation_side)


def token_stats(example):
    instruction = example["instruction"].strip()
    inp = example.get("input", "").strip()
    output = example["output"].strip()

    if inp:
        user_content = (
            instruction
            + "\n\nInput:\n"
            + inp
        )
    else:
        user_content = instruction

    user_messages = [
        {
            "role": "user",
            "content": user_content
        }
    ]

    full_messages = [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": output
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False
    )["input_ids"]

    prompt_tokens = len(prompt_ids)
    full_tokens = len(full_ids)

    completion_tokens = max(
        0,
        full_tokens - prompt_tokens
    )

    # Approximation under ordinary right truncation.
    remaining = max(
        0,
        MAX_LENGTH - prompt_tokens
    )

    effective_completion_tokens = min(
        completion_tokens,
        remaining
    )

    return (
        prompt_tokens,
        completion_tokens,
        full_tokens,
        effective_completion_tokens,
    )


# ============================================================
# ANALYSE ONE SUBSET
# ============================================================

def analyse_subset(selection, budget, data):

    instruction_words = []
    input_words = []
    response_words = []
    sentences = []

    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    effective_completion_tokens = []

    task_counts = Counter()

    exact_none = 0
    very_short = 0
    short = 0
    very_long = 0

    refusals = 0
    hedges = 0
    polite_start = 0
    first_person = 0

    bullets = 0
    numbered = 0
    code_blocks = 0

    ifd_scores = []

    print(
        f"\nAnalysing {selection} {budget}% "
        f"({len(data)} examples)..."
    )

    for i, x in enumerate(data, start=1):

        instruction = str(
            x.get("instruction", "")
        ).strip()

        inp = str(
            x.get("input", "")
        ).strip()

        response = str(
            x.get("output", "")
        ).strip()

        iw = word_count(instruction)
        rw = word_count(response)

        instruction_words.append(iw)
        input_words.append(word_count(inp))
        response_words.append(rw)
        sentences.append(sentence_count(response))

        task_counts[
            classify_task(instruction)
        ] += 1

        normalized = (
            response.lower()
            .strip()
            .strip(".")
            .strip()
        )

        if normalized in {
            "none",
            "##none##",
            "n/a",
            "unknown",
            "i don't know",
            "i do not know",
        }:
            exact_none += 1

        if rw <= 3:
            very_short += 1

        if rw <= 10:
            short += 1

        if rw >= 200:
            very_long += 1

        if matches(response, REFUSAL_PATTERNS):
            refusals += 1

        if matches(response, HEDGE_PATTERNS):
            hedges += 1

        if response.lower().startswith(POLITE_STARTS):
            polite_start += 1

        if re.search(
            r"\b(i|i'm|i've|i'd|my)\b",
            response.lower()
        ):
            first_person += 1

        if re.search(
            r"(^|\n)\s*[-*]\s+",
            response
        ):
            bullets += 1

        if re.search(
            r"(^|\n)\s*\d+[.)]\s+",
            response
        ):
            numbered += 1

        if "```" in response:
            code_blocks += 1

        if "ifd_ppl" in x:
            try:
                ifd_scores.append(
                    float(x["ifd_ppl"])
                )
            except Exception:
                pass

        pt, ct, tt, ect = token_stats(x)

        prompt_tokens.append(pt)
        completion_tokens.append(ct)
        total_tokens.append(tt)
        effective_completion_tokens.append(ect)

        if i % 1000 == 0:
            print(
                f"  processed {i}/{len(data)}"
            )

    n = len(data)

    result = {
        "selection": selection,
        "budget": budget,
        "n": n,

        "instruction_words_mean":
            mean(instruction_words),

        "response_words_mean":
            mean(response_words),

        "response_words_median":
            median(response_words),

        "response_words_p90":
            percentile(response_words, .90),

        "response_sentences_mean":
            mean(sentences),

        "prompt_tokens_mean":
            mean(prompt_tokens),

        "completion_tokens_mean":
            mean(completion_tokens),

        "completion_tokens_median":
            median(completion_tokens),

        "total_tokens_mean":
            mean(total_tokens),

        "effective_completion_tokens_mean":
            mean(effective_completion_tokens),

        "examples_over_1024_pct":
            100 * sum(
                x > MAX_LENGTH
                for x in total_tokens
            ) / n,

        "exact_none_pct":
            100 * exact_none / n,

        "response_le3_words_pct":
            100 * very_short / n,

        "response_le10_words_pct":
            100 * short / n,

        "response_ge200_words_pct":
            100 * very_long / n,

        "refusal_phrase_pct":
            100 * refusals / n,

        "hedge_phrase_pct":
            100 * hedges / n,

        "polite_start_pct":
            100 * polite_start / n,

        "first_person_pct":
            100 * first_person / n,

        "bullet_pct":
            100 * bullets / n,

        "numbered_pct":
            100 * numbered / n,

        "code_block_pct":
            100 * code_blocks / n,

        "total_effective_completion_tokens":
            sum(effective_completion_tokens),

        # Approx assistant-target tokens processed
        # in the new matched-975-step control.
        "fixed975_est_completion_tokens":
            FIXED_STEPS
            * EFFECTIVE_BATCH
            * mean(effective_completion_tokens),

        "ifd_score_mean":
            mean(ifd_scores)
            if ifd_scores else float("nan"),

        "ifd_score_median":
            median(ifd_scores)
            if ifd_scores else float("nan"),
    }

    category_rows = []

    for category, count in sorted(
        task_counts.items()
    ):
        category_rows.append({
            "selection": selection,
            "budget": budget,
            "category": category,
            "count": count,
            "pct": 100 * count / n,
        })

    return result, category_rows


# ============================================================
# RUN
# ============================================================

summaries = []
category_rows = []
datasets = {}

for selection in ["IFD", "Random"]:
    for budget in BUDGETS:

        path = FILES[(selection, budget)]

        data = load(path)

        datasets[(selection, budget)] = data

        summary, cats = analyse_subset(
            selection,
            budget,
            data
        )

        summaries.append(summary)
        category_rows.extend(cats)


# ============================================================
# ADD MODEL BEHAVIOUR
# ============================================================

for r in summaries:

    key = (
        r["selection"],
        r["budget"]
    )

    if key in MODEL_RESULTS:
        abq, uaq = MODEL_RESULTS[key]

        r["ABQ_refusal_pct"] = abq
        r["UAQ_refusal_pct"] = uaq
        r["refusal_gap_pp"] = uaq - abq


# ============================================================
# OVERLAP IFD vs RANDOM
# ============================================================

overlap_rows = []

for budget in BUDGETS:

    ifd = {
        canonical_key(x)
        for x in datasets[("IFD", budget)]
    }

    rnd = {
        canonical_key(x)
        for x in datasets[("Random", budget)]
    }

    intersection = len(ifd & rnd)
    union = len(ifd | rnd)

    overlap_rows.append({
        "budget": budget,
        "IFD_n": len(ifd),
        "Random_n": len(rnd),
        "overlap_n": intersection,

        "overlap_pct_of_IFD":
            100 * intersection / len(ifd),

        "overlap_pct_of_Random":
            100 * intersection / len(rnd),

        "jaccard_pct":
            100 * intersection / union,
    })


# ============================================================
# PAIRED DIFFERENCE TABLE
# ============================================================

summary_map = {
    (x["selection"], x["budget"]): x
    for x in summaries
}

features = [
    "response_words_mean",
    "completion_tokens_mean",
    "effective_completion_tokens_mean",
    "examples_over_1024_pct",
    "exact_none_pct",
    "response_le3_words_pct",
    "response_le10_words_pct",
    "response_ge200_words_pct",
    "refusal_phrase_pct",
    "hedge_phrase_pct",
    "polite_start_pct",
    "first_person_pct",
    "bullet_pct",
    "numbered_pct",
    "code_block_pct",
]

difference_rows = []

for budget in BUDGETS:

    a = summary_map[("IFD", budget)]
    b = summary_map[("Random", budget)]

    for feature in features:

        difference_rows.append({
            "budget": budget,
            "feature": feature,
            "IFD": a[feature],
            "Random": b[feature],
            "Random_minus_IFD":
                b[feature] - a[feature],
        })


# ============================================================
# CORRELATIONS WITH REFUSAL
# ============================================================

corr_rows = []

correlation_features = [
    "response_words_mean",
    "completion_tokens_mean",
    "effective_completion_tokens_mean",
    "exact_none_pct",
    "response_le3_words_pct",
    "response_le10_words_pct",
    "response_ge200_words_pct",
    "refusal_phrase_pct",
    "hedge_phrase_pct",
    "polite_start_pct",
    "first_person_pct",
]

for feature in correlation_features:

    xs = [r[feature] for r in summaries]

    ys_abq = [
        r["ABQ_refusal_pct"]
        for r in summaries
    ]

    ys_uaq = [
        r["UAQ_refusal_pct"]
        for r in summaries
    ]

    corr_rows.append({
        "feature": feature,
        "pearson_ABQ_refusal":
            pearson(xs, ys_abq),
        "pearson_UAQ_refusal":
            pearson(xs, ys_uaq),
    })


# ============================================================
# SAVE
# ============================================================

def write_csv(path, rows):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)


write_csv(
    OUT / "subset_summary.csv",
    summaries
)

write_csv(
    OUT / "task_categories.csv",
    category_rows
)

write_csv(
    OUT / "ifd_random_overlap.csv",
    overlap_rows
)

write_csv(
    OUT / "paired_differences.csv",
    difference_rows
)

write_csv(
    OUT / "feature_refusal_correlations.csv",
    corr_rows
)


# ============================================================
# PRINT MAIN RESULTS
# ============================================================

print("\n\n")
print("=" * 120)
print("MAIN IFD vs RANDOM COMPOSITION SUMMARY")
print("=" * 120)

print(
    f"{'Cond':<11}"
    f"{'RespWords':>11}"
    f"{'CompTok':>10}"
    f"{'<=3w%':>9}"
    f"{'<=10w%':>10}"
    f"{'None%':>9}"
    f"{'Ref%':>9}"
    f"{'Hedge%':>9}"
    f"{'>1024%':>9}"
    f"{'ABQ-R':>9}"
    f"{'UAQ-R':>9}"
)

for r in summaries:

    name = (
        f"{r['selection']}_{r['budget']}"
    )

    print(
        f"{name:<11}"
        f"{r['response_words_mean']:>11.1f}"
        f"{r['completion_tokens_mean']:>10.1f}"
        f"{r['response_le3_words_pct']:>9.2f}"
        f"{r['response_le10_words_pct']:>10.2f}"
        f"{r['exact_none_pct']:>9.2f}"
        f"{r['refusal_phrase_pct']:>9.2f}"
        f"{r['hedge_phrase_pct']:>9.2f}"
        f"{r['examples_over_1024_pct']:>9.2f}"
        f"{r['ABQ_refusal_pct']:>9.2f}"
        f"{r['UAQ_refusal_pct']:>9.2f}"
    )


print("\n")
print("=" * 120)
print("IFD vs RANDOM DATA OVERLAP")
print("=" * 120)

for r in overlap_rows:
    print(
        f"{r['budget']}%: "
        f"overlap={r['overlap_n']} "
        f"({r['overlap_pct_of_IFD']:.2f}% of IFD), "
        f"Jaccard={r['jaccard_pct']:.2f}%"
    )


print("\n")
print("=" * 120)
print("FEATURE CORRELATIONS WITH MODEL REFUSAL")
print("=" * 120)
print(
    "Exploratory only: n=10 conditions and budgets are not independent."
)

for r in corr_rows:
    print(
        f"{r['feature']:<35}"
        f" ABQ r={r['pearson_ABQ_refusal']:+.3f}"
        f"  UAQ r={r['pearson_UAQ_refusal']:+.3f}"
    )


print("\nSaved to:")
print(OUT)
