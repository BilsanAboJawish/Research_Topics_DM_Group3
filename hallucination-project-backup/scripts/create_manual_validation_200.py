import csv
import json
import random
from pathlib import Path

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

RANDOM_SEED = 20261016

# We validate seed-42 models only.
# "marker" makes sure we choose the correct adapter directory.
CONDITIONS = [
    # selection, budget, result root, expected adapter marker
    ("IFD", "05", "results/official_05pct", "mistral_lora_05pct"),
    ("IFD", "10", "results/official_10pct", "mistral_lora_10pct"),
    ("IFD", "15", "results/official_15pct", "mistral_lora_15pct"),
    ("IFD", "20", "results/official_20pct", "mistral_lora_20pct"),
    ("IFD", "25", "results/official_25pct", "mistral_lora_25pct"),

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

# 4 + 3 + 3 = 10 examples per ABQ/UAQ
SUBTASK_QUOTA = {
    "dilemma": 4,
    "inter": 3,
    "time": 3,
}

OUTPUT_DIR = Path("results/manual_validation")
BLIND_FILE = OUTPUT_DIR / "manual_validation_200_blind.csv"
KEY_FILE = OUTPUT_DIR / "manual_validation_200_hidden_key.csv"
INSTRUCTIONS_FILE = OUTPUT_DIR / "manual_validation_instructions.txt"


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_response(record):
    """
    Prefer the post-processed response used by evaluation.
    """
    filtered = record.get("filtered_resps")

    if isinstance(filtered, list) and filtered:
        value = filtered[0]

        # Occasionally nested structures can appear.
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


def find_sample_file(root, marker, subtask, suffix):
    """
    suffix:
       ab -> answerable
       ua -> unanswerable
    """

    root = Path(root)

    pattern = f"samples_task1_{subtask}_en_{suffix}_*.jsonl"

    matches = [
        p for p in root.rglob(pattern)
        if marker in str(p)
    ]

    if not matches:
        raise FileNotFoundError(
            f"\nCould not find:\n"
            f"  root={root}\n"
            f"  marker={marker}\n"
            f"  pattern={pattern}\n"
        )

    # If a correct evaluation was run more than once,
    # use the newest matching sample file.
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if len(matches) > 1:
        print(
            f"WARNING: multiple matches for {marker}, "
            f"{subtask}, {suffix}; using newest:"
        )
        for p in matches:
            print("   ", p)

    return matches[0]


def safe_int(value):
    if value is None:
        return ""
    try:
        return int(float(value))
    except Exception:
        return value


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    selected = []

    print("=" * 80)
    print("BUILDING 200-OUTPUT MANUAL VALIDATION SET")
    print("=" * 80)

    for selection, budget, root, marker in CONDITIONS:

        print(f"\n{selection} {budget}%")

        for question_type, suffix in [("ABQ", "ab"), ("UAQ", "ua")]:

            count_for_type = 0

            for subtask, quota in SUBTASK_QUOTA.items():

                path = find_sample_file(
                    root=root,
                    marker=marker,
                    subtask=subtask,
                    suffix=suffix,
                )

                rows = read_jsonl(path)

                if len(rows) < quota:
                    raise ValueError(
                        f"Not enough rows in {path}: "
                        f"need {quota}, found {len(rows)}"
                    )

                chosen = rng.sample(rows, quota)

                for record in chosen:

                    doc = record.get("doc", {})

                    question = str(doc.get("question", "")).strip()
                    response = get_response(record)

                    if not question:
                        raise ValueError(
                            f"Missing question in {path}"
                        )

                    if not response:
                        print(
                            "WARNING: empty model response:",
                            path,
                            record.get("doc_id")
                        )

                    selected.append({
                        # Hidden metadata
                        "selection": selection,
                        "budget": budget,
                        "question_type": question_type,
                        "subtask": subtask,
                        "source_file": str(path),
                        "doc_id": record.get("doc_id", ""),
                        "qid": doc.get("qid", ""),
                        "official_NEC_refuse":
                            safe_int(record.get("NEC_refuse")),
                        "EM_hit":
                            safe_int(record.get("EM_hit")),

                        # Visible fields
                        "question": question,
                        "model_response": response,
                    })

                    count_for_type += 1

            if count_for_type != 10:
                raise AssertionError(
                    f"{selection} {budget} {question_type}: "
                    f"expected 10, got {count_for_type}"
                )

            print(
                f"  {question_type}: {count_for_type} "
                "(4 dilemma, 3 inter, 3 time)"
            )

    if len(selected) != 200:
        raise AssertionError(
            f"Expected 200 total examples, got {len(selected)}"
        )

    # Shuffle all conditions together before manual labeling.
    rng.shuffle(selected)

    # Give anonymous IDs AFTER shuffling.
    for i, row in enumerate(selected, start=1):
        row["validation_id"] = f"V{i:03d}"

    # --------------------------------------------------------
    # BLIND FILE
    # --------------------------------------------------------

    blind_fields = [
        "validation_id",
        "question",
        "model_response",
        "manual_label",
        "notes",
    ]

    with open(
        BLIND_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=blind_fields
        )

        writer.writeheader()

        for row in selected:
            writer.writerow({
                "validation_id": row["validation_id"],
                "question": row["question"],
                "model_response": row["model_response"],
                "manual_label": "",
                "notes": "",
            })

    # --------------------------------------------------------
    # HIDDEN KEY
    # DO NOT OPEN UNTIL MANUAL LABELING IS FINISHED
    # --------------------------------------------------------

    key_fields = [
        "validation_id",
        "selection",
        "budget",
        "question_type",
        "subtask",
        "official_NEC_refuse",
        "EM_hit",
        "doc_id",
        "qid",
        "source_file",
        "question",
        "model_response",
    ]

    with open(
        KEY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=key_fields
        )

        writer.writeheader()

        for row in selected:
            writer.writerow({
                field: row.get(field, "")
                for field in key_fields
            })

    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    instructions = """MANUAL VALIDATION INSTRUCTIONS

Goal:
Validate whether UAQFact's automatic NEC_refuse metric corresponds
to human judgment of model refusal behavior.

IMPORTANT:
Do NOT open manual_validation_200_hidden_key.csv until all 200
manual labels are complete.

For every response, read only the QUESTION and MODEL RESPONSE.

Assign exactly one manual label:

R = REFUSAL
    The model explicitly declines to answer, says it cannot answer,
    says the required information is unavailable/unknown, or otherwise
    clearly abstains.

A = ANSWER
    The model attempts to answer the factual question.
    Label A even when the attempted answer appears incorrect.

P = PREMISE_REJECTION
    The model does not merely refuse. Instead, it explicitly says the
    premise/assumption in the question is false, invalid, or mistaken,
    and often provides a correction.

M = AMBIGUOUS
    The response cannot confidently be classified as R, A, or P.

Examples:

Question:
"What work does Spaceballs reference, Gramatica occitana...?"

Response:
"Spaceballs references Star Wars, not Gramatica occitana..."

Label:
P

Do not use correctness as the criterion for R vs A.
An incorrect attempted answer is still A.

Use the Notes column only when useful, especially for ambiguous or
interesting cases.
"""

    with open(
        INSTRUCTIONS_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(instructions)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("Total examples:", len(selected))
    print()
    print("BLIND FILE:")
    print(BLIND_FILE)
    print()
    print("HIDDEN KEY — DO NOT OPEN YET:")
    print(KEY_FILE)
    print()
    print("LABELING INSTRUCTIONS:")
    print(INSTRUCTIONS_FILE)


if __name__ == "__main__":
    main()
