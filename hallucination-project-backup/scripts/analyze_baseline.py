import json
import re

RESULTS_PATH = "results/mistral_baseline_uaqfact_200.json"

with open(RESULTS_PATH) as f:
    data = json.load(f)


def is_abstention(text):
    text = text.lower().strip()

    phrases = [
        "i don't know",
        "i do not know",
        "cannot be answered",
        "can't be answered",
        "no such person",
        "no such individual",
        "no such answer",
        "there is no",
        "does not exist",
        "not possible",
        "cannot determine",
        "unable to determine",
    ]

    return any(p in text for p in phrases)


def get_reference_answers(reference):
    answers = []

    if isinstance(reference, list):
        for item in reference:
            if isinstance(item, dict):
                label = item.get("label")
                if label:
                    answers.append(label.lower())
            elif isinstance(item, str):
                answers.append(item.lower())

    elif isinstance(reference, str):
        answers.append(reference.lower())

    return answers


unanswerable = [
    x for x in data
    if x["category"] == "unanswerable"
]

answerable = [
    x for x in data
    if x["category"] == "answerable"
]


# -------------------------
# Unanswerable questions
# -------------------------

abstained = []

fabricated = []

for x in unanswerable:
    if is_abstention(x["model_response"]):
        abstained.append(x)
    else:
        fabricated.append(x)


abstention_rate = len(abstained) / len(unanswerable)
fabrication_rate = len(fabricated) / len(unanswerable)


# -------------------------
# Answerable questions
# -------------------------

correct = []

incorrect = []

for x in answerable:

    response = x["model_response"].lower()

    references = get_reference_answers(
        x["reference_answer"]
    )

    hit = any(
        ref in response
        for ref in references
        if ref != "none"
    )

    if hit:
        correct.append(x)
    else:
        incorrect.append(x)


accuracy = len(correct) / len(answerable)


print("=" * 60)
print("MISTRAL BASELINE RESULTS")
print("=" * 60)

print("\nUnanswerable questions:", len(unanswerable))

print(
    "Abstentions:",
    len(abstained),
    f"({abstention_rate:.1%})"
)

print(
    "Fabrications:",
    len(fabricated),
    f"({fabrication_rate:.1%})"
)

print("\nAnswerable questions:", len(answerable))

print(
    "Correct:",
    len(correct),
    f"({accuracy:.1%})"
)

print(
    "Incorrect:",
    len(incorrect),
    f"({1-accuracy:.1%})"
)


print("\n" + "=" * 60)
print("EXAMPLE FABRICATIONS")
print("=" * 60)

for x in fabricated[:5]:
    print("\nQuestion:", x["question"])
    print("Response:", x["model_response"])


print("\n" + "=" * 60)
print("EXAMPLE ABSTENTIONS")
print("=" * 60)

for x in abstained[:5]:
    print("\nQuestion:", x["question"])
    print("Response:", x["model_response"])
