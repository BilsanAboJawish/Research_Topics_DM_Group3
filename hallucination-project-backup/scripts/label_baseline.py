import json
import csv
import os

INPUT = "results/mistral_baseline_uaqfact_200.json"
OUTPUT = "results/baseline_manual_labels.csv"

with open(INPUT) as f:
    data = json.load(f)

# Resume previous work if the file already exists
existing = {}

if os.path.exists(OUTPUT):
    with open(OUTPUT, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[row["qid"]] = row


fieldnames = [
    "qid",
    "category",
    "question",
    "reference_answer",
    "model_response",
    "manual_label",
    "notes",
]


with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i, x in enumerate(data, 1):

        qid = x["qid"]

        # Keep already-labelled rows
        if qid in existing and existing[qid]["manual_label"]:
            writer.writerow(existing[qid])
            continue

        print("\n" + "=" * 90)
        print(f"{i}/{len(data)}")
        print("ID:", qid)
        print("TYPE:", x["category"])
        print("\nQUESTION:")
        print(x["question"])

        print("\nREFERENCE:")
        print(x["reference_answer"])

        print("\nMODEL RESPONSE:")
        print(x["model_response"])

        print("\nChoose label:")

        if x["category"] == "unanswerable":
            print("  a = abstention / correctly rejects premise")
            print("  f = fabrication")
            print("  m = ambiguous")
            allowed = {
                "a": "abstention",
                "f": "fabrication",
                "m": "ambiguous",
            }

        else:
            print("  c = correct")
            print("  i = incorrect")
            print("  m = ambiguous")
            allowed = {
                "c": "correct",
                "i": "incorrect",
                "m": "ambiguous",
            }

        while True:
            choice = input("\nLabel: ").strip().lower()

            if choice in allowed:
                break

            print("Invalid choice.")

        row = {
            "qid": qid,
            "category": x["category"],
            "question": x["question"],
            "reference_answer": json.dumps(x["reference_answer"]),
            "model_response": x["model_response"],
            "manual_label": allowed[choice],
            "notes": "",
        }

        writer.writerow(row)
        f.flush()


print("\nFinished.")
print("Saved to:", OUTPUT)
