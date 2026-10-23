import json
from pathlib import Path

CONDITIONS = [
    (
        "BASE",
        "results/official_baseline",
        "mistralai__Mistral-7B-Instruct-v0.3",
    ),
    (
        "IFD_05",
        "results/official_05pct",
        "mistral_lora_05pct",
    ),
]

SUBTASKS = ["dilemma", "inter", "time"]


def response(row):
    x = row.get("filtered_resps", row.get("resps", [""]))
    if isinstance(x, list):
        x = x[0]
        while isinstance(x, list) and x:
            x = x[0]
    return str(x)


for name, root, marker in CONDITIONS:

    overlaps = []

    for task in SUBTASKS:

        files = [
            p for p in Path(root).rglob(
                f"samples_task1_{task}_en_ab_*.jsonl"
            )
            if marker in str(p)
        ]

        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        path = files[0]

        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)

                if (
                    int(float(r["EM_hit"])) == 1
                    and int(float(r["NEC_refuse"])) == 1
                ):
                    overlaps.append({
                        "task": task,
                        "question": r["doc"].get("question", ""),
                        "response": response(r),
                        "references": r.get("target", ""),
                    })

    print("\n" + "=" * 80)
    print(name, "OVERLAP:", len(overlaps))
    print("=" * 80)

    for r in overlaps[:20]:
        print("\nTASK:", r["task"])
        print("QUESTION:", r["question"])
        print("REFERENCE:", r["references"])
        print("RESPONSE:", r["response"])
