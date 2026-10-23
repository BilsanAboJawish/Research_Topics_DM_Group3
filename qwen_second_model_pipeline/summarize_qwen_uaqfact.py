#!/usr/bin/env python3
import argparse, json, math, re
from pathlib import Path
from statistics import NormalDist

def newest_result_file(root):
    files = list(Path(root).rglob("results_*.json"))
    if not files:
        files = list(Path(root).rglob("*.json"))
    files = [p for p in files if p.name != "run_info.json"]
    if not files:
        raise FileNotFoundError(f"No result JSON found under {root}")
    return max(files, key=lambda p: p.stat().st_mtime)

def metric_value(d, needle):
    for k, v in d.items():
        if needle.lower() in k.lower() and isinstance(v, (int, float)):
            return float(v)
    return None

def n_for_task(blob, task):
    for key in ["n-samples", "n_samples", "nsamples"]:
        obj = blob.get(key)
        if isinstance(obj, dict) and task in obj:
            val = obj[task]
            if isinstance(val, dict):
                for kk in ["effective", "original", "n"]:
                    if kk in val:
                        return int(val[kk])
            if isinstance(val, (int, float)):
                return int(val)
    return None

def aggregate(root):
    f = newest_result_file(root)
    blob = json.load(open(f))
    results = blob.get("results", {})
    rows = []

    for task, vals in results.items():
        t = task.lower()
        if "_en_ab" in t:
            cls = "ABQ"
        elif "_en_ua" in t:
            cls = "UAQ"
        else:
            continue

        refusal = metric_value(vals, "nec_refuse")
        em = metric_value(vals, "em_hit")
        n = n_for_task(blob, task)

        # Fallback: count logged samples for this task.
        if n is None:
            sample_files = list(Path(root).rglob(f"*{task}*samples*.jsonl"))
            if sample_files:
                n = sum(sum(1 for _ in open(sf)) for sf in sample_files)

        rows.append((task, cls, n, refusal, em))

    if not rows:
        raise RuntimeError(f"No English Task-1 rows found in {f}")

    out = {"source_file": str(f), "tasks": []}
    for task, cls, n, refusal, em in rows:
        out["tasks"].append(
            {"task": task, "class": cls, "n": n, "refusal": refusal, "em_hit": em}
        )

    def wavg(cls, col):
        vals = [(r[2], r[col]) for r in rows if r[1] == cls and r[col] is not None]
        if not vals:
            return None
        if all(n is not None for n, _ in vals):
            return sum(n*v for n, v in vals) / sum(n for n, _ in vals)
        # If sample counts are unavailable, report the unweighted mean and flag it.
        out["WARNING"] = "Task sample counts unavailable; aggregate is unweighted."
        return sum(v for _, v in vals) / len(vals)

    abq_ref = wavg("ABQ", 3)
    uaq_ref = wavg("UAQ", 3)
    abq_em = wavg("ABQ", 4)

    # lm-eval metrics can be proportions or percentages. Normalize to proportions.
    def prop(x):
        if x is None:
            return None
        return x / 100.0 if x > 1.0 else x

    ar, ur = prop(abq_ref), prop(uaq_ref)
    gap = None if ar is None or ur is None else ur - ar
    dprime = None
    if ar is not None and ur is not None:
        eps = 1e-6
        ar2 = min(max(ar, eps), 1-eps)
        ur2 = min(max(ur, eps), 1-eps)
        dprime = NormalDist().inv_cdf(ur2) - NormalDist().inv_cdf(ar2)

    out["aggregate"] = {
        "ABQ_refusal": abq_ref,
        "UAQ_refusal": uaq_ref,
        "gap": gap,
        "dprime": dprime,
        "ABQ_EM_hit": abq_em,
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    x = aggregate(args.root)
    print(json.dumps(x, indent=2))
    agg = x["aggregate"]

    def pct(v):
        if v is None:
            return "NA"
        return f"{100*v:.2f}%" if abs(v) <= 1.0 else f"{v:.2f}%"

    print("\nSUMMARY")
    print("condition:", args.name)
    print("ABQ refusal:", pct(agg["ABQ_refusal"]))
    print("UAQ refusal:", pct(agg["UAQ_refusal"]))
    print("gap:", pct(agg["gap"]))
    print("d':", "NA" if agg["dprime"] is None else f"{agg['dprime']:.3f}")
    print("ABQ EM_hit:", pct(agg["ABQ_EM_hit"]))

if __name__ == "__main__":
    main()
