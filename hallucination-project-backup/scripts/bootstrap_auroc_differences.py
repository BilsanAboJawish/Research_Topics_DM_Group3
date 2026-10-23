import numpy as np
import pandas as pd
from pathlib import Path


FILES = {
    "Base": "results/risk_coverage/base.csv",
    "IFD15": "results/risk_coverage/ifd15.csv",
    "Random15": "results/risk_coverage/random15.csv",
}

N_BOOT = 5000
SEED = 42

OUT_DIR = Path("results/risk_coverage")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUROC
# ============================================================

def auc_pos_neg(pos_scores, neg_scores):
    """
    AUROC = P(score_positive > score_negative)
            + 0.5 * P(tie)

    Positive class = UAQ.
    Higher log P(None) means stronger unanswerability confidence.
    """

    neg_sorted = np.sort(neg_scores)

    left = np.searchsorted(
        neg_sorted,
        pos_scores,
        side="left"
    )

    right = np.searchsorted(
        neg_sorted,
        pos_scores,
        side="right"
    )

    wins = left
    ties = right - left

    return np.mean(
        (wins + 0.5 * ties)
        / len(neg_scores)
    )


# ============================================================
# LOAD + ALIGN EXACT SAME EXAMPLES
# ============================================================

dfs = {}

for name, path in FILES.items():
    df = pd.read_csv(path)

    key_cols = [
        "task",
        "qtype",
        "doc_id",
    ]

    if df.duplicated(key_cols).any():
        raise RuntimeError(
            f"{name}: duplicate example keys found"
        )

    dfs[name] = df[
        key_cols + ["logp_none"]
    ].rename(
        columns={
            "logp_none": f"logp_{name}"
        }
    )


merged = dfs["Base"]

for name in ["IFD15", "Random15"]:
    merged = merged.merge(
        dfs[name],
        on=["task", "qtype", "doc_id"],
        how="inner",
        validate="one_to_one"
    )


print("=" * 80)
print("PAIRED DATA VALIDATION")
print("=" * 80)

print("Merged N:", len(merged))
print(
    "ABQ:",
    (merged["qtype"] == "ab").sum()
)
print(
    "UAQ:",
    (merged["qtype"] == "ua").sum()
)

if len(merged) != 13970:
    raise RuntimeError(
        "Expected 13,970 paired examples."
    )

ab = merged[
    merged["qtype"] == "ab"
].reset_index(drop=True)

ua = merged[
    merged["qtype"] == "ua"
].reset_index(drop=True)

if len(ab) != 6985 or len(ua) != 6985:
    raise RuntimeError(
        "Unexpected ABQ/UAQ counts."
    )


# ============================================================
# ARRAYS
# ============================================================

models = [
    "Base",
    "IFD15",
    "Random15"
]

ab_scores = {
    m: ab[f"logp_{m}"].to_numpy()
    for m in models
}

ua_scores = {
    m: ua[f"logp_{m}"].to_numpy()
    for m in models
}


# ============================================================
# POINT ESTIMATES
# ============================================================

point_auc = {}

for m in models:
    point_auc[m] = auc_pos_neg(
        ua_scores[m],
        ab_scores[m]
    )


print()
print("=" * 80)
print("POINT AUROC")
print("=" * 80)

for m in models:
    print(
        f"{m:<10}: {point_auc[m]:.6f}"
    )


# ============================================================
# PAIRED STRATIFIED BOOTSTRAP
#
# Resample UAQs and ABQs separately.
# The SAME sampled item indices are used for all three models.
# ============================================================

rng = np.random.default_rng(SEED)

boot = {
    m: np.empty(N_BOOT)
    for m in models
}

n_ab = len(ab)
n_ua = len(ua)

print()
print(
    f"Running {N_BOOT:,} paired bootstrap replicates..."
)

for b in range(N_BOOT):

    ab_idx = rng.integers(
        0,
        n_ab,
        size=n_ab
    )

    ua_idx = rng.integers(
        0,
        n_ua,
        size=n_ua
    )

    for m in models:
        boot[m][b] = auc_pos_neg(
            ua_scores[m][ua_idx],
            ab_scores[m][ab_idx]
        )

    if (b + 1) % 500 == 0:
        print(
            f"  {b + 1:,}/{N_BOOT:,}"
        )


# ============================================================
# MODEL-SPECIFIC CIs
# ============================================================

def ci95(x):
    return np.quantile(
        x,
        [0.025, 0.975]
    )


print()
print("=" * 80)
print("BOOTSTRAP AUROC 95% CI")
print("=" * 80)

individual_rows = []

for m in models:

    lo, hi = ci95(boot[m])

    individual_rows.append({
        "model": m,
        "point_auc": point_auc[m],
        "bootstrap_mean":
            np.mean(boot[m]),
        "ci95_low": lo,
        "ci95_high": hi,
    })

    print(
        f"{m:<10} "
        f"AUROC={point_auc[m]:.4f} "
        f"95% CI [{lo:.4f}, {hi:.4f}]"
    )


# ============================================================
# PAIRED DIFFERENCES
# ============================================================

comparisons = [
    ("IFD15", "Base"),
    ("Random15", "Base"),
    ("IFD15", "Random15"),
]

difference_rows = []


print()
print("=" * 80)
print("PAIRED AUROC DIFFERENCES")
print("=" * 80)

for a, b in comparisons:

    delta_point = (
        point_auc[a]
        - point_auc[b]
    )

    delta_boot = (
        boot[a]
        - boot[b]
    )

    lo, hi = ci95(delta_boot)

    # Two-sided bootstrap sign probability.
    p_lower = (
        np.sum(delta_boot <= 0) + 1
    ) / (N_BOOT + 1)

    p_upper = (
        np.sum(delta_boot >= 0) + 1
    ) / (N_BOOT + 1)

    p_two = min(
        1.0,
        2 * min(p_lower, p_upper)
    )

    excludes_zero = (
        lo > 0 or hi < 0
    )

    difference_rows.append({
        "comparison": f"{a} - {b}",
        "delta_auc": delta_point,
        "bootstrap_mean_delta":
            np.mean(delta_boot),
        "ci95_low": lo,
        "ci95_high": hi,
        "ci_excludes_zero":
            excludes_zero,
        "bootstrap_two_sided_p":
            p_two,
    })

    print(
        f"{a} - {b}"
    )

    print(
        f"  ΔAUROC = {delta_point:+.4f}"
    )

    print(
        f"  95% CI = "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )

    print(
        f"  CI excludes 0: "
        f"{excludes_zero}"
    )

    print(
        f"  bootstrap p ≈ {p_two:.4f}"
    )

    print()


# ============================================================
# SAVE
# ============================================================

pd.DataFrame(
    individual_rows
).to_csv(
    OUT_DIR /
    "auroc_bootstrap_individual.csv",
    index=False
)

pd.DataFrame(
    difference_rows
).to_csv(
    OUT_DIR /
    "auroc_bootstrap_differences.csv",
    index=False
)

boot_df = pd.DataFrame({
    "Base": boot["Base"],
    "IFD15": boot["IFD15"],
    "Random15": boot["Random15"],

    "IFD15_minus_Base":
        boot["IFD15"] - boot["Base"],

    "Random15_minus_Base":
        boot["Random15"] - boot["Base"],

    "IFD15_minus_Random15":
        boot["IFD15"] - boot["Random15"],
})

boot_df.to_csv(
    OUT_DIR /
    "auroc_bootstrap_distributions.csv",
    index=False
)


print("=" * 80)
print("SAVED")
print("=" * 80)

print(
    OUT_DIR /
    "auroc_bootstrap_individual.csv"
)

print(
    OUT_DIR /
    "auroc_bootstrap_differences.csv"
)

print(
    OUT_DIR /
    "auroc_bootstrap_distributions.csv"
)
