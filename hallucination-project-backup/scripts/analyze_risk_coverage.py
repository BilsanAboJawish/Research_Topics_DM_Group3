import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


FILES = {
    "Base": "results/risk_coverage/base.csv",
    "IFD15": "results/risk_coverage/ifd15.csv",
    "Random15": "results/risk_coverage/random15.csv",
}

OUT = Path("results/risk_coverage")
OUT.mkdir(parents=True, exist_ok=True)


def roc_auc(labels, scores):
    """
    AUROC for distinguishing UAQ from ABQ.

    label:
        1 = UAQ
        0 = ABQ

    score:
        log P(None | prompt)

    Higher score should indicate stronger belief that
    the question is unanswerable.
    """

    labels = np.asarray(labels)
    scores = np.asarray(scores)

    # Average ranks correctly handle tied scores.
    ranks = pd.Series(scores).rank(
        method="average"
    ).to_numpy()

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    rank_sum_pos = np.sum(
        ranks[labels == 1]
    )

    auc = (
        rank_sum_pos
        - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)

    return auc


summary = []
curves = {}


for name, path in FILES.items():

    df = pd.read_csv(path)

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("N:", len(df))
    print(
        "ABQ:",
        (df["qtype"] == "ab").sum()
    )
    print(
        "UAQ:",
        (df["qtype"] == "ua").sum()
    )

    # True answerability label
    df["label"] = (
        df["qtype"] == "ua"
    ).astype(int)

    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    auc = roc_auc(
        df["label"],
        df["logp_none"]
    )

    # --------------------------------------------------------
    # RISK-COVERAGE
    #
    # Answer questions with lowest refusal confidence first.
    #
    # Risk = fraction of answered questions that are actually
    # UAQ.
    # --------------------------------------------------------

    ranked = df.sort_values(
        "logp_none",
        ascending=True
    ).reset_index(drop=True)

    n = len(ranked)

    k = np.arange(
        1,
        n + 1
    )

    coverage = k / n

    cumulative_uaq = (
        ranked["label"]
        .cumsum()
        .to_numpy()
    )

    risk = cumulative_uaq / k

    # NumPy compatibility
    if hasattr(np, "trapezoid"):
        aurc = np.trapezoid(
            risk,
            coverage
        )
    else:
        aurc = np.trapz(
            risk,
            coverage
        )

    curves[name] = (
        coverage,
        risk
    )

    # Risk at representative coverage points
    risk_points = {}

    for c in [
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        1.00,
    ]:

        idx = min(
            int(np.ceil(c * n)) - 1,
            n - 1
        )

        risk_points[c] = risk[idx]

    summary.append({
        "model": name,
        "AUROC_UAQ_vs_ABQ": auc,
        "AURC_answerability": aurc,
        "risk_at_10pct": risk_points[0.10],
        "risk_at_25pct": risk_points[0.25],
        "risk_at_50pct": risk_points[0.50],
        "risk_at_75pct": risk_points[0.75],
        "risk_at_90pct": risk_points[0.90],
        "risk_at_100pct": risk_points[1.00],
    })

    print(f"AUROC: {auc:.4f}")
    print(f"AURC:  {aurc:.4f}")

    print("\nRisk by coverage:")

    for c, r in risk_points.items():
        print(
            f"  Coverage {100*c:>5.0f}%"
            f" -> risk {100*r:6.2f}%"
        )

    # Save curve data
    pd.DataFrame({
        "coverage": coverage,
        "risk": risk
    }).to_csv(
        OUT / f"{name.lower()}_risk_coverage.csv",
        index=False
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame(
    summary
)

print()
print("=" * 100)
print("ANSWERABILITY CONFIDENCE SUMMARY")
print("=" * 100)

display = summary_df.copy()

for col in display.columns:
    if col != "model":
        display[col] = display[col].map(
            lambda x: f"{x:.4f}"
        )

print(
    display.to_string(
        index=False
    )
)

summary_df.to_csv(
    OUT / "risk_coverage_summary.csv",
    index=False
)


# ============================================================
# COMBINED FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

for name, (
    coverage,
    risk
) in curves.items():

    ax.plot(
        100 * coverage,
        100 * risk,
        label=name
    )

ax.set_xlabel(
    "Coverage (%)"
)

ax.set_ylabel(
    "Answerability Risk (%)"
)

ax.set_title(
    "Risk–Coverage Using Refusal Confidence"
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    OUT / "risk_coverage_combined.png",
    dpi=300
)

fig.savefig(
    OUT / "risk_coverage_combined.pdf"
)

plt.close(fig)


print()
print("Saved:")
print(
    OUT / "risk_coverage_summary.csv"
)
print(
    OUT / "risk_coverage_combined.png"
)
print(
    OUT / "risk_coverage_combined.pdf"
)
