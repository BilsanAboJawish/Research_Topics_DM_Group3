import matplotlib.pyplot as plt


results = {
    "Base":       (24.04, 69.09),

    "IFD 5%":     (24.65, 71.99),
    "IFD 10%":    (39.16, 78.35),
    "IFD 15%":    (41.02, 77.74),
    "IFD 20%":    (40.64, 76.77),
    "IFD 25%":    (46.27, 78.20),

    "Random 5%":  (49.61, 79.91),
    "Random 10%": (49.49, 81.04),
    "Random 15%": (55.65, 80.41),
    "Random 20%": (59.48, 85.51),
    "Random 25%": (49.99, 81.08),
}


fig, ax = plt.subplots(figsize=(8, 6))

for name, (refusal, answered_accuracy) in results.items():

    coverage = 100 - refusal
    risk = 100 - answered_accuracy

    if name == "Base":
        marker = "*"
        size = 120
    elif name.startswith("Random"):
        marker = "s"
        size = 55
    else:
        marker = "o"
        size = 55

    ax.scatter(
        coverage,
        risk,
        marker=marker,
        s=size
    )

    ax.annotate(
        name,
        (coverage, risk),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )


ax.set_xlabel("ABQ Coverage (%)")
ax.set_ylabel("Risk among Answered ABQs (%)")

ax.set_title(
    "Coverage–Risk Operating Points"
)

ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    "results/figures/coverage_risk_operating_points.png",
    dpi=300
)

fig.savefig(
    "results/figures/coverage_risk_operating_points.pdf"
)

print("Saved coverage-risk figures.")
