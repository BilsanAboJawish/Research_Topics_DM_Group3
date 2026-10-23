import matplotlib.pyplot as plt

conditions = {
    "BASE": (24.04, 52.84),

    "IFD 5%":  (24.65, 50.41),
    "IFD 10%": (39.16, 67.93),
    "IFD 15%": (41.02, 72.88),
    "IFD 20%": (40.64, 70.52),
    "IFD 25%": (46.27, 76.01),

    "Random 5%":  (49.61, 80.42),
    "Random 10%": (49.49, 79.31),
    "Random 15%": (55.65, 84.68),
    "Random 20%": (59.48, 87.72),
    "Random 25%": (49.99, 77.80),
}

fig, ax = plt.subplots(figsize=(8, 6))

# Base model
x, y = conditions["BASE"]
ax.scatter(x, y, s=100, marker="*")
ax.annotate(
    "Base",
    (x, y),
    xytext=(6, 6),
    textcoords="offset points"
)

# IFD
for label in ["IFD 5%", "IFD 10%", "IFD 15%", "IFD 20%", "IFD 25%"]:
    x, y = conditions[label]
    ax.scatter(x, y, s=55)
    ax.annotate(
        label,
        (x, y),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

# Random
for label in [
    "Random 5%",
    "Random 10%",
    "Random 15%",
    "Random 20%",
    "Random 25%"
]:
    x, y = conditions[label]
    ax.scatter(x, y, s=55, marker="s")
    ax.annotate(
        label,
        (x, y),
        xytext=(5, -10),
        textcoords="offset points",
        fontsize=8
    )

# y = x reference: no discrimination
lo = 20
hi = 92
ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)

ax.set_xlabel("ABQ refusal rate (%)")
ax.set_ylabel("UAQ refusal rate (%)")
ax.set_title("Refusal Operating Points Across Fine-Tuning Conditions")

ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    "results/figures/refusal_operating_points.png",
    dpi=300
)

fig.savefig(
    "results/figures/refusal_operating_points.pdf"
)

print("Saved:")
print("results/figures/refusal_operating_points.png")
print("results/figures/refusal_operating_points.pdf")
