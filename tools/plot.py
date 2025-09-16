import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams.update({"font.size": 20})
interval = 40

# Load data
df = pd.read_csv("dataset/math.csv")

# Strip whitespace from column names and values
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df = df.astype(float)  # Ensure numeric types
df = df[df["Step"] % interval == 0]

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(8, 6))

colors = ["#ed7d31", "#5b9ad5", "#ed7d31", "#5b9ad5"]
markers = ["o", "o", "s", "s"]

# Plot each line with markers
for i, col in enumerate(df.columns[1:]):  # Skip 'Step' column
    ax.plot(
        df["Step"],
        df[col],
        label=col,
        color=colors[i],
        marker=markers[i],
        markersize=12,
        linewidth=4,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )

# --- STYLING ---
# ax.set_title("Algorithm Performance Comparison", fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel("Step")
ax.set_ylabel("Average Validation Reward")

# Grid and background
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
ax.set_facecolor("#FAFAFA")

# Legend
ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)

# Spine styling
for spine in ax.spines.values():
    spine.set_visible(False)

# Improve layout
plt.tight_layout()

# Show plot
plt.show()
