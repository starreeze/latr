from argparse import ArgumentParser

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def auto_convert_numeric(df):
    df = df.copy()  # Avoid modifying original
    for col in df.columns:
        # Try to convert to numeric, errors='ignore' leaves non-numeric as object
        numeric_series = pd.to_numeric(df[col], errors="ignore")  # type: ignore

        # If conversion succeeded (i.e., not object dtype anymore), proceed
        if not pd.api.types.is_object_dtype(numeric_series):
            # Check if all non-NaN values are whole numbers → try int
            if numeric_series.dropna().apply(lambda x: isinstance(x, int) or x.is_integer()).all():
                try:
                    # Try converting to nullable integer (supports NaN)
                    df[col] = numeric_series.astype("Int64")
                except Exception:
                    # Fallback to original numeric (likely float due to NaN)
                    df[col] = numeric_series
            else:
                # Keep as float (default from pd.to_numeric)
                df[col] = numeric_series
        # else: leave as original (non-numeric, object dtype)
    return df


parser = ArgumentParser()
parser.add_argument("--path", required=True)
parser.add_argument("--interval", type=int, default=0)
parser.add_argument("--min_y", type=float, default=0)
parser.add_argument("--max_y", type=float, default=0)
parser.add_argument("--y_tick", type=float, nargs="+", default=[])
parser.add_argument("--x_tick", type=float, nargs="+", default=[])
parser.add_argument("--set_label", default="true")
parser.add_argument("--legend", default="lower_right")
args = parser.parse_args()

mpl.rcParams.update({"font.size": 20})

# Load data
df = pd.read_csv(args.path)

# Strip whitespace from column names and values
df.columns = df.columns.str.strip()
main_key = df.columns[0]
df = auto_convert_numeric(df)

if args.interval > 0:
    mask = df[main_key] % args.interval == 0
    # mask.iloc[-1] = True
    df = df[mask]

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(6, 6))

colors = ["#5b9ad5", "#ed7d31"]
markers = ["o", "o"]

# Plot each line with markers
for i, col in enumerate(df.columns[1:]):  # Skip main column
    ax.plot(
        df[main_key],
        df[col] * 100,
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
if args.set_label == "true":
    ax.set_xlabel(main_key)
    ax.set_ylabel("Average Validation Score")

min_y = args.min_y if args.min_y else None
max_y = args.max_y if args.max_y else None
ax.set_ylim([min_y, max_y])

if args.y_tick:
    ax.set_yticks(args.y_tick)

# Grid and background
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
ax.set_facecolor("#FAFAFA")

# Legend
ax.legend(loc=args.legend.replace("_", " "), frameon=True, fancybox=True, shadow=True)

# Spine styling
for spine in ax.spines.values():
    spine.set_visible(False)

# Improve layout
plt.tight_layout()

# Show plot
plt.show()
