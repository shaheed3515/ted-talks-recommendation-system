"""
Data Quality Dimensions Visualization Script
Generates high-resolution charts for:
1. Completeness
2. Validity
3. Accuracy
4. Consistency
"""

import os
import ast
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

output_dir = "docs/figures"
os.makedirs(output_dir, exist_ok=True)

print("Loading data for Data Quality Assessment...")
ted_df = pd.read_csv("data/raw/ted_main.csv")
transcripts_df = pd.read_csv("data/raw/transcripts.csv")
merged_df = pd.merge(ted_df, transcripts_df, on="url", how="left")

# -------------------------------------------------------------
# 1. COMPLETENESS: Percentage of Present vs Missing Data
# -------------------------------------------------------------
print("Generating Chart 1: Completeness...")
completeness = (1 - merged_df.isnull().mean()) * 100
completeness = completeness.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
colors = ["#e63946" if x < 99 else "#2a9d8f" for x in completeness.values]
bars = ax.barh(completeness.index, completeness.values, color=colors, edgecolor="black", height=0.65)

ax.set_xlim(85, 101)
ax.set_title("Data Quality Dimension 1: Completeness\n(% of Non-Null Values per Column)", fontsize=14, weight="bold", pad=15)
ax.set_xlabel("Completeness (%)", fontsize=12, labelpad=10)
ax.axvline(100, color="gray", linestyle="--", alpha=0.7)

for bar in bars:
    w = bar.get_width()
    ax.text(w - 2.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", ha="right", color="white", weight="bold", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "01_completeness_chart.png"))
plt.close(fig)

# -------------------------------------------------------------
# 2. VALIDITY: Duration Distribution & Format Conformance
# -------------------------------------------------------------
print("Generating Chart 2: Validity...")
merged_df["duration_mins"] = merged_df["duration"] / 60.0
valid_durations = merged_df["duration_mins"][(merged_df["duration_mins"] > 0) & (merged_df["duration_mins"] < 60)]

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
sns.histplot(valid_durations, bins=35, kde=True, color="#457b9d", edgecolor="black", ax=ax)
ax.axvline(valid_durations.mean(), color="#e63946", linestyle="--", linewidth=2, label=f"Mean: {valid_durations.mean():.1f} min")
ax.axvline(valid_durations.median(), color="#f4a261", linestyle="-", linewidth=2, label=f"Median: {valid_durations.median():.1f} min")

ax.set_title("Data Quality Dimension 2: Validity\n(Talk Duration Distribution within Valid 0–60 min range)", fontsize=14, weight="bold", pad=15)
ax.set_xlabel("Duration (Minutes)", fontsize=12)
ax.set_ylabel("Number of Talks", fontsize=12)
ax.legend(frameon=True, facecolor="white", loc="upper right")

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "02_validity_durations.png"))
plt.close(fig)

# -------------------------------------------------------------
# 3. ACCURACY: Word Count vs Duration (Speaking Rate Plausibility)
# -------------------------------------------------------------
print("Generating Chart 3: Accuracy...")
def get_clean_word_count(text):
    if not isinstance(text, str) or pd.isna(text):
        return 0
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return len(clean.split())

merged_df["word_count"] = merged_df["transcript"].apply(get_clean_word_count)
speech_sample = merged_df[(merged_df["word_count"] > 100) & (merged_df["duration_mins"] > 2) & (merged_df["duration_mins"] < 35)].copy()
speech_sample["words_per_minute"] = speech_sample["word_count"] / speech_sample["duration_mins"]

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
sns.regplot(
    data=speech_sample,
    x="duration_mins",
    y="word_count",
    scatter_kws={"alpha": 0.45, "color": "#1d3557", "s": 25},
    line_kws={"color": "#e63946", "linewidth": 2.5, "label": f"Linear Fit (Avg {speech_sample['words_per_minute'].mean():.0f} WPM)"},
    ax=ax
)

ax.set_title("Data Quality Dimension 3: Accuracy\n(Speech Rate Verification: Spoken Words vs Talk Duration)", fontsize=14, weight="bold", pad=15)
ax.set_xlabel("Duration (Minutes)", fontsize=12)
ax.set_ylabel("Spoken Words in Transcript", fontsize=12)
ax.legend(frameon=True, facecolor="white", loc="upper left")

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "03_accuracy_speech_rate.png"))
plt.close(fig)

# -------------------------------------------------------------
# 4. CONSISTENCY: Tag Categorization & Cross-table Integrity
# -------------------------------------------------------------
print("Generating Chart 4: Consistency...")
def parse_tags_list(tags_str):
    if not isinstance(tags_str, str) or pd.isna(tags_str):
        return []
    try:
        parsed = ast.literal_eval(tags_str)
        return [str(t).strip().lower() for t in parsed] if isinstance(parsed, list) else []
    except:
        return []

all_tags = [t for tags in merged_df["tags"].apply(parse_tags_list) for t in tags]
tag_series = pd.Series(all_tags).value_counts().head(12)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
sns.barplot(x=tag_series.values, y=tag_series.index, palette="mako", edgecolor="black", ax=ax)
ax.set_title("Data Quality Dimension 4: Consistency\n(Standardized Topic Categories After Syntax Parsing)", fontsize=14, weight="bold", pad=15)
ax.set_xlabel("Number of Talks", fontsize=12)
ax.set_ylabel("Standardized Category Tag", fontsize=12)

for i, v in enumerate(tag_series.values):
    ax.text(v + 10, i, str(v), va="center", weight="bold", color="#1d3557")

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "04_consistency_tag_distribution.png"))
plt.close(fig)

print("All 4 Data Quality Charts successfully saved in docs/figures/!")
