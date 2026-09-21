from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
AUDIT = ROOT / "feature_extraction" / "src" / "ecg_cascade" / "web_demo" / "audit_results.json"

NAVY = "#193946"
TEAL = "#0A7C74"
ORANGE = "#D67A35"
RED = "#B94A48"
GRAY = "#70858D"
GRID = "#DCE7E9"


with AUDIT.open("r", encoding="utf-8") as stream:
    audit = json.load(stream)

selected = {row["target_id"]: row for row in audit["selected"]}
missing = {
    row["target_id"]: row
    for row in audit["baselines"]
    if row["baseline"] == "feature missingness only"
}


def style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#AABDC2")
    ax.tick_params(colors=NAVY, labelsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)


# Figure 1: simple patient-weighted result versus shortcut control.
targets = ["artifact", "seizure", "abnormal_beat", "noise_active", "signal_unusable"]
labels = ["Artifact", "Seizure", "Abnormal beat", "Active noise", "Strict unusable"]
real = np.array([selected[t]["group_macro_balanced_accuracy"] * 100 for t in targets])
shortcut = np.array([missing[t]["group_macro_balanced_accuracy"] * 100 for t in targets])
groups = [selected[t]["groups"] for t in targets]

fig, ax = plt.subplots(figsize=(10.8, 4.0), constrained_layout=True)
y = np.arange(len(targets))
h = 0.32
ax.barh(y - h / 2, real, height=h, color=NAVY, label="Uses the numerical ECG measurements")
ax.barh(y + h / 2, shortcut, height=h, color=ORANGE, label="Knows only whether each measurement exists")
ax.axvline(50, color=GRAY, linestyle="--", linewidth=1.2)
for i, (value, control, group_count) in enumerate(zip(real, shortcut, groups)):
    ax.text(value + 0.7, i - h / 2, f"{value:.1f}%", va="center", fontsize=10, color=NAVY, weight="bold")
    ax.text(control + 0.7, i + h / 2, f"{control:.1f}%", va="center", fontsize=9, color=ORANGE)
    ax.text(101.0, i, f"{group_count} groups", va="center", ha="right", fontsize=9, color=GRAY)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(45, 102)
ax.set_xlabel("Fair score: 50% = chance, 100% = perfect", color=NAVY, fontsize=11)
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=10)
style(ax)
fig.savefig(OUT / "simple_overview.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# Figure 2: seizure score and the two clinically important denominators.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 3.7), gridspec_kw={"width_ratios": [0.9, 1.35]}, constrained_layout=True)

seizure_scores = [50.0, missing["seizure"]["group_macro_balanced_accuracy"] * 100, selected["seizure"]["group_macro_balanced_accuracy"] * 100]
seizure_labels = ["Chance", "Only knows if measurement exists", "Uses the measurement values"]
colors = ["#BFCBCF", ORANGE, NAVY]
bars = ax1.barh(np.arange(3), seizure_scores, color=colors, height=0.55)
for bar, score in zip(bars, seizure_scores):
    ax1.text(score + 0.7, bar.get_y() + bar.get_height() / 2, f"{score:.1f}%", va="center", fontsize=10, weight="bold", color=NAVY)
ax1.set_yticks(np.arange(3), seizure_labels)
ax1.invert_yaxis()
ax1.set_xlim(45, 73)
ax1.set_xlabel("Fair score: 50% = chance", color=NAVY, fontsize=10)
style(ax1)

rows = ["True seizure rows", "Positive alerts"]
good = np.array([728, 728], dtype=float)
bad = np.array([428, 2960], dtype=float)
totals = good + bad
good_pct = good / totals * 100
bad_pct = bad / totals * 100
y2 = np.arange(2)
ax2.barh(y2, good_pct, color=TEAL, height=0.54, label="Detected / true alert")
ax2.barh(y2, bad_pct, left=good_pct, color=RED, height=0.54, label="Missed / false alert")
ax2.text(good_pct[0] / 2, 0, "728 detected", ha="center", va="center", color="white", fontsize=10, weight="bold")
ax2.text(good_pct[0] + bad_pct[0] / 2, 0, "428 missed", ha="center", va="center", color="white", fontsize=10, weight="bold")
ax2.text(good_pct[1] / 2, 1, "728 true", ha="center", va="center", color="white", fontsize=10, weight="bold")
ax2.text(good_pct[1] + bad_pct[1] / 2, 1, "2,960 false", ha="center", va="center", color="white", fontsize=10, weight="bold")
ax2.set_yticks(y2, rows)
ax2.invert_yaxis()
ax2.set_xlim(0, 100)
ax2.set_xlabel("Percentage of each group", color=NAVY, fontsize=10)
ax2.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=9)
style(ax2)
fig.savefig(OUT / "seizure_reality.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# Figure 3: the three jointly evaluable mixed-context drops.
names = ["Abnormal detector", "Noise detector", "Artifact detector"]
easy_context = ["Clean interval", "Normal beat", "Normal beat"]
hard_context = ["During noise", "Abnormal beat", "Abnormal beat"]
easy = np.array([87.5, 76.6, 81.0])
hard = np.array([74.1, 60.0, 64.7])

fig, ax = plt.subplots(figsize=(10.8, 3.55), constrained_layout=True)
y = np.arange(3)
for i in range(3):
    ax.plot([hard[i], easy[i]], [i, i], color="#AFC3C8", linewidth=5, solid_capstyle="round")
    ax.scatter(easy[i], i, s=115, color=NAVY, zorder=3)
    ax.scatter(hard[i], i, s=115, color=RED, zorder=3)
    ax.text(easy[i] + 0.8, i - 0.09, f"{easy[i]:.1f}%", va="center", fontsize=10, color=NAVY, weight="bold")
    ax.text(hard[i] - 0.8, i - 0.09, f"{hard[i]:.1f}%", va="center", ha="right", fontsize=10, color=RED, weight="bold")
    drop_y = i + 0.24 if i < 2 else i - 0.24
    ax.text((easy[i] + hard[i]) / 2, drop_y, f"drop {easy[i] - hard[i]:.1f} points", ha="center", fontsize=9, color=GRAY)
ax.set_yticks(y, names)
ax.invert_yaxis()
ax.set_xlim(53, 93)
ax.set_xlabel("Sensitivity", color=NAVY, fontsize=11)
ax.scatter([], [], s=95, color=NAVY, label="Cleaner/easier context")
ax.scatter([], [], s=95, color=RED, label="Noisy/abnormal context")
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=10)
style(ax)
fig.savefig(OUT / "mixed_context_simple.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("Created simple_overview.png, seizure_reality.png, mixed_context_simple.png")
