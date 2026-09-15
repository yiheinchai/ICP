#!/usr/bin/env python3
"""Render usefulness masks as highlighted context text (heatmaps over words)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "artifacts"

SKILL = [
    "[Access] New hires request laptop access via the IT portal using form HR-221.",
    "[Access] VPN enrollment requires a manager-approved ticket before day 3.",
    "[Expenses] Meal expenses under $75 do not need itemized receipts.",
    "[Expenses] International flights require VP approval at least 14 days ahead.",
    "[Incidents] Sev-1 outages must page the on-call within 5 minutes.",
    "[Incidents] Customer data incidents escalate to Legal via alias #privacy-page.",
    "[Legacy] The 2019 holiday party was held in Building C cafeteria.",
    "[Legacy] Fax machine #4 in Wing B was decommissioned in March 2021.",
    "[Trivia] The company espresso machine's model number is GX-9000.",
    "[Trivia] Parking lot B has 412 spaces painted teal.",
]


def draw_heatmap(ax, chunks, scores, title, vmax=None):
    scores = list(map(float, scores))
    vmax = max(scores) if vmax is None or vmax <= 0 else vmax
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    y = 0.95
    for i, (c, s) in enumerate(zip(chunks, scores)):
        norm = s / vmax if vmax > 0 else 0.0
        color = cm.Reds(0.08 + 0.85 * norm)
        label = f"[{i+1}] score={s:.2f}"
        ax.text(0.01, y, label, fontsize=7, color="#555555",
                transform=ax.transAxes, va="top", family="monospace")
        y -= 0.035
        for line in textwrap.wrap(c, width=72):
            ax.text(0.01, y, line, fontsize=8.5, family="monospace",
                    transform=ax.transAxes, va="top",
                    bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.25"))
            y -= 0.055
        y -= 0.02


def main():
    # 1) per-task post-hoc masks (from time-EMA run's step 0 of each type, or qonly file)
    qonly = json.load(open(OUT / "mask_prediction_qonly.json"))
    post = {r["task"]: r["posthoc"] for r in qonly["rows"]}
    llm = {r["task"]: r["llm_qonly"] for r in qonly["rows"]}
    tfidf = {r["task"]: r["tfidf"] for r in qonly["rows"]}
    names = {"task1_vpn": "VPN enrollment", "task2_meals": "Meal expenses",
             "task3_sev1": "Sev-1 paging", "task4_privacy": "Privacy escalation"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (tid, title) in zip(axes.flat, names.items()):
        draw_heatmap(ax, SKILL, post[tid], f"Post-hoc mask — {title}")
    fig.suptitle("Skill text colored by post-hoc usefulness (ablation) — brighter red = more useful", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "textmap_posthoc_tasks.png", dpi=140)
    plt.close(fig)

    # 2) question-only predicted vs true, side by side for sev1 (most dramatic)
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    draw_heatmap(axes[0], SKILL, post["task3_sev1"], "Sev-1 TRUE mask (post-hoc)")
    draw_heatmap(axes[1], SKILL, llm["task3_sev1"], "Sev-1 LLM question-only (flat → fails)")
    draw_heatmap(axes[2], SKILL, tfidf["task3_sev1"], "Sev-1 TF-IDF question-only (finds key)")
    fig.suptitle("Mask prediction from question alone — Sev-1 task", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "textmap_qonly_sev1.png", dpi=140)
    plt.close(fig)

    # 3) EMA evolution over time (session stream)
    ema = json.load(open(OUT / "time_ema_pruning.json"))
    steps = ema["steps"]
    stream = ema["stream"]
    picks = [2, 6, 10, 15]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, t in zip(axes.flat, picks):
        # reconstruct EMA_{t-1} approx from history
        hist = ema["ema_history"]["0.9"]
        ema_vec = hist[t - 1] if t > 0 else [0.1] * len(SKILL)
        draw_heatmap(ax, SKILL, ema_vec, f"t={t} ({stream[t]}): time-EMA routing mask (λ=0.9)")
    fig.suptitle("What the time-only EMA router 'sees' before each step — no question used", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "textmap_ema_time.png", dpi=140)
    plt.close(fig)

    # 4) hard vs soft: poison-before-live item 1 (zeldria) — live vs poison focus
    rig = json.load(open(OUT / "hard_vs_soft_rigorous.json"))
    tr = [x for x in rig["trials"] if x["item_id"] == "zeldria_capital" and x["order"] == "poison_before_live"]
    lp = {x["condition"]: x["gold_logprob"] for x in tr}
    print("zeldria poison-before-live gold logprobs:", {k: round(v, 3) for k, v in lp.items()})

    print("Wrote textmap_posthoc_tasks.png, textmap_qonly_sev1.png, textmap_ema_time.png")


if __name__ == "__main__":
    main()
