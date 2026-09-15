#!/usr/bin/env python3
"""
Skill-bloat pruning — intended task setup
=========================================

Shared skill S (company docs / procedural knowledge) stays in context.
Run tasks 1..n. After each task, generate a usefulness mask over skill chunks.
After n tasks, find chunks that were least useful across all tasks (dead knowledge).
Mask those away before task n+1.

Timing:
  load skill S
  for t in 1..n:
      run task_t on full skill
      AFTER complete → mask_t = usefulness(S | task_t, answer_t)   # generation
  dead = chunks with low max_t(mask_t)                              # aggregation
  run task_{n+1} on S \\ dead                                        # application
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts"
FIG_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

SKILL_CHUNKS = [
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

TASKS = [
    {
        "id": "task1_vpn",
        "question": "What must a new hire complete before day 3 for VPN enrollment?",
        "gold": "manager-approved ticket",
        "true_useful": {1},
    },
    {
        "id": "task2_meals",
        "question": "Do meal expenses under $75 need itemized receipts?",
        "gold": "no",
        "true_useful": {2},
    },
    {
        "id": "task3_sev1",
        "question": "Within how many minutes must a Sev-1 outage page the on-call?",
        "gold": "5",
        "true_useful": {4},
    },
]

TASK_N1 = {
    "id": "task4_privacy",
    "question": "Where should customer data incidents be escalated?",
    "gold": "Legal",
    "true_useful": {5},
}


def chat(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_skill(chunks: list[str], keep: list[int] | None = None) -> str:
    idxs = list(range(len(chunks))) if keep is None else list(keep)
    return "\n".join(f"[{i+1}] {chunks[i]}" for i in idxs)


def task_prompt(tokenizer, chunks, question, keep=None) -> str:
    user = (
        "You are following company skill docs. Answer using only the skill context. "
        "Reply with a short phrase only.\n\n"
        f"Skill document:\n{format_skill(chunks, keep)}\n\n"
        f"Task: {question}"
    )
    return chat(tokenizer, user)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 32) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def mean_logprob(model, tokenizer, prompt: str, answer: str) -> float:
    p = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        DEVICE
    )
    a = tokenizer(
        " " + answer, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(DEVICE)
    full = torch.cat([p, a], dim=1)
    with torch.no_grad():
        logits = model(full).logits
    vals = []
    for i in range(a.shape[1]):
        lp = torch.log_softmax(logits[0, p.shape[1] + i - 1], dim=-1)
        vals.append(lp[a[0, i]].item())
    return float(np.mean(vals))


def llm_mask(model, tokenizer, chunks, question, answer):
    n = len(chunks)
    skill = format_skill(chunks)
    pick_user = (
        f"Skill document:\n{skill}\n\n"
        f"Task: {question}\nAnswer produced: {answer}\n\n"
        f"Which ONE skill chunk number (1-{n}) most directly provided this answer? "
        "Reply with only the number."
    )
    pick_raw = generate(model, tokenizer, chat(tokenizer, pick_user), max_new_tokens=8)
    m = re.search(r"\b([1-9]|10)\b", pick_raw)
    pick = int(m.group(1)) if m else None
    if pick is not None and not (1 <= pick <= n):
        pick = None

    votes = np.zeros(n, dtype=np.float64)
    logs = []
    for i, c in enumerate(chunks):
        yn_user = (
            f"Skill chunk: [{i+1}] {c}\n"
            f"Task: {question}\nAnswer produced: {answer}\n\n"
            "Was this chunk useful for producing the answer? Reply yes or no."
        )
        yn = generate(model, tokenizer, chat(tokenizer, yn_user), max_new_tokens=4).lower()
        logs.append(f"{i+1}:{yn.strip()}")
        if re.search(r"\byes\b", yn) and not re.search(r"\bno\b", yn):
            votes[i] = 1.0
        elif re.search(r"\byes\b", yn):
            votes[i] = 0.5

    scores = votes.copy()
    if pick is not None:
        scores[pick - 1] += 2.0
    if scores.sum() == 0 and pick is not None:
        scores = np.zeros(n)
        scores[pick - 1] = 1.0
    if scores.sum() > 0:
        scores = scores / scores.sum()
    return scores, f"pick={pick_raw.strip()} | {'; '.join(logs)}"


def ablation_mask(model, tokenizer, chunks, question, gold) -> np.ndarray:
    base_prompt = task_prompt(tokenizer, chunks, question)
    base = mean_logprob(model, tokenizer, base_prompt, gold)
    drops = []
    for i in range(len(chunks)):
        red = list(chunks)
        red[i] = "[REDACTED]"
        lp = mean_logprob(model, tokenizer, task_prompt(tokenizer, red, question), gold)
        drops.append(base - lp)
    drops = np.maximum(np.asarray(drops, dtype=np.float64), 0.0)
    if drops.sum() > 0:
        drops = drops / drops.sum()
    return drops


def aggregate_dead(masks, mode="max_below", threshold=0.05, bottom_frac=0.4):
    stack = np.stack(masks, axis=0)
    max_across = stack.max(axis=0)
    mean_across = stack.mean(axis=0)
    if mode == "max_below":
        dead = [int(i) for i, v in enumerate(max_across) if v < threshold]
    else:
        k = max(1, int(round(len(max_across) * bottom_frac)))
        dead = [int(i) for i in np.argsort(max_across)[:k]]
    tops = {int(np.argmax(m)) for m in masks if float(m.sum()) > 0}
    dead = [i for i in dead if i not in tops]
    return dead, max_across, mean_across


def run_task(model, tokenizer, chunks, task, keep=None, label=""):
    keep = list(range(len(chunks))) if keep is None else list(keep)
    prompt = task_prompt(tokenizer, chunks, task["question"], keep)
    answer = generate(model, tokenizer, prompt)
    lp = mean_logprob(model, tokenizer, prompt, task["gold"])
    return {
        "label": label or task["id"],
        "task_id": task["id"],
        "keep_1based": [i + 1 for i in keep],
        "n_keep": len(keep),
        "answer": answer,
        "gold": task["gold"],
        "correct": task["gold"].lower() in answer.lower(),
        "gold_logprob": lp,
        "kept_true_useful": bool(set(keep) & set(task["true_useful"])),
    }


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    print("\n=== Phase A: tasks 1..n on FULL skill; generate mask AFTER each ===")
    llm_masks, abl_masks, phase_a = [], [], []
    for t in TASKS:
        result = run_task(model, tokenizer, SKILL_CHUNKS, t, label=f"{t['id']}_full")
        print(f"  {result['task_id']}: ans={result['answer']!r} ok={result['correct']}")

        # GENERATION TIME = after task completes
        llm, raw = llm_mask(
            model, tokenizer, SKILL_CHUNKS, t["question"], result["answer"]
        )
        abl = ablation_mask(
            model, tokenizer, SKILL_CHUNKS, t["question"], t["gold"]
        )
        print(f"    llm_mask={np.round(llm, 3)}")
        print(f"    abl_mask={np.round(abl, 3)}")
        llm_masks.append(llm)
        abl_masks.append(abl)
        phase_a.append(
            {
                **result,
                "llm_mask": llm.tolist(),
                "ablation_mask": abl.tolist(),
                "mask_raw": raw,
            }
        )

    print("\n=== Phase B: aggregate least-useful overlapping chunks ===")
    # Use ablation masks for stable dead-set; LLM masks kept for comparison
    dead, max_across, mean_across = aggregate_dead(
        abl_masks, mode="max_below", threshold=0.05
    )
    dead_bottom, _, _ = aggregate_dead(abl_masks, mode="bottom_frac", bottom_frac=0.4)
    dead_llm, max_llm, _ = aggregate_dead(llm_masks, mode="bottom_frac", bottom_frac=0.4)
    oracle_dead = [
        i
        for i, c in enumerate(SKILL_CHUNKS)
        if c.startswith("[Legacy]") or c.startswith("[Trivia]")
    ]
    print("ablation max_across:", np.round(max_across, 3))
    print("dead max_below:", [i + 1 for i in dead])
    print("dead bottom40%:", [i + 1 for i in dead_bottom])
    print("dead llm bottom40%:", [i + 1 for i in dead_llm])
    print("oracle dead:", [i + 1 for i in oracle_dead])

    print("\n=== Phase C: APPLY dead mask BEFORE task n+1 ===")
    policies = {
        "n1_full_skill": list(range(len(SKILL_CHUNKS))),
        "n1_prune_maxbelow": [i for i in range(len(SKILL_CHUNKS)) if i not in set(dead)],
        "n1_prune_bottom40": [
            i for i in range(len(SKILL_CHUNKS)) if i not in set(dead_bottom)
        ],
        "n1_prune_oracle": [
            i for i in range(len(SKILL_CHUNKS)) if i not in set(oracle_dead)
        ],
    }
    phase_c = []
    for name, keep in policies.items():
        r = run_task(model, tokenizer, SKILL_CHUNKS, TASK_N1, keep=keep, label=name)
        phase_c.append(r)
        print(
            f"  {name}: keep={r['keep_1based']} ok={r['correct']} "
            f"lp={r['gold_logprob']:.3f} ans={r['answer']!r}"
        )

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [2, 1.2]}
    )
    x = np.arange(len(SKILL_CHUNKS))
    w = 0.2
    for ti, (mask, task) in enumerate(zip(abl_masks, TASKS)):
        axes[0].bar(x + (ti - 1) * w, mask, w, label=f"mask after {task['id']}")
    axes[0].plot(x, max_across, "k--", lw=1.5, label="max across tasks")
    for i in dead:
        axes[0].axvline(i, color="red", alpha=0.25, lw=6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"[{i+1}]" for i in x])
    axes[0].set_ylabel("Ablation usefulness")
    axes[0].set_title(
        "After each task: usefulness mask → least-useful overlap (red) pruned for n+1"
    )
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].barh(
        [r["label"] for r in phase_c],
        [r["gold_logprob"] for r in phase_c],
        color=["#2ca02c" if r["correct"] else "#d62728" for r in phase_c],
    )
    axes[1].set_xlabel("Mean logprob of gold answer on task n+1")
    axes[1].set_title("Application: mask dead skill chunks, then run task n+1")
    fig.tight_layout()
    fig_path = FIG_DIR / "skill_bloat_pruning.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    summary = {
        "model": MODEL_NAME,
        "timing": {
            "map_generation": "after each of tasks 1..n completes",
            "aggregation": "after all n masks exist; dead := low max_across_tasks",
            "map_application": "before task n+1 only",
        },
        "skill_chunks": SKILL_CHUNKS,
        "phase_a": phase_a,
        "aggregation": {
            "signal": "ablation_mask (primary); llm_mask compared",
            "max_across_tasks": max_across.tolist(),
            "mean_across_tasks": mean_across.tolist(),
            "dead_maxbelow_1based": [i + 1 for i in dead],
            "dead_bottom40_1based": [i + 1 for i in dead_bottom],
            "dead_llm_bottom40_1based": [i + 1 for i in dead_llm],
            "llm_max_across": max_llm.tolist(),
            "oracle_dead_1based": [i + 1 for i in oracle_dead],
            "overlap_dead_vs_oracle": [
                i + 1 for i in sorted(set(dead) & set(oracle_dead))
            ],
        },
        "phase_c": phase_c,
        "figure": str(fig_path),
    }
    out_path = OUT_DIR / "skill_bloat_pruning.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
