#!/usr/bin/env python3
"""
Time-only EMA mask pruning (no question routing).

Assumption: task usefulness is AUTOCORRELATED in time —
recent tasks predict near-future tasks. So the mask for step t+1
is just an exponential moving average of past post-hoc masks:

    EMA_t = λ·EMA_{t-1} + (1-λ)·mask_t

At step t, prune using ONLY EMA_{t-1} (no question content):
    keep = top-m chunks of EMA_{t-1}  (+ safety: always keep per-chunk floor)

Task stream is a Markov chain over 4 task types with stay-prob rho
(high rho = strong autocorrelation → EMA should work;
 low rho = near-iid → EMA should fail to beat static).

Per step we record:
  - Spearman(EMA_{t-1}, true mask_t): "predictability from time alone"
  - full vs EMA-pruned accuracy / gold logprob / tokens kept
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

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

TASK_TYPES = [
    {"id": "vpn", "question": "What must a new hire complete before day 3 for VPN enrollment?", "gold": "manager-approved ticket", "key": 1},
    {"id": "meals", "question": "Do meal expenses under $75 need itemized receipts?", "gold": "no", "key": 2},
    {"id": "sev1", "question": "Within how many minutes must a Sev-1 outage page the on-call?", "gold": "5", "key": 4},
    {"id": "privacy", "question": "Where should customer data incidents be escalated?", "gold": "Legal", "key": 5},
]

# Stream params
T = 16
RHO = 0.75       # P(stay in same task type); autocorrelation strength
SEED = 7
LAMBDAS = [0.9, 0.7]
KEEP_M = 5       # keep top-M of EMA (50% tokens)
WARMUP = 2       # first WARMUP steps run full (no history yet)


def chat(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
    )


def format_skill(chunks, keep=None):
    idxs = list(range(len(chunks))) if keep is None else [int(i) for i in keep]
    return "\n".join(f"[{i+1}] {chunks[i]}" for i in idxs)


def task_prompt(tokenizer, chunks, question, keep=None):
    user = (
        "You are following company skill docs. Answer using only the skill context. "
        "Reply with a short phrase only.\n\n"
        f"Skill document:\n{format_skill(chunks, keep)}\n\nTask: {question}"
    )
    return chat(tokenizer, user)


def generate(model, tokenizer, prompt, max_new_tokens=24):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def mean_logprob(model, tokenizer, prompt, answer):
    p = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    a = tokenizer(" " + answer, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    full = torch.cat([p, a], dim=1)
    with torch.no_grad():
        logits = model(full).logits
    vals = []
    for i in range(a.shape[1]):
        lp = torch.log_softmax(logits[0, p.shape[1] + i - 1], dim=-1)
        vals.append(lp[a[0, i]].item())
    return float(np.mean(vals))


def ablation_mask(model, tokenizer, chunks, question, gold):
    base = mean_logprob(model, tokenizer, task_prompt(tokenizer, chunks, question), gold)
    drops = []
    for i in range(len(chunks)):
        red = list(chunks)
        red[i] = "[REDACTED]"
        lp = mean_logprob(model, tokenizer, task_prompt(tokenizer, red, question), gold)
        drops.append(base - lp)
    drops = np.maximum(np.asarray(drops, dtype=float), 0.0)
    if drops.sum() > 0:
        drops /= drops.sum()
    return drops


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    return float(spearmanr(a, b).correlation)


def build_stream(rho=RHO, T=T, seed=SEED):
    rng = np.random.default_rng(seed)
    seq = [int(rng.integers(len(TASK_TYPES)))]
    for _ in range(1, T):
        if rng.random() < rho:
            seq.append(seq[-1])
        else:
            opts = [i for i in range(len(TASK_TYPES)) if i != seq[-1]]
            seq.append(int(rng.choice(opts)))
    return seq


def main():
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    seq = build_stream()
    print("Task stream:", [TASK_TYPES[i]["id"] for i in seq])

    n_chunks = len(SKILL)
    emas = {lam: np.full(n_chunks, 1.0 / n_chunks) for lam in LAMBDAS}
    ema_hist = {lam: [] for lam in LAMBDAS}
    true_masks = []
    steps = []

    for t, ti in enumerate(seq):
        task = TASK_TYPES[ti]
        q, gold = task["question"], task["gold"]

        # --- predict from time alone: keep top-M of EMA_{t-1} ---
        policies = {"full": list(range(n_chunks))}
        for lam in LAMBDAS:
            if t < WARMUP:
                keep = list(range(n_chunks))
            else:
                keep = [int(i) for i in np.argsort(-emas[lam])[:KEEP_M]]
            policies[f"ema_lam{lam}"] = sorted(keep)

        results = {}
        for name, keep in policies.items():
            p = task_prompt(tok, SKILL, q, keep)
            ans = generate(model, tok, p)
            lp = mean_logprob(model, tok, p, gold)
            ok = gold.lower() in ans.lower()
            results[name] = {"keep": [int(i) + 1 for i in keep], "ok": bool(ok), "lp": lp, "ans": ans}

        # --- observe true mask AFTER task (full skill, gold-targeted) ---
        mask_t = ablation_mask(model, tok, SKILL, q, gold)
        true_masks.append(mask_t)

        # predictability of time-only EMA
        preds = {}
        for lam in LAMBDAS:
            preds[f"lam{lam}"] = spear(emas[lam], mask_t)

        # --- update EMAs ---
        for lam in LAMBDAS:
            emas[lam] = lam * emas[lam] + (1 - lam) * mask_t
            ema_hist[lam].append(emas[lam].copy())

        key_rank_full = None
        row = {
            "t": t, "task": task["id"], "gold": gold,
            "true_mask": [float(v) for v in mask_t],
            "spear_ema_vs_true": preds,
            "results": results,
        }
        steps.append(row)
        print(
            f"[t={t:02d} {task['id']:7s}] true_key=[{task['key']+1}] "
            + " ".join(f"spear(EMA{lam})={preds[f'lam{lam}']}" for lam in LAMBDAS)
        )
        for name, r in results.items():
            print(f"    {name:14s} ok={r['ok']} lp={r['lp']:.3f} keep={r['keep']} ans={r['ans']!r}")

    # ---- summary stats ----
    def agg(policy):
        oks = [float(s["results"][policy]["ok"]) for s in steps]
        lps = [s["results"][policy]["lp"] for s in steps]
        keeps = [len(s["results"][policy]["keep"]) for s in steps]
        return {"acc": float(np.mean(oks)), "mean_lp": float(np.mean(lps)), "mean_keep": float(np.mean(keeps))}

    summary = {
        "model": MODEL_ID,
        "params": {"T": T, "rho": RHO, "seed": SEED, "lambdas": LAMBDAS, "keep_m": KEEP_M, "warmup": WARMUP},
        "stream": [TASK_TYPES[i]["id"] for i in seq],
        "policy_summary": {p: agg(p) for p in ["full"] + [f"ema_lam{lam}" for lam in LAMBDAS]},
        "mean_spear_ema_vs_true": {
            f"lam{lam}": float(np.mean([s["spear_ema_vs_true"][f"lam{lam}"] for s in steps[WARMUP:] if s["spear_ema_vs_true"][f"lam{lam}"] is not None]))
            for lam in LAMBDAS
        },
        "steps": steps,
        "ema_history": {str(lam): [list(map(float, v)) for v in ema_hist[lam]] for lam in LAMBDAS},
        "true_masks": [list(map(float, m)) for m in true_masks],
    }
    (OUT / "time_ema_pruning.json").write_text(json.dumps(summary, indent=2))

    # ---- plots ----
    lam0 = LAMBDAS[0]
    tm = np.array(summary["true_masks"])          # T x C
    em = np.array(summary["ema_history"][str(lam0)])  # T x C
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].imshow(tm.T, aspect="auto", cmap="YlOrRd")
    axes[0].set_yticks(range(n_chunks))
    axes[0].set_yticklabels([f"[{i+1}]" for i in range(n_chunks)], fontsize=8)
    axes[0].set_xlabel("time step t")
    axes[0].set_title("True post-hoc mask per step (rows=chunks)")
    axes[1].imshow(em.T, aspect="auto", cmap="YlOrRd")
    axes[1].set_yticks(range(n_chunks))
    axes[1].set_yticklabels([f"[{i+1}]" for i in range(n_chunks)], fontsize=8)
    axes[1].set_xlabel("time step t")
    axes[1].set_title(f"Time-only EMA (λ={lam0}) — no question content used")

    xs = list(range(T))
    axes[2].plot(xs, [s["results"]["full"]["lp"] for s in steps], "o-", label="full")
    for lam in LAMBDAS:
        axes[2].plot(xs, [s["results"][f"ema_lam{lam}"]["lp"] for s in steps], "s--", label=f"EMA λ={lam} top-{KEEP_M}")
    axes[2].set_xlabel("time step t")
    axes[2].set_ylabel("gold logprob")
    axes[2].set_title("Full vs time-EMA-pruned (same accuracy check in JSON)")
    axes[2].legend(fontsize=8)
    for i, s in enumerate(steps):
        axes[2].text(i, axes[2].get_ylim()[0], s["task"][:4], fontsize=6, ha="center")
    fig.tight_layout()
    fig.savefig(FIG / "time_ema_pruning.png", dpi=140)
    plt.close(fig)

    print("\n=== SUMMARY ===")
    print("stream:", summary["stream"])
    print("policy:", json.dumps(summary["policy_summary"], indent=2))
    print("mean spear EMA-vs-true:", summary["mean_spear_ema_vs_true"])
    print("Wrote artifacts/time_ema_pruning.json + figures/time_ema_pruning.png")


if __name__ == "__main__":
    main()
