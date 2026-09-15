#!/usr/bin/env python3
"""
Multi-question pruning demo:
  Shared context → answer Q1 → prune Q1-only evidence → answer Q2.

Compares:
  - full context
  - prune by LLM usefulness for *current* question (keep Q2-useful)
  - prune by forgetting Q1-useful sentences (cross-question forget)
  - prune by attention for current question
  - recent-window baseline (keep last half of sentences)
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
OUT = ROOT / "artifacts"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

SENTENCES = [
    "The capital of Zeldria is Mirathen.",  # Q1 key
    "Zeldria's currency is called the vort.",  # Q1 support
    "Mount Cindara is the tallest peak in Zeldria.",  # distractor
    "At sea level, Brammel boils at 87 degrees Celsius.",  # Q2 key
    "Brammel freezes at -12 degrees Celsius.",  # Q2 support
    "Neon signs were invented long before Brammel was isolated.",  # distractor
]

Q1 = {
    "question": "What is the capital of Zeldria?",
    "gold": "Mirathen",
    "key": {0},
}
Q2 = {
    "question": "At sea level, what is the boiling point of Brammel in Celsius?",
    "gold": "87",
    "key": {3},
}


def chat(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def ctx(sents: list[str], idxs: list[int] | None = None) -> str:
    idxs = idxs if idxs is not None else list(range(len(sents)))
    return "\n".join(f"[{i+1}] {sents[i]}" for i in idxs)


def qa_prompt(tokenizer, sents: list[str], question: str, keep: list[int]) -> str:
    user = (
        "Answer using only the context. Reply with a short phrase only.\n\n"
        f"Context:\n{ctx(sents, keep)}\n\nQuestion: {question}"
    )
    return chat(tokenizer, user)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 24) -> str:
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
    lps = []
    for i in range(a.shape[1]):
        lp = torch.log_softmax(logits[0, p.shape[1] + i - 1], dim=-1)
        lps.append(lp[a[0, i]].item())
    return float(np.mean(lps))


def llm_usefulness(model, tokenizer, sents: list[str], question: str, gold: str) -> np.ndarray:
    n = len(sents)
    context = ctx(sents)
    pick_user = (
        f"Context:\n{context}\n\nQuestion: {question}\nCorrect answer: {gold}\n\n"
        f"Which ONE context sentence number (1-{n}) most directly provides the answer? "
        "Reply with only the number."
    )
    pick_raw = generate(model, tokenizer, chat(tokenizer, pick_user), max_new_tokens=8)
    m = re.search(r"\b([1-9])\b", pick_raw)
    pick = int(m.group(1)) if m else None

    votes = np.zeros(n)
    logs = []
    for i, s in enumerate(sents):
        yn_user = (
            f"Context sentence: [{i+1}] {s}\nQuestion: {question}\n"
            f"Correct answer: {gold}\n\n"
            "Is this sentence necessary to know the answer? Reply yes or no."
        )
        yn = generate(model, tokenizer, chat(tokenizer, yn_user), max_new_tokens=4).lower()
        logs.append(f"{i+1}:{yn}")
        if re.search(r"\byes\b", yn) and not re.search(r"\bno\b", yn):
            votes[i] = 1.0
        elif re.search(r"\byes\b", yn):
            votes[i] = 0.5
    scores = votes.copy()
    if pick is not None and 1 <= pick <= n:
        scores[pick - 1] += 2.0
    if scores.sum() == 0 and pick is not None and 1 <= pick <= n:
        scores = np.zeros(n)
        scores[pick - 1] = 1.0
    if scores.sum() > 0:
        scores = scores / scores.sum()
    return scores, f"pick={pick_raw.strip()} | {'; '.join(logs)}"


def attention_usefulness(
    model, tokenizer, sents: list[str], question: str, gold: str
) -> np.ndarray:
    keep = list(range(len(sents)))
    prompt = qa_prompt(tokenizer, sents, question, keep)
    context = ctx(sents, keep)
    offset = prompt.find(context)
    p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    a_ids = tokenizer(
        " " + gold, return_tensors="pt", add_special_tokens=False
    ).to(DEVICE)
    full = torch.cat([p_ids["input_ids"], a_ids["input_ids"]], dim=1)
    with torch.no_grad():
        out = model(full, output_attentions=True, use_cache=False)
    att = torch.stack([a[0] for a in out.attentions]).mean(dim=(0, 1))  # S,S
    plen = p_ids["input_ids"].shape[1]
    qidx = list(range(plen, plen + a_ids["input_ids"].shape[1]))
    paid = att[qidx, :plen].mean(0).cpu().numpy()

    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    scores = np.zeros(len(sents))
    for i, s in enumerate(sents):
        needle = f"[{i+1}] {s}"
        c0 = context.find(needle)
        c1 = c0 + len(needle)
        abs0, abs1 = offset + c0, offset + c1
        idxs = [ti for ti, (a, b) in enumerate(offsets) if b > abs0 and a < abs1 and a != b]
        if idxs:
            scores[i] = float(paid[idxs].sum())
    if scores.sum() > 0:
        scores /= scores.sum()
    return scores


def keep_topk(scores: np.ndarray, k: int) -> list[int]:
    order = list(np.argsort(-scores))
    return sorted(order[:k])


def keep_above(scores: np.ndarray, frac_of_max: float = 0.25) -> list[int]:
    thr = scores.max() * frac_of_max
    kept = [i for i, s in enumerate(scores) if s >= thr and s > 0]
    return kept if kept else keep_topk(scores, 1)


def evaluate(model, tokenizer, sents, question, gold, keep, label):
    prompt = qa_prompt(tokenizer, sents, question, keep)
    ans = generate(model, tokenizer, prompt)
    lp = mean_logprob(model, tokenizer, prompt, gold)
    ok = gold.lower() in ans.lower()
    return {
        "policy": label,
        "keep_1based": [i + 1 for i in keep],
        "n_keep": len(keep),
        "answer": ans,
        "gold_logprob": lp,
        "correct": ok,
        "kept_key": any(i in Q2["key"] for i in keep)
        if question == Q2["question"]
        else any(i in Q1["key"] for i in keep),
    }


def main():
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    # Maps for each question
    llm1, raw1 = llm_usefulness(model, tok, SENTENCES, Q1["question"], Q1["gold"])
    llm2, raw2 = llm_usefulness(model, tok, SENTENCES, Q2["question"], Q2["gold"])
    attn1 = attention_usefulness(model, tok, SENTENCES, Q1["question"], Q1["gold"])
    attn2 = attention_usefulness(model, tok, SENTENCES, Q2["question"], Q2["gold"])

    print("LLM Q1:", np.round(llm1, 3), raw1)
    print("LLM Q2:", np.round(llm2, 3), raw2)
    print("Attn Q1:", np.round(attn1, 3))
    print("Attn Q2:", np.round(attn2, 3))

    # Cross-question forget: remove sentences that are Q1-useful and not Q2-useful
    q1_mark = llm1 >= (llm1.max() * 0.25) if llm1.max() > 0 else np.zeros_like(llm1, dtype=bool)
    q2_mark = llm2 >= (llm2.max() * 0.25) if llm2.max() > 0 else np.zeros_like(llm2, dtype=bool)
    forget_q1 = [i for i in range(len(SENTENCES)) if not (q1_mark[i] and not q2_mark[i])]
    # If everything forgotten somehow, keep all
    if not forget_q1:
        forget_q1 = list(range(len(SENTENCES)))

    policies = {
        "full_context": list(range(len(SENTENCES))),
        "llm_keep_current_q2": keep_above(llm2, 0.25),
        "llm_forget_q1_only": forget_q1,
        "attn_keep_current_q2": keep_above(attn2, 0.25),
        "keep_last_half": list(range(len(SENTENCES) // 2, len(SENTENCES))),
        "oracle_q2_keys": sorted(Q2["key"] | {4}),  # key + support
    }

    # Also answer Q1 on full context for sanity
    q1_full = evaluate(
        model, tok, SENTENCES, Q1["question"], Q1["gold"], policies["full_context"], "q1_full"
    )
    print("Q1 full:", q1_full)

    rows = []
    for name, keep in policies.items():
        row = evaluate(model, tok, SENTENCES, Q2["question"], Q2["gold"], keep, name)
        rows.append(row)
        print(name, row)

    # Plot maps + Q2 logprobs under policies
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(SENTENCES))
    w = 0.2
    axes[0].bar(x - 1.5 * w, llm1, w, label="LLM Q1")
    axes[0].bar(x - 0.5 * w, llm2, w, label="LLM Q2")
    axes[0].bar(x + 0.5 * w, attn1, w, label="Attn Q1")
    axes[0].bar(x + 1.5 * w, attn2, w, label="Attn Q2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"[{i+1}]" for i in x])
    axes[0].set_title("Usefulness maps by question")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Normalized score")

    names = [r["policy"] for r in rows]
    lps = [r["gold_logprob"] for r in rows]
    colors = ["#2ca02c" if r["correct"] else "#d62728" for r in rows]
    axes[1].barh(names, lps, color=colors)
    axes[1].set_xlabel("Mean logprob of gold Q2 answer")
    axes[1].set_title("Q2 after pruning (green=correct)")
    fig.tight_layout()
    fig.savefig(FIG / "multi_question_pruning.png", dpi=140)
    plt.close(fig)

    summary = {
        "model": MODEL_ID,
        "sentences": SENTENCES,
        "q1": {**Q1, "key": sorted(Q1["key"])},
        "q2": {**Q2, "key": sorted(Q2["key"])},
        "maps": {
            "llm_q1": llm1.tolist(),
            "llm_q2": llm2.tolist(),
            "attn_q1": attn1.tolist(),
            "attn_q2": attn2.tolist(),
            "llm_q1_raw": raw1,
            "llm_q2_raw": raw2,
        },
        "q1_full": q1_full,
        "q2_policies": rows,
    }
    (OUT / "multi_question_pruning.json").write_text(json.dumps(summary, indent=2))
    print("Wrote artifacts/multi_question_pruning.json")


if __name__ == "__main__":
    main()
