#!/usr/bin/env python3
"""
Can we predict the mask from the question ALONE?

For each task, compare:
  post-hoc mask (question + gold answer, ablation) — the target
  vs question-only predictors:
    a) LLM relevance judge (no answer shown)
    b) TF-IDF cosine (question vs chunk, pure lexical baseline)

Metrics: Spearman vs post-hoc, top-k overlap, and downstream
gold logprob / accuracy when pruning with the PREDICTED mask.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

TASKS = [
    {"id": "task1_vpn", "question": "What must a new hire complete before day 3 for VPN enrollment?", "gold": "manager-approved ticket", "key": 1},
    {"id": "task2_meals", "question": "Do meal expenses under $75 need itemized receipts?", "gold": "no", "key": 2},
    {"id": "task3_sev1", "question": "Within how many minutes must a Sev-1 outage page the on-call?", "gold": "5", "key": 4},
    {"id": "task4_privacy", "question": "Where should customer data incidents be escalated?", "gold": "Legal", "key": 5},
]


def chat(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
    )


def format_skill(chunks, keep=None):
    idxs = list(range(len(chunks))) if keep is None else list(keep)
    return "\n".join(f"[{i+1}] {chunks[i]}" for i in idxs)


def task_prompt(tokenizer, chunks, question, keep=None):
    user = (
        "You are following company skill docs. Answer using only the skill context. "
        "Reply with a short phrase only.\n\n"
        f"Skill document:\n{format_skill(chunks, keep)}\n\nTask: {question}"
    )
    return chat(tokenizer, user)


def generate(model, tokenizer, prompt, max_new_tokens=32):
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


def llm_question_only(model, tokenizer, chunks, question):
    """Relevance from question alone — no answer shown."""
    n = len(chunks)
    skill = format_skill(chunks)
    scores = np.zeros(n)
    logs = []
    for i, c in enumerate(chunks):
        u = (
            f"Skill chunk: [{i+1}] {c}\nQuestion: {question}\n\n"
            "Is this chunk likely relevant for answering the question? Reply yes or no."
        )
        yn = generate(model, tokenizer, chat(tokenizer, u), max_new_tokens=4).lower()
        logs.append(f"{i+1}:{yn.strip()}")
        if re.search(r"\byes\b", yn) and not re.search(r"\bno\b", yn):
            scores[i] = 1.0
        elif re.search(r"\byes\b", yn):
            scores[i] = 0.5
    if scores.sum() > 0:
        scores /= scores.sum()
    return scores, "; ".join(logs)


def tfidf_scores(question, chunks):
    vec = TfidfVectorizer().fit(chunks + [question])
    C = vec.transform(chunks)
    q = vec.transform([question])
    sims = cosine_similarity(q, C)[0]
    sims = np.maximum(sims, 0)
    if sims.sum() > 0:
        sims /= sims.sum()
    return sims


def topk_overlap(a, b, k=2):
    sa = set(np.argsort(-np.asarray(a))[:k])
    sb = set(np.argsort(-np.asarray(b))[:k])
    return len(sa & sb) / k


def spear(a, b):
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    return float(spearmanr(a, b).correlation)


def main():
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    rows = []
    for t in TASKS:
        print(f"\n=== {t['id']} ===")
        post = ablation_mask(model, tok, SKILL, t["question"], t["gold"])
        llm, raw = llm_question_only(model, tok, SKILL, t["question"])
        tf = tfidf_scores(t["question"], SKILL)
        print("post-hoc :", np.round(post, 3))
        print("llm Q-only:", np.round(llm, 3), raw[:150])
        print("tfidf    :", np.round(tf, 3))

        # downstream: prune with predicted top-3, evaluate gold
        def eval_keep(keep, label):
            keep = [int(i) for i in keep]
            p = task_prompt(tok, SKILL, t["question"], keep)
            ans = generate(model, tok, p)
            lp = mean_logprob(model, tok, p, t["gold"])
            ok = t["gold"].lower() in ans.lower()
            print(f"  {label}: keep={[i+1 for i in keep]} ok={ok} lp={lp:.3f} ans={ans!r}")
            return {"label": label, "keep": keep, "ok": ok, "lp": lp, "ans": ans}

        k = 3
        keep_post = [int(i) for i in np.argsort(-post)[:k]]
        keep_llm = [int(i) for i in np.argsort(-llm)[:k]]
        keep_tf = [int(i) for i in np.argsort(-tf)[:k]]
        full = eval_keep(list(range(len(SKILL))), "full")
        r_post = eval_keep(keep_post, "pred_posthoc_top3")
        r_llm = eval_keep(keep_llm, "pred_llmQonly_top3")
        r_tf = eval_keep(keep_tf, "pred_tfidf_top3")

        rows.append({
            "task": t["id"], "key_1based": t["key"] + 1,
            "posthoc": post.tolist(), "llm_qonly": llm.tolist(), "tfidf": tf.tolist(),
            "spear_llm_vs_post": spear(llm, post),
            "spear_tfidf_vs_post": spear(tf, post),
            "overlap2_llm": topk_overlap(llm, post, 2),
            "overlap2_tfidf": topk_overlap(tf, post, 2),
            "rank_key_post": int(np.where(np.argsort(-post) == t["key"])[0][0] + 1),
            "rank_key_llm": int(np.where(np.argsort(-llm) == t["key"])[0][0] + 1),
            "rank_key_tfidf": int(np.where(np.argsort(-tf) == t["key"])[0][0] + 1),
            "downstream": {"full": full, "post": r_post, "llm": r_llm, "tfidf": r_tf},
        })

    # summary
    def avg(key):
        vs = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vs)) if vs else None

    summary = {
        "model": MODEL_ID,
        "mean_spear_llm_vs_post": avg("spear_llm_vs_post"),
        "mean_spear_tfidf_vs_post": avg("spear_tfidf_vs_post"),
        "mean_overlap2_llm": avg("overlap2_llm"),
        "mean_overlap2_tfidf": avg("overlap2_tfidf"),
        "mean_rank_key_post": avg("rank_key_post"),
        "mean_rank_key_llm": avg("rank_key_llm"),
        "mean_rank_key_tfidf": avg("rank_key_tfidf"),
        "rows": rows,
    }
    (OUT / "mask_prediction_qonly.json").write_text(json.dumps(summary, indent=2))

    # plot: rank of true key chunk under each predictor
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = [r["task"] for r in rows]
    x = np.arange(len(names))
    w = 0.25
    axes[0].bar(x - w, [r["rank_key_post"] for r in rows], w, label="post-hoc (target)")
    axes[0].bar(x, [r["rank_key_llm"] for r in rows], w, label="LLM Q-only")
    axes[0].bar(x + w, [r["rank_key_tfidf"] for r in rows], w, label="TF-IDF Q-only")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel("Rank of true key chunk (1=best)")
    axes[0].set_title("Can question-only routing find the right chunk?")
    axes[0].legend()
    axes[0].invert_yaxis()

    labels = ["full", "post top3", "llmQ top3", "tfidf top3"]
    lps = []
    for lab in ["full", "post", "llm", "tfidf"]:
        lps.append(np.mean([r["downstream"][lab]["lp"] for r in rows]))
    oks = []
    for lab in ["full", "post", "llm", "tfidf"]:
        oks.append(np.mean([float(r["downstream"][lab]["ok"]) for r in rows]))
    axes[1].bar(labels, lps, color="#4c72b0")
    axes[1].set_ylabel("Mean gold logprob")
    axes[1].set_title(f"Downstream after Q-only pruning (acc: {['%.2f' % v for v in oks]})")
    fig.tight_layout()
    fig.savefig(FIG / "mask_prediction_qonly.png", dpi=140)
    plt.close(fig)

    print("\n=== SUMMARY ===")
    print("mean spear llm-vs-post:", summary["mean_spear_llm_vs_post"])
    print("mean spear tfidf-vs-post:", summary["mean_spear_tfidf_vs_post"])
    print("mean rank key post/llm/tfidf:", summary["mean_rank_key_post"], summary["mean_rank_key_llm"], summary["mean_rank_key_tfidf"])
    print("mean gold lp:", {lab: float(np.mean([r['downstream'][lab]['lp'] for r in rows])) for lab in ['full','post','llm','tfidf']})
    print("mean acc:", {lab: float(np.mean([r['downstream'][lab]['ok'] for r in rows])) for lab in ['full','post','llm','tfidf']})
    print("Wrote artifacts/mask_prediction_qonly.json")


if __name__ == "__main__":
    main()
