#!/usr/bin/env python3
"""
Quantify: hard skill deletion vs soft attention masking of "dead" chunks.

Hard delete:
  dead text removed from the prompt entirely → cannot affect anything.

Soft attention mask (user's point #2):
  dead chunks remain in the prompt.
  During prefilling, skill tokens may still attend to dead tokens (leakage into
  other skill representations).
  Answer-token queries are blocked from attending to dead key positions.

Also contrasts with "LLM, please delete dead parts" (single-shot judgment),
which is hard-delete by nature and awkward to aggregate across tasks.
"""

from __future__ import annotations

import json
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

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
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

# Use oracle dead set so the comparison isolates hard vs soft, not mask quality.
ORACLE_DEAD = [6, 7, 8, 9]  # legacy + trivia (0-based)

TASK = {
    "question": "Where should customer data incidents be escalated?",
    "gold": "Legal",
    "true_useful": 5,
}


def chat(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_skill(chunks, keep=None) -> str:
    idxs = list(range(len(chunks))) if keep is None else list(keep)
    return "\n".join(f"[{i+1}] {chunks[i]}" for i in idxs)


def make_prompt(tokenizer, chunks, question, keep=None) -> tuple[str, str]:
    skill = format_skill(chunks, keep)
    user = (
        "You are following company skill docs. Answer using only the skill context. "
        "Reply with a short phrase only.\n\n"
        f"Skill document:\n{skill}\n\nTask: {question}"
    )
    prompt = chat(tokenizer, user)
    return prompt, skill


def token_spans_for_chunks(tokenizer, prompt: str, skill: str, chunks: list[str]):
    """Map each chunk index → token indices inside full prompt."""
    offset = prompt.find(skill)
    if offset < 0:
        raise RuntimeError("skill block not found in prompt")
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    spans = []
    for i, c in enumerate(chunks):
        needle = f"[{i+1}] {c}"
        c0 = skill.find(needle)
        if c0 < 0:
            spans.append([])
            continue
        c1 = c0 + len(needle)
        a0, a1 = offset + c0, offset + c1
        idxs = [ti for ti, (a, b) in enumerate(offsets) if b > a0 and a < a1 and a != b]
        spans.append(idxs)
    return spans


def mean_answer_logprob(model, tokenizer, prompt: str, answer: str, attn4d=None) -> float:
    p = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    a = tokenizer(" " + answer, return_tensors="pt", add_special_tokens=False)
    p_ids = p["input_ids"].to(DEVICE)
    a_ids = a["input_ids"].to(DEVICE)
    full = torch.cat([p_ids, a_ids], dim=1)
    plen, alen, slen = p_ids.shape[1], a_ids.shape[1], full.shape[1]

    kwargs = {}
    if attn4d is None:
        kwargs["attention_mask"] = torch.ones_like(full)
    else:
        # Extend 4D mask from prompt-only to prompt+answer if needed
        # attn4d expected shape (1,1,Q,K) for sequence length slen
        kwargs["attention_mask"] = attn4d

    with torch.no_grad():
        logits = model(full, **kwargs).logits

    lps = []
    for i in range(alen):
        lp = torch.log_softmax(logits[0, plen + i - 1], dim=-1)
        lps.append(lp[a_ids[0, i]].item())
    return float(np.mean(lps))


def causal_bias(q_len, kv_len, device):
    """Standard causal additive mask: 0 keep, -inf block future."""
    # allow attend to j <= i (for the overlapping prefix)
    m = torch.zeros(q_len, kv_len, device=device)
    for i in range(q_len):
        # keys beyond this query's position blocked if kv aligns with same seq
        if kv_len == q_len:
            m[i, i + 1 :] = torch.finfo(torch.float32).min
        else:
            # when kv_len == full and q is full, same
            m[i, i + 1 :] = torch.finfo(torch.float32).min
    return m


def build_soft_mask(
    seq_len: int,
    prompt_len: int,
    dead_token_idxs: list[int],
    mode: str,
) -> torch.Tensor:
    """
    Returns additive attention bias (1,1,L,L).

    mode:
      soft_answer_block:
        all queries use causal mask;
        PLUS answer queries cannot attend dead keys.
        Skill/question queries CAN attend dead keys → leakage path.
      soft_postskill_block:
        stronger: any query after skill (question+answer) cannot attend dead keys.
        Skill-internal still can.
      hard_block_all_queries:
        every query (including other skill tokens) cannot attend dead keys.
        Closer to deletion but dead tokens still occupy positions / residual as values
        only for queries that somehow see them — here nobody sees them as keys.
    """
    neg = torch.finfo(torch.float32).min
    bias = torch.zeros(1, 1, seq_len, seq_len, device=DEVICE)
    # causal
    for i in range(seq_len):
        bias[0, 0, i, i + 1 :] = neg

    dead = sorted(set(dead_token_idxs))
    answer_q = list(range(prompt_len, seq_len))
    # Approximate "post skill" as last 40% of prompt is question scaffolding —
    # cleaner: block only answer queries for soft_answer_block.

    if mode == "soft_answer_block":
        for q in answer_q:
            for d in dead:
                if d <= q:
                    bias[0, 0, q, d] = neg
    elif mode == "soft_postprompt_block":
        # question+answer queries: anything from mid-prompt is messy; use answer only
        # plus final 32 prompt tokens as "task" region heuristic
        task_start = max(0, prompt_len - 32)
        for q in list(range(task_start, seq_len)):
            for d in dead:
                if d <= q:
                    bias[0, 0, q, d] = neg
    elif mode == "hard_block_all_queries":
        for q in range(seq_len):
            for d in dead:
                if d <= q:
                    bias[0, 0, q, d] = neg
    else:
        raise ValueError(mode)
    return bias


def generate(model, tokenizer, prompt, max_new_tokens=24):
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


def llm_delete_dead(model, tokenizer, chunks, tasks_summary: str) -> list[int]:
    """
    Single-shot LLM judgment: which chunks to delete.
    If tasks_summary is short = only current task; long = all task rollouts in context.
    Returns keep indices.
    """
    skill = format_skill(chunks)
    user = (
        f"Skill document:\n{skill}\n\n"
        f"Observed task history:\n{tasks_summary}\n\n"
        "Delete skill chunks that look like dead/legacy/trivia knowledge never needed "
        f"by those tasks. Keep procedural chunks that might be needed later. "
        f"Reply with a comma-separated list of chunk numbers TO DELETE (1-{len(chunks)})."
    )
    raw = generate(model, tokenizer, chat(tokenizer, user), max_new_tokens=48)
    nums = [int(x) for x in re.findall(r"\b([1-9]|10)\b", raw)]
    delete = {n - 1 for n in nums if 1 <= n <= len(chunks)}
    # safety: never delete the known true-useful for upcoming task if model goes crazy
    keep = [i for i in range(len(chunks)) if i not in delete]
    return keep, raw


# fix missing import
import re


def main():
    print(f"Loading {MODEL}...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    dead = ORACLE_DEAD
    keep_hard = [i for i in range(len(SKILL)) if i not in dead]

    # --- prompts ---
    prompt_full, skill_full = make_prompt(tok, SKILL, TASK["question"], keep=None)
    prompt_hard, skill_hard = make_prompt(tok, SKILL, TASK["question"], keep=keep_hard)

    spans = token_spans_for_chunks(tok, prompt_full, skill_full, SKILL)
    dead_toks = [ti for i in dead for ti in spans[i]]
    print(f"Dead chunks {[i+1 for i in dead]} → {len(dead_toks)} tokens")

    gold = TASK["gold"]

    # sequence lengths for soft masks
    p_ids = tok(prompt_full, return_tensors="pt", add_special_tokens=False)["input_ids"]
    a_ids = tok(" " + gold, return_tensors="pt", add_special_tokens=False)["input_ids"]
    plen, alen = p_ids.shape[1], a_ids.shape[1]
    slen = plen + alen

    results = []

    def add(name, prompt, attn=None, note=""):
        lp = mean_answer_logprob(model, tok, prompt, gold, attn4d=attn)
        ans = generate(model, tok, prompt)
        ok = gold.lower() in ans.lower()
        row = {
            "condition": name,
            "gold_logprob": lp,
            "answer": ans,
            "correct": ok,
            "note": note,
        }
        results.append(row)
        print(f"{name:28s} lp={lp:8.3f} ok={ok} ans={ans!r}")

    # 1) Full context
    add("full_context", prompt_full, None, "dead chunks present, normal attention")

    # 2) Hard delete
    add("hard_delete", prompt_hard, None, "dead chunks removed from prompt string")

    # 3) Soft: answer queries blocked from dead keys; skill can still read dead
    bias_ans = build_soft_mask(slen, plen, dead_toks, "soft_answer_block")
    add(
        "soft_answer_block",
        prompt_full,
        bias_ans,
        "dead remain; only answer queries blocked from dead keys (leakage into skill)",
    )

    # 4) Soft-stronger: late prompt + answer blocked from dead
    bias_post = build_soft_mask(slen, plen, dead_toks, "soft_postprompt_block")
    add(
        "soft_postprompt_block",
        prompt_full,
        bias_post,
        "dead remain; task/answer region blocked from dead keys",
    )

    # 5) Hard attention block: no query may attend dead keys
    bias_all = build_soft_mask(slen, plen, dead_toks, "hard_block_all_queries")
    add(
        "attn_block_all_queries",
        prompt_full,
        bias_all,
        "dead remain as tokens but never usable as keys",
    )

    # 6) LLM delete based on ONE task only
    keep1, raw1 = llm_delete_dead(
        model,
        tok,
        SKILL,
        "Task: Within how many minutes must a Sev-1 outage page the on-call?\n"
        "Answer: 5 minutes",
    )
    prompt_llm1, _ = make_prompt(tok, SKILL, TASK["question"], keep=keep1)
    add(
        "llm_delete_one_task",
        prompt_llm1,
        None,
        f"keep={ [i+1 for i in keep1] } raw={raw1!r}",
    )

    # 7) LLM delete with all n task rollouts in context
    history = (
        "Task1: What must a new hire complete before day 3 for VPN enrollment?\n"
        "Answer1: manager-approved ticket\n"
        "Task2: Do meal expenses under $75 need itemized receipts?\n"
        "Answer2: no\n"
        "Task3: Within how many minutes must a Sev-1 outage page the on-call?\n"
        "Answer3: 5"
    )
    keepn, rawn = llm_delete_dead(model, tok, SKILL, history)
    prompt_llmn, _ = make_prompt(tok, SKILL, TASK["question"], keep=keepn)
    add(
        "llm_delete_all_tasks_in_ctx",
        prompt_llmn,
        None,
        f"keep={ [i+1 for i in keepn] } raw={rawn!r}",
    )

    # Leakage metric: how much soft_answer retains full vs moves toward hard delete
    lp = {r["condition"]: r["gold_logprob"] for r in results}
    full, hard, soft = lp["full_context"], lp["hard_delete"], lp["soft_answer_block"]
    denom = full - hard
    if abs(denom) > 1e-6:
        # 0 = identical to hard delete; 1 = identical to full (max leakage)
        leakage_index = (soft - hard) / denom
    else:
        leakage_index = None

    summary = {
        "model": MODEL,
        "task": TASK,
        "oracle_dead_1based": [i + 1 for i in dead],
        "dead_token_count": len(dead_toks),
        "results": results,
        "leakage_index_soft_answer": leakage_index,
        "leakage_index_definition": (
            "(lp_soft_answer - lp_hard_delete) / (lp_full - lp_hard_delete); "
            "1≈full leakage, 0≈no leakage (soft≈hard delete)"
        ),
    }
    out = OUT / "hard_vs_soft_pruning.json"
    out.write_text(json.dumps(summary, indent=2))

    # plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = [r["condition"] for r in results]
    vals = [r["gold_logprob"] for r in results]
    colors = ["#2ca02c" if r["correct"] else "#d62728" for r in results]
    ax.barh(names, vals, color=colors)
    ax.axvline(full, color="gray", ls=":", lw=1, label="full")
    ax.axvline(hard, color="black", ls="--", lw=1, label="hard delete")
    ax.set_xlabel("Mean logprob of gold answer")
    ax.set_title(
        f"Hard delete vs soft attn mask (leakage_index={leakage_index:.3f})"
        if leakage_index is not None
        else "Hard delete vs soft attn mask"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig_path = FIG / "hard_vs_soft_pruning.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"\nLeakage index (soft_answer): {leakage_index}")
    print(f"Wrote {out}\nWrote {fig_path}")


if __name__ == "__main__":
    main()
