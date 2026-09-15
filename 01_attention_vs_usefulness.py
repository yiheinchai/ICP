#!/usr/bin/env python3
"""
Compare attention, LLM self-reported usefulness, and causal ablation
on synthetic facts (so parametric memory can't bypass the context).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts"
FIG_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
DTYPE = torch.float32

# Synthetic world: answers must come from context, not memorized facts.
EXAMPLES = [
    {
        "id": "zeldria_capital",
        "sentences": [
            "Zeldria's national animal is the silver fox.",
            "Mount Cindara is the tallest peak in Zeldria.",
            "The capital of Zeldria is Mirathen.",
            "Zeldrian coffee is exported mainly in spring.",
            "The River Quell runs west across Zeldria.",
            "Zeldria's currency is called the vort.",
        ],
        "question": "What is the capital of Zeldria?",
        "gold_answer": "Mirathen",
        "key_sentence": 2,  # 0-based
    },
    {
        "id": "brammel_boiling",
        "sentences": [
            "Brammel freezes at -12 degrees Celsius.",
            "At sea level, Brammel boils at 87 degrees Celsius.",
            "Brammel is a pale blue liquid at room temperature.",
            "Brammel was first isolated in 1911.",
            "Pure Brammel has no odor.",
            "Standard labs store Brammel in glass vials.",
        ],
        "question": "At sea level, what is the boiling point of Brammel in Celsius?",
        "gold_answer": "87",
        "key_sentence": 1,
    },
    {
        "id": "novara_author",
        "sentences": [
            "Lina Voss wrote the play Glass Harbor.",
            "The novel Night of Novara was written by Kest Relm.",
            "Night of Novara is set on a desert moon.",
            "Kest Relm was born in South Oriel.",
            "Glass Harbor premiered in 2044.",
            "The Pacific Archive ranks Novara among cult classics.",
        ],
        "question": "Who wrote the novel Night of Novara?",
        "gold_answer": "Kest Relm",
        "key_sentence": 1,
    },
]


def build_context(sentences: list[str]) -> str:
    return "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sentences))


def chat_text(tokenizer, user: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def qa_prompt(tokenizer, sentences: list[str], question: str) -> tuple[str, str, int]:
    context = build_context(sentences)
    user = (
        "Answer using only the context. Reply with a short phrase only.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    prompt = chat_text(tokenizer, user)
    offset = prompt.find(context)
    if offset < 0:
        raise RuntimeError("context missing from prompt")
    return prompt, context, offset


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 24) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def mean_answer_logprob(model, tokenizer, prompt: str, answer: str) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(DEVICE)
    # Leading space helps tokenization continuity for many BPE models
    answer_ids = tokenizer(
        " " + answer if not answer.startswith(" ") else answer,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(DEVICE)
    if answer_ids.numel() == 0:
        return float("nan")
    full = torch.cat([prompt_ids, answer_ids], dim=1)
    with torch.no_grad():
        logits = model(full).logits
    prompt_len = prompt_ids.shape[1]
    lps = []
    for i in range(answer_ids.shape[1]):
        lp = torch.log_softmax(logits[0, prompt_len + i - 1], dim=-1)
        lps.append(lp[answer_ids[0, i]].item())
    return float(np.mean(lps))


def sentence_char_spans(context: str, sentences: list[str]) -> list[tuple[int, int]]:
    spans = []
    for i, s in enumerate(sentences):
        needle = f"[{i + 1}] {s}"
        start = context.find(needle)
        if start < 0:
            raise ValueError(needle)
        spans.append((start, start + len(needle)))
    return spans


def token_spans_for_sentences(
    tokenizer, prompt: str, context: str, sentences: list[str], context_offset: int
) -> list[list[int]]:
    encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    char_spans = sentence_char_spans(context, sentences)
    token_lists: list[list[int]] = []
    for c0, c1 in char_spans:
        abs0, abs1 = context_offset + c0, context_offset + c1
        idxs = [
            ti for ti, (a, b) in enumerate(offsets) if b > abs0 and a < abs1 and a != b
        ]
        token_lists.append(idxs)
    return token_lists


def attention_sentence_scores(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    context: str,
    sentences: list[str],
    context_offset: int,
) -> dict[str, np.ndarray]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        DEVICE
    )
    answer_ids = tokenizer(
        " " + answer, return_tensors="pt", add_special_tokens=False
    ).to(DEVICE)
    full_ids = torch.cat([prompt_ids["input_ids"], answer_ids["input_ids"]], dim=1)
    with torch.no_grad():
        out = model(full_ids, output_attentions=True, use_cache=False)
    att = torch.stack([a[0] for a in out.attentions], dim=0)  # L,H,S,S
    prompt_len = prompt_ids["input_ids"].shape[1]
    ans_len = answer_ids["input_ids"].shape[1]
    q_idx = list(range(prompt_len, prompt_len + ans_len))

    variants = {
        "attn_mean_all": att.mean(dim=(0, 1))[q_idx, :prompt_len].mean(0).cpu().numpy(),
        "attn_last_layer": att[-1].mean(dim=0)[q_idx, :prompt_len].mean(0).cpu().numpy(),
    }

    tok_lists = token_spans_for_sentences(
        tokenizer, prompt, context, sentences, context_offset
    )
    scored = {}
    for name, paid in variants.items():
        scores = np.zeros(len(sentences), dtype=np.float64)
        for i, idxs in enumerate(tok_lists):
            if idxs:
                scores[i] = float(paid[idxs].sum())
        if scores.sum() > 0:
            scores = scores / scores.sum()
        scored[name] = scores
    return scored


def parse_usefulness_scores(text: str, n: int) -> np.ndarray | None:
    try:
        m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
        if m:
            obj = json.loads(m.group(0))
            if isinstance(obj, list) and len(obj) == n:
                return np.array([float(x) for x in obj], dtype=np.float64)
            if isinstance(obj, dict):
                vals = []
                for i in range(1, n + 1):
                    found = None
                    for key in (str(i), f"[{i}]", f"sentence_{i}", f"s{i}"):
                        if key in obj:
                            found = float(obj[key])
                            break
                    if found is None:
                        break
                    vals.append(found)
                if len(vals) == n:
                    return np.array(vals, dtype=np.float64)
    except Exception:
        pass
    nums = re.findall(
        r"(?:sentence\s*)?\[?(\d+)\]?\s*[:=\-]\s*([0-9]*\.?[0-9]+)", text, flags=re.I
    )
    if len(nums) >= n:
        by_idx = {}
        for idx, val in nums:
            i = int(idx)
            if 1 <= i <= n and i not in by_idx:
                by_idx[i] = float(val)
        if len(by_idx) == n:
            return np.array([by_idx[i] for i in range(1, n + 1)], dtype=np.float64)
    return None


def llm_usefulness_scores(
    model, tokenizer, context: str, question: str, answer: str, n: int
) -> tuple[np.ndarray, str]:
    """
    Two-step LLM map:
      1) pick the single most necessary sentence number
      2) yes/no for each sentence → soft map
    Falls back gracefully if parsing fails.
    """
    # Step 1: single best sentence
    pick_user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Correct answer: {answer}\n\n"
        f"Which ONE context sentence number (1-{n}) most directly provides the answer? "
        f"Reply with only the number."
    )
    pick_prompt = chat_text(tokenizer, pick_user)
    pick_raw = generate(model, tokenizer, pick_prompt, max_new_tokens=8)
    pick_m = re.search(r"\b([1-9]|[1-9][0-9])\b", pick_raw)
    pick = int(pick_m.group(1)) if pick_m else None
    if pick is not None and not (1 <= pick <= n):
        pick = None

    # Step 2: per-sentence necessary? (yes/no)
    votes = np.zeros(n, dtype=np.float64)
    vote_logs = []
    for i in range(n):
        yn_user = (
            f"Context sentence: [{i+1}] {context.splitlines()[i].split('] ', 1)[-1]}\n"
            f"Question: {question}\n"
            f"Correct answer: {answer}\n\n"
            "Is this sentence necessary to know the answer? Reply yes or no."
        )
        yn_prompt = chat_text(tokenizer, yn_user)
        yn_raw = generate(model, tokenizer, yn_prompt, max_new_tokens=4).lower()
        vote_logs.append(f"{i+1}:{yn_raw}")
        if re.search(r"\byes\b", yn_raw) and not re.search(r"\bno\b", yn_raw):
            votes[i] = 1.0
        elif re.search(r"\byes\b", yn_raw):
            votes[i] = 0.5

    raw = f"pick={pick_raw.strip()} | votes={'; '.join(vote_logs)}"

    scores = votes.copy()
    if pick is not None:
        scores[pick - 1] += 2.0  # boost chosen sentence
    if scores.sum() == 0 and pick is not None:
        scores = np.zeros(n)
        scores[pick - 1] = 1.0
    if scores.sum() > 0:
        scores = scores / scores.sum()
    else:
        scores = np.full(n, np.nan)
    return scores, raw


def ablation_scores(
    model, tokenizer, sentences: list[str], question: str, gold_answer: str
) -> np.ndarray:
    """In-place redaction; usefulness = drop in gold-answer logprob."""

    def prompt_with(sents: list[str]) -> str:
        p, _, _ = qa_prompt(tokenizer, sents, question)
        return p

    base = prompt_with(sentences)
    base_lp = mean_answer_logprob(model, tokenizer, base, gold_answer)
    drops = []
    for i in range(len(sentences)):
        redacted = list(sentences)
        redacted[i] = "[REDACTED]"
        lp = mean_answer_logprob(
            model, tokenizer, prompt_with(redacted), gold_answer
        )
        drops.append(base_lp - lp)
    drops = np.array(drops, dtype=np.float64)
    # Keep signed signal for ranking; for bars use positive part normalized
    return drops


def normalize_positive(x: np.ndarray) -> np.ndarray:
    y = np.maximum(x, 0)
    if y.sum() > 0:
        return y / y.sum()
    # fallback: shift
    y = x - x.min()
    return y / y.sum() if y.sum() > 0 else y


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return None
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    return float(spearmanr(a, b).correlation)


def plot_example(ex_id: str, sentences: list[str], metrics: dict, key_sentence: int) -> Path:
    labels = [f"[{i+1}]" for i in range(len(sentences))]
    x = np.arange(len(sentences))
    width = 0.25
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7.2), gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax = axes[0]
    ax.bar(x - width, metrics["attention"], width, label="Attention", color="#1f77b4")
    ax.bar(x, metrics["llm"], width, label="LLM usefulness", color="#ff7f0e")
    ax.bar(
        x + width, metrics["ablation"], width, label="Ablation (causal)", color="#2ca02c"
    )
    ax.axvline(key_sentence, color="red", ls="--", lw=1, alpha=0.7, label="True key sent.")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Normalized score")
    ax.set_title(f"Importance signals — {ex_id}")
    ax.legend(loc="upper right", fontsize=8)
    llm_max = 0.0 if np.all(np.isnan(metrics["llm"])) else float(np.nanmax(metrics["llm"]))
    ymax = float(
        np.nanmax(
            [
                metrics["attention"].max(),
                llm_max,
                metrics["ablation"].max(),
            ]
        )
    )
    ax.set_ylim(0, max(0.01, ymax) * 1.3)

    ax2 = axes[1]
    ax2.axis("off")
    text = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences))
    text += (
        f"\n\nSpearman  attn↔llm={metrics['corr_attn_llm']}"
        f"  attn↔ablate={metrics['corr_attn_ablate']}"
        f"  llm↔ablate={metrics['corr_llm_ablate']}"
    )
    ax2.text(0, 1, text, va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    path = FIG_DIR / f"{ex_id}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_summary(rows: list[dict]) -> Path:
    names = [r["id"] for r in rows]
    x = np.arange(len(names))
    width = 0.25

    def vals(key: str) -> list[float]:
        return [r[key] if r[key] is not None else 0.0 for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, vals("corr_attn_llm"), width, label="Attention ↔ LLM")
    ax.bar(x, vals("corr_attn_ablate"), width, label="Attention ↔ Ablation")
    ax.bar(x + width, vals("corr_llm_ablate"), width, label="LLM ↔ Ablation")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Agreement between importance signals (synthetic facts)")
    ax.legend()
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    path = FIG_DIR / "summary_correlations.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def rank_of_key(scores: np.ndarray, key: int) -> int | None:
    if np.any(np.isnan(scores)):
        return None
    if np.allclose(scores, scores[0]):
        return None  # complete tie / refusal
    order = np.argsort(-scores)
    return int(np.where(order == key)[0][0] + 1)


def run():
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        attn_implementation="eager",
    ).to(DEVICE)
    model.eval()

    results = []
    for ex in EXAMPLES:
        print(f"\n=== Example: {ex['id']} ===")
        sentences = ex["sentences"]
        question = ex["question"]
        gold = ex["gold_answer"]
        key = ex["key_sentence"]

        prompt, context, offset = qa_prompt(tokenizer, sentences, question)
        free_answer = generate(model, tokenizer, prompt)
        print(f"Free answer: {free_answer}")
        print(f"Gold answer: {gold}")

        # Use gold answer for fair attention/ablation targeting
        attn_variants = attention_sentence_scores(
            model, tokenizer, prompt, gold, context, sentences, offset
        )
        attn = attn_variants["attn_mean_all"]
        print("Attention(mean):", np.round(attn, 3))
        print("Attention(last):", np.round(attn_variants["attn_last_layer"], 3))

        llm, raw_llm = llm_usefulness_scores(
            model, tokenizer, context, question, gold, len(sentences)
        )
        print("LLM usefulness:", np.round(llm, 3))
        print("LLM raw:", raw_llm.replace("\n", " ")[:220])

        ablate_raw = ablation_scores(model, tokenizer, sentences, question, gold)
        ablate = normalize_positive(ablate_raw)
        print("Ablation raw drops:", np.round(ablate_raw, 3))
        print("Ablation norm:", np.round(ablate, 3))

        c_al = corr(attn, llm)
        c_aa = corr(attn, ablate_raw)  # use signed drops for ranking
        c_la = corr(llm, ablate_raw)
        print(f"Spearman attn-llm={c_al} attn-ablate={c_aa} llm-ablate={c_la}")
        print(
            "Key-sentence rank (1=best):",
            f"attn={rank_of_key(attn, key)}",
            f"llm={rank_of_key(llm, key)}",
            f"ablate={rank_of_key(ablate_raw, key)}",
        )

        row = {
            "id": ex["id"],
            "question": question,
            "free_answer": free_answer,
            "gold_answer": gold,
            "key_sentence_1based": key + 1,
            "sentences": sentences,
            "attention_mean_all": attn.tolist(),
            "attention_last_layer": attn_variants["attn_last_layer"].tolist(),
            "llm_usefulness": llm.tolist(),
            "llm_raw": raw_llm,
            "ablation_logprob_drop": ablate_raw.tolist(),
            "ablation_norm": ablate.tolist(),
            "corr_attn_llm": c_al,
            "corr_attn_ablate": c_aa,
            "corr_llm_ablate": c_la,
            "rank_key_attn": rank_of_key(attn, key),
            "rank_key_llm": rank_of_key(llm, key),
            "rank_key_ablate": rank_of_key(ablate_raw, key),
            "free_answer_ok": gold.lower() in free_answer.lower(),
        }
        results.append(row)
        plot_example(
            ex["id"],
            sentences,
            {
                "attention": attn,
                "llm": llm,
                "ablation": ablate,
                "corr_attn_llm": c_al,
                "corr_attn_ablate": c_aa,
                "corr_llm_ablate": c_la,
            },
            key_sentence=key,
        )

    plot_summary(results)

    def mean_corr(key: str) -> float | None:
        vals = [r[key] for r in results if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    def mean_rank(key: str) -> float | None:
        vals = [r[key] for r in results if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "model": MODEL_ID,
        "design": "synthetic facts; gold-answer targeted attention/ablation; in-place redaction",
        "n_examples": len(results),
        "mean_spearman_attn_llm": mean_corr("corr_attn_llm"),
        "mean_spearman_attn_ablation": mean_corr("corr_attn_ablate"),
        "mean_spearman_llm_ablation": mean_corr("corr_llm_ablate"),
        "mean_rank_of_true_key_attn": mean_rank("rank_key_attn"),
        "mean_rank_of_true_key_llm": mean_rank("rank_key_llm"),
        "mean_rank_of_true_key_ablate": mean_rank("rank_key_ablate"),
        "examples": results,
    }
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    print("Mean Spearman:", summary["mean_spearman_attn_llm"], summary["mean_spearman_attn_ablation"], summary["mean_spearman_llm_ablation"])
    print(
        "Mean rank of true key sentence (1=best):",
        f"attn={summary['mean_rank_of_true_key_attn']}",
        f"llm={summary['mean_rank_of_true_key_llm']}",
        f"ablate={summary['mean_rank_of_true_key_ablate']}",
    )
    return summary


if __name__ == "__main__":
    run()
