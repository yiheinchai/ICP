#!/usr/bin/env python3
"""
Reviewer-grade eval: hard skill deletion vs soft attention masking.

Claim
-----
Soft-masking poison keys only for *answer queries* ≠ deleting those tokens.
Under causal attention, earlier poison can contaminate later live skill tokens
during prefilling; answers can then read contaminated live keys without
attending to poison directly.

Design
------
Each item: 1 live correct fact, 1 conflicting poison fact, 3 fillers.

Orders:
  poison_before_live → live can attend to poison (leakage possible)
  poison_after_live  → live cannot attend to poison (causal control)

Conditions (paired per item×order):
  full, soft_answer_block, soft_all_block, hard_delete, sanity_block_live

Primary metric: mean logprob of gold answer tokens
Uncertainty: paired bootstrap 95% CIs (2000 resamples)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
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
NEG = torch.finfo(torch.float32).min
N_BOOT = 2000
RNG = np.random.default_rng(0)

ITEMS = [
    {
        "id": "zeldria_capital",
        "question": "What is the capital of Zeldria?",
        "gold": "Mirathen",
        "poison": "Caldorn",
        "live": "The capital of Zeldria is Mirathen.",
        "poison_chunk": "The capital of Zeldria is Caldorn.",
        "fillers": [
            "Zeldria's currency is called the vort.",
            "Mount Cindara is the tallest peak in Zeldria.",
            "Zeldrian coffee exports peak in spring.",
        ],
    },
    {
        "id": "brammel_boil",
        "question": "At sea level, what is the boiling point of Brammel in Celsius?",
        "gold": "87",
        "poison": "112",
        "live": "At sea level, Brammel boils at 87 degrees Celsius.",
        "poison_chunk": "At sea level, Brammel boils at 112 degrees Celsius.",
        "fillers": [
            "Brammel freezes at -12 degrees Celsius.",
            "Brammel is stored in sealed glass vials.",
            "Brammel was first isolated in 1911.",
        ],
    },
    {
        "id": "novara_author",
        "question": "Who wrote the novel Night of Novara?",
        "gold": "Kest Relm",
        "poison": "Lina Voss",
        "live": "The novel Night of Novara was written by Kest Relm.",
        "poison_chunk": "The novel Night of Novara was written by Lina Voss.",
        "fillers": [
            "Night of Novara is set on a desert moon.",
            "Kest Relm was born in South Oriel.",
            "Glass Harbor premiered in 2044.",
        ],
    },
    {
        "id": "orli_port",
        "question": "What port does the ORLI protocol listen on by default?",
        "gold": "7481",
        "poison": "2200",
        "live": "The ORLI protocol listens on port 7481 by default.",
        "poison_chunk": "The ORLI protocol listens on port 2200 by default.",
        "fillers": [
            "ORLI sessions expire after 15 minutes of idle time.",
            "ORLI handshake tokens are base64url encoded.",
            "ORLI was ratified by the Circinus Working Group.",
        ],
    },
    {
        "id": "helm_quota",
        "question": "What is Helmward's monthly API quota in requests?",
        "gold": "250000",
        "poison": "10000",
        "live": "Helmward accounts have a monthly API quota of 250000 requests.",
        "poison_chunk": "Helmward accounts have a monthly API quota of 10000 requests.",
        "fillers": [
            "Helmward retries use exponential backoff starting at 200ms.",
            "Helmward audit logs are retained for 90 days.",
            "Helmward supports mutual TLS on enterprise plans.",
        ],
    },
    {
        "id": "quevish_genders",
        "question": "How many grammatical genders does Quevish have?",
        "gold": "4",
        "poison": "2",
        "live": "Quevish has 4 grammatical genders.",
        "poison_chunk": "Quevish has 2 grammatical genders.",
        "fillers": [
            "Quevish verbs mark evidentiality.",
            "Quevish orthography uses a 28-letter alphabet.",
            "Quevish is spoken mainly in the river deltas of Nael.",
        ],
    },
    {
        "id": "synapse_timeout",
        "question": "What is the default SynapseX RPC timeout in milliseconds?",
        "gold": "3500",
        "poison": "800",
        "live": "SynapseX uses a default RPC timeout of 3500 milliseconds.",
        "poison_chunk": "SynapseX uses a default RPC timeout of 800 milliseconds.",
        "fillers": [
            "SynapseX heartbeats are sent every 10 seconds.",
            "SynapseX payloads are length-prefixed.",
            "SynapseX was open-sourced under Apache-2.0.",
        ],
    },
    {
        "id": "veld_vat",
        "question": "What is the standard VAT rate in Veldmark?",
        "gold": "19%",
        "poison": "7%",
        "live": "The standard VAT rate in Veldmark is 19%.",
        "poison_chunk": "The standard VAT rate in Veldmark is 7%.",
        "fillers": [
            "Veldmark fiscal years start on 1 April.",
            "Reduced VAT applies to books in Veldmark.",
            "Veldmark tax IDs begin with the letters VM.",
        ],
    },
]


@dataclass
class TrialResult:
    item_id: str
    order: str
    condition: str
    gold_logprob: float
    poison_logprob: float
    prefer_gold: bool  # forced choice under same mask: lp(gold) > lp(poison)
    answer: str
    correct: bool  # greedy generate (secondary; soft masks not applied at decode)
    poison_mentioned: bool
    n_poison_tokens: int
    n_live_tokens: int


def chat(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_chunks(item, order: str):
    f = item["fillers"]
    if order == "poison_before_live":
        chunks = [item["poison_chunk"], f[0], item["live"], f[1], f[2]]
        return chunks, 2, 0  # live_idx, poison_idx
    if order == "poison_after_live":
        chunks = [item["live"], f[0], item["poison_chunk"], f[1], f[2]]
        return chunks, 0, 2
    raise ValueError(order)


def format_skill(chunks):
    return "\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))


def make_prompt(tokenizer, chunks, question):
    skill = format_skill(chunks)
    user = (
        "Answer using ONLY the skill document below. "
        "Reply with a short phrase only.\n\n"
        f"Skill document:\n{skill}\n\nQuestion: {question}"
    )
    return chat(tokenizer, user), skill


def chunk_token_spans(tokenizer, prompt, skill, chunks):
    off = prompt.find(skill)
    if off < 0:
        raise RuntimeError("skill missing from prompt")
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    spans = []
    for i, c in enumerate(chunks):
        needle = f"[{i+1}] {c}"
        c0 = skill.find(needle)
        if c0 < 0:
            spans.append([])
            continue
        a0, a1 = off + c0, off + c0 + len(needle)
        spans.append(
            [
                ti
                for ti, (a, b) in enumerate(enc["offset_mapping"])
                if b > a0 and a < a1 and a != b
            ]
        )
    return spans


def causal_plus_blocks(seq_len, blocked_pairs):
    bias = torch.zeros(1, 1, seq_len, seq_len, device=DEVICE)
    for i in range(seq_len):
        bias[0, 0, i, i + 1 :] = NEG
    for q, k in blocked_pairs:
        if 0 <= k <= q < seq_len:
            bias[0, 0, q, k] = NEG
    return bias


def mean_gold_logprob(model, tokenizer, prompt, gold, attn4d=None):
    p = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        DEVICE
    )
    a = tokenizer(
        " " + gold, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(DEVICE)
    full = torch.cat([p, a], dim=1)
    am = attn4d if attn4d is not None else torch.ones_like(full)
    with torch.no_grad():
        logits = model(full, attention_mask=am).logits
    plen = p.shape[1]
    lps = []
    for i in range(a.shape[1]):
        lp = torch.log_softmax(logits[0, plen + i - 1], dim=-1)
        lps.append(lp[a[0, i]].item())
    return float(np.mean(lps))


def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def is_correct(answer, gold):
    return norm(gold) in norm(answer)


def mentions_poison(answer, poison):
    return norm(poison) in norm(answer)


def build_mask(mode, seq_len, prompt_len, poison_toks, live_toks):
    if mode in ("full", "hard_delete"):
        return None
    pairs = []
    answer_qs = list(range(prompt_len, seq_len))
    if mode == "soft_answer_block":
        for q in answer_qs:
            for k in poison_toks:
                pairs.append((q, k))
    elif mode == "soft_all_block":
        for q in range(seq_len):
            for k in poison_toks:
                pairs.append((q, k))
    elif mode == "sanity_block_live":
        for q in answer_qs:
            for k in live_toks:
                pairs.append((q, k))
    else:
        raise ValueError(mode)
    return causal_plus_blocks(seq_len, pairs)


def _mask_for_answer_len(condition, plen, answer, tokenizer, poison_toks, live_toks):
    alen = tokenizer(
        " " + answer, return_tensors="pt", add_special_tokens=False
    )["input_ids"].shape[1]
    return build_mask(condition, plen + alen, plen, poison_toks, live_toks), alen


def run_trial(model, tokenizer, item, order, condition) -> TrialResult:
    chunks, live_i, poison_i = build_chunks(item, order)

    if condition == "hard_delete":
        kept = [c for j, c in enumerate(chunks) if j != poison_i]
        prompt, _ = make_prompt(tokenizer, kept, item["question"])
        lp_g = mean_gold_logprob(model, tokenizer, prompt, item["gold"], None)
        lp_p = mean_gold_logprob(model, tokenizer, prompt, item["poison"], None)
        ans = generate_answer(model, tokenizer, prompt)
        return TrialResult(
            item["id"],
            order,
            condition,
            lp_g,
            lp_p,
            lp_g > lp_p,
            ans,
            is_correct(ans, item["gold"]),
            mentions_poison(ans, item["poison"]),
            0,
            -1,
        )

    prompt, skill = make_prompt(tokenizer, chunks, item["question"])
    spans = chunk_token_spans(tokenizer, prompt, skill, chunks)
    poison_toks, live_toks = spans[poison_i], spans[live_i]
    plen = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].shape[1]

    attn_g, _ = _mask_for_answer_len(
        condition, plen, item["gold"], tokenizer, poison_toks, live_toks
    )
    attn_p, _ = _mask_for_answer_len(
        condition, plen, item["poison"], tokenizer, poison_toks, live_toks
    )
    lp_g = mean_gold_logprob(model, tokenizer, prompt, item["gold"], attn_g)
    lp_p = mean_gold_logprob(model, tokenizer, prompt, item["poison"], attn_p)
    # Secondary: unmasked greedy generate (custom 4D mask not supported by generate)
    ans = generate_answer(model, tokenizer, prompt)
    return TrialResult(
        item["id"],
        order,
        condition,
        lp_g,
        lp_p,
        lp_g > lp_p,
        ans,
        is_correct(ans, item["gold"]),
        mentions_poison(ans, item["poison"]),
        len(poison_toks),
        len(live_toks),
    )


def bootstrap_mean_ci(x, n=N_BOOT, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), (float("nan"), float("nan"))
    boots = [
        RNG.choice(x, size=len(x), replace=True).mean() for _ in range(n)
    ]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), (float(lo), float(hi))


def summarize(trials):
    rows = []
    keys = sorted({(t.order, t.condition) for t in trials})
    for order, cond in keys:
        ts = [t for t in trials if t.order == order and t.condition == cond]
        lps = np.array([t.gold_logprob for t in ts])
        margin = np.array([t.gold_logprob - t.poison_logprob for t in ts])
        pref = np.array([float(t.prefer_gold) for t in ts])
        acc = np.array([float(t.correct) for t in ts])
        poi = np.array([float(t.poison_mentioned) for t in ts])
        lp_m, lp_ci = bootstrap_mean_ci(lps)
        mar_m, mar_ci = bootstrap_mean_ci(margin)
        pref_m, pref_ci = bootstrap_mean_ci(pref)
        acc_m, acc_ci = bootstrap_mean_ci(acc)
        poi_m, poi_ci = bootstrap_mean_ci(poi)
        rows.append(
            {
                "order": order,
                "condition": cond,
                "n": len(ts),
                "mean_gold_logprob": lp_m,
                "gold_logprob_ci95": list(lp_ci),
                "mean_margin_gold_minus_poison": mar_m,
                "margin_ci95": list(mar_ci),
                "prefer_gold_rate": pref_m,
                "prefer_gold_ci95": list(pref_ci),
                "greedy_accuracy": acc_m,
                "greedy_accuracy_ci95": list(acc_ci),
                "poison_mention_rate": poi_m,
                "poison_mention_ci95": list(poi_ci),
            }
        )
    return rows


def paired_contrasts(trials):
    specs = [
        ("soft_answer_block", "full", "leakage_proxy_soft_minus_full"),
        ("hard_delete", "soft_answer_block", "hard_minus_soft"),
        ("hard_delete", "full", "hard_minus_full"),
        ("soft_all_block", "soft_answer_block", "allblock_minus_answerblock"),
        ("sanity_block_live", "full", "sanity_liveblock_minus_full"),
    ]
    out = []
    items = sorted({t.item_id for t in trials})
    for order in ("poison_before_live", "poison_after_live"):
        tab = {}
        for t in trials:
            if t.order == order:
                tab.setdefault(t.item_id, {})[t.condition] = t
        for a, b, name in specs:
            if not all(a in tab[i] and b in tab[i] for i in items):
                continue
            lp_a = np.array([tab[i][a].gold_logprob for i in items])
            lp_b = np.array([tab[i][b].gold_logprob for i in items])
            mar_a = np.array(
                [tab[i][a].gold_logprob - tab[i][a].poison_logprob for i in items]
            )
            mar_b = np.array(
                [tab[i][b].gold_logprob - tab[i][b].poison_logprob for i in items]
            )
            mean_d, ci = bootstrap_mean_ci(lp_a - lp_b)
            mean_md, mci = bootstrap_mean_ci(mar_a - mar_b)
            out.append(
                {
                    "order": order,
                    "contrast": name,
                    "a": a,
                    "b": b,
                    "mean_delta_logprob_a_minus_b": mean_d,
                    "delta_ci95": list(ci),
                    "mean_delta_margin_a_minus_b": mean_md,
                    "margin_delta_ci95": list(mci),
                    "frac_items_a_gt_b": float(np.mean(lp_a > lp_b + 1e-9)),
                    "n_items": len(items),
                    "ci_excludes_zero": not (ci[0] <= 0 <= ci[1]),
                    "margin_ci_excludes_zero": not (mci[0] <= 0 <= mci[1]),
                }
            )
    return out


def plot_results(summary_rows, contrasts, path: Path):
    orders = ["poison_before_live", "poison_after_live"]
    conds = [
        "full",
        "soft_answer_block",
        "soft_all_block",
        "hard_delete",
        "sanity_block_live",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, order in zip(axes, orders):
        xs, ys, ylo, yhi = [], [], [], []
        labels = []
        for i, c in enumerate(conds):
            row = next(
                (r for r in summary_rows if r["order"] == order and r["condition"] == c),
                None,
            )
            if not row:
                continue
            xs.append(i)
            ys.append(row["mean_gold_logprob"])
            ylo.append(row["gold_logprob_ci95"][0])
            yhi.append(row["gold_logprob_ci95"][1])
            labels.append(c)
        yerr = np.vstack([np.array(ys) - np.array(ylo), np.array(yhi) - np.array(ys)])
        ax.bar(xs, ys, color="#4c72b0", alpha=0.9)
        ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor="black", capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(order.replace("_", " "))
        ax.set_ylabel("Mean gold logprob")
    fig.suptitle(
        "Hard delete vs soft attention mask (8 items, 95% bootstrap CI)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(10, 4.8))
    labels, means, cis = [], [], []
    for c in contrasts:
        if c["contrast"] == "sanity_liveblock_minus_full":
            continue
        labels.append(f"{c['order'].split('_')[1]}|{c['contrast']}")
        means.append(c["mean_delta_logprob_a_minus_b"])
        cis.append(c["delta_ci95"])
    y = np.arange(len(labels))
    ax.axvline(0, color="black", lw=0.8)
    for i, (m, ci) in enumerate(zip(means, cis)):
        color = "#2ca02c" if (ci[0] > 0 or ci[1] < 0) else "#7f7f7f"
        ax.errorbar(
            m,
            i,
            xerr=[[m - ci[0]], [ci[1] - m]],
            fmt="o",
            color=color,
            capsize=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Δ gold logprob (a − b), 95% CI")
    ax.set_title("Paired contrasts (green = CI excludes 0)")
    fig2.tight_layout()
    fig2.savefig(path.with_name("hard_vs_soft_contrasts.png"), dpi=150)
    plt.close(fig2)


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    conditions = [
        "full",
        "soft_answer_block",
        "soft_all_block",
        "hard_delete",
        "sanity_block_live",
    ]
    orders = ["poison_before_live", "poison_after_live"]

    trials = []
    total = len(ITEMS) * len(orders) * len(conditions)
    done = 0
    for item in ITEMS:
        for order in orders:
            for cond in conditions:
                tr = run_trial(model, tokenizer, item, order, cond)
                trials.append(tr)
                done += 1
                print(
                    f"[{done}/{total}] {item['id']}|{order}|{cond}: "
                    f"lpG={tr.gold_logprob:.3f} lpP={tr.poison_logprob:.3f} "
                    f"prefG={tr.prefer_gold} ok={tr.correct} ans={tr.answer!r}"
                )

    summary_rows = summarize(trials)
    contrasts = paired_contrasts(trials)

    def get_c(order, name):
        return next(
            c for c in contrasts if c["order"] == order and c["contrast"] == name
        )

    items = sorted({t.item_id for t in trials})

    def per_item_delta(order, ca, cb):
        tab = {}
        for t in trials:
            if t.order == order:
                tab.setdefault(t.item_id, {})[t.condition] = t.gold_logprob
        return np.array([tab[i][ca] - tab[i][cb] for i in items])

    d_before = per_item_delta(
        "poison_before_live", "hard_delete", "soft_answer_block"
    )
    d_after = per_item_delta(
        "poison_after_live", "hard_delete", "soft_answer_block"
    )
    inter = d_before - d_after
    inter_m, inter_ci = bootstrap_mean_ci(inter)

    hyp = {
        "H0_sanity_mask_hurts": {
            **{
                k: get_c("poison_before_live", "sanity_liveblock_minus_full")[k]
                for k in (
                    "mean_delta_logprob_a_minus_b",
                    "delta_ci95",
                    "ci_excludes_zero",
                )
            },
            "pass": get_c("poison_before_live", "sanity_liveblock_minus_full")[
                "mean_delta_logprob_a_minus_b"
            ]
            < 0
            and get_c("poison_before_live", "sanity_liveblock_minus_full")["delta_ci95"][
                1
            ]
            < 0,
        },
        "H1_before_soft_approx_full": {
            **{
                k: get_c("poison_before_live", "leakage_proxy_soft_minus_full")[k]
                for k in (
                    "mean_delta_logprob_a_minus_b",
                    "delta_ci95",
                    "ci_excludes_zero",
                )
            },
            "pass_approx": (
                not get_c("poison_before_live", "leakage_proxy_soft_minus_full")[
                    "ci_excludes_zero"
                ]
            )
            or abs(
                get_c("poison_before_live", "leakage_proxy_soft_minus_full")[
                    "mean_delta_logprob_a_minus_b"
                ]
            )
            < 0.05,
        },
        "H2_before_hard_beats_soft": {
            **{
                k: get_c("poison_before_live", "hard_minus_soft")[k]
                for k in (
                    "mean_delta_logprob_a_minus_b",
                    "delta_ci95",
                    "ci_excludes_zero",
                    "frac_items_a_gt_b",
                )
            },
            "pass": get_c("poison_before_live", "hard_minus_soft")[
                "mean_delta_logprob_a_minus_b"
            ]
            > 0
            and get_c("poison_before_live", "hard_minus_soft")["delta_ci95"][0] > 0,
        },
        "H3_after_hard_approx_soft": {
            **{
                k: get_c("poison_after_live", "hard_minus_soft")[k]
                for k in (
                    "mean_delta_logprob_a_minus_b",
                    "delta_ci95",
                    "ci_excludes_zero",
                )
            },
            "pass_approx": (
                not get_c("poison_after_live", "hard_minus_soft")["ci_excludes_zero"]
            )
            or abs(
                get_c("poison_after_live", "hard_minus_soft")[
                    "mean_delta_logprob_a_minus_b"
                ]
            )
            < 0.1,
        },
        "H4_interaction_leakage_path": {
            "mean_interaction_before_minus_after": inter_m,
            "ci95": list(inter_ci),
            "pass": inter_m > 0 and inter_ci[0] > 0,
            "interpretation": (
                "E[hard-soft | poison_before] - E[hard-soft | poison_after] > 0 "
                "supports that soft≈full leakage depends on live←poison access"
            ),
        },
    }

    plot_results(summary_rows, contrasts, FIG / "hard_vs_soft_rigorous.png")

    payload = {
        "model": MODEL_ID,
        "design": {
            "n_items": len(ITEMS),
            "orders": orders,
            "conditions": conditions,
            "primary_metric": "mean logprob of gold answer tokens (paired)",
            "secondary_metrics": [
                "greedy accuracy (unmasked generate; secondary only)",
                "poison mention rate",
            ],
            "uncertainty": f"{N_BOOT}-resample bootstrap 95% CIs",
            "identification_strategy": (
                "Compare poison_before_live vs poison_after_live. "
                "Leakage from soft answer-only masking requires live attending "
                "to earlier poison; after_live is the causal placebo control."
            ),
            "limitations": [
                "Single small model (0.5B) on CPU",
                "Greedy accuracy for soft masks does not apply custom 4D mask at decode",
                "Synthetic facts; not production company docs",
            ],
        },
        "hypotheses": hyp,
        "summary": summary_rows,
        "contrasts": contrasts,
        "trials": [asdict(t) for t in trials],
    }
    out_path = OUT / "hard_vs_soft_rigorous.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n=== Hypothesis outcomes ===")
    for k, v in hyp.items():
        print(f"{k}: {json.dumps({kk: vv for kk, vv in v.items() if kk != 'detail'})}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
