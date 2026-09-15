# Hard delete vs soft attention-mask pruning — evaluation note

## Question
Is **soft attention masking** of dead/poison skill tokens (block only at answer queries) equivalent to **hard-deleting** those tokens from the skill string?

## Identification strategy
Synthetic skills with one **live** correct fact and one conflicting **poison** fact.

| Order | Causal structure | Expected if leakage is real |
|-------|------------------|-----------------------------|
| `poison_before_live` | live tokens can attend to poison during prefilling | soft answer-only mask ≈ still contaminated; **hard delete beats soft** |
| `poison_after_live` | live cannot attend to later poison (causal LM) | **hard ≈ soft** (placebo control) |

Paired across 8 items × 2 orders × 5 conditions. Primary metric: mean logprob of gold answer. Uncertainty: 2000-resample bootstrap 95% CIs. Sanity check: blocking live keys must hurt.

## Conditions
- `full` — poison present, normal attention
- `soft_answer_block` — poison present; answer queries ↛ poison keys
- `soft_all_block` — poison present; all queries ↛ poison keys
- `hard_delete` — poison removed from prompt text
- `sanity_block_live` — answer queries ↛ live keys (mask implementation check)

## How to run
```bash
python3 hard_vs_soft_rigorous.py
```

Artifacts: `artifacts/hard_vs_soft_rigorous.json`, `figures/hard_vs_soft_rigorous.png`, `figures/hard_vs_soft_contrasts.png`.
