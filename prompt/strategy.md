# Strategy

## Evidence So Far

- The stable `20260512` checkpoint family remains the strongest platform family.
- Inference-only `seq_c` ablation improved AUC from `0.820768` to `0.820776`, but the optimum is shallow.
- Recent-crop training and focal hard-focus training did not earn evaluation priority because local validation stayed below the gate or ranking decayed.
- Runtime is acceptable for BS128 safe inference; AUC should dominate decisions.

## Modeling Direction

Prefer small, evidence-backed changes:

- ranking-aligned objective: BCE plus low-weight BPR
- mild sequence-domain robustness: `seq_c` dropout around `0.06`
- target-aware domain summaries
- time-decay pooling from existing time fields
- small late unified block only after the above ablations

Avoid large bundled changes:

- no full HSTU/OneTrans rewrite until smaller target/time changes beat the local gate
- no bigger hidden size or more blocks in the same run as objective changes
- no platform eval for locally weak checkpoints

## Promotion Criteria

- Platform promotion requires AUC greater than current best `0.820776`.
- Training checkpoint evaluation requires local validation around `0.86632` or stronger, plus healthy hist bucket behavior.
- Treat logloss-only improvement as insufficient; the competition score is ranking AUC.

## Experiment Discipline

- One conceptual change per training job when possible.
- Record package path, checkpoint path, validation AUC/logloss, hist bucket AUCs, platform AUC, inference time, and promotion decision.
- Keep training and evaluation packages clean: no `__pycache__`, no temporary inspection folders.
