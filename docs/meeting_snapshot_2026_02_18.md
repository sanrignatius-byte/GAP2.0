# Meeting Snapshot (2026-02-18)

This file preserves a side-by-side snapshot of **pre-fix** and **post-fix**
Phase-2 probing results for tomorrow's meeting.

## Snapshot Source

- Pre-fix reference commit: `131dd49`
  - `probe_index_strategy=assistant_boundary_v2_teacher_forced`
  - `answer_token_source=teacher_forced_first_label_token`
- Post-fix reference: current `main` working tree after rerun
  - `probe_index_strategy=assistant_boundary_v2_prediction_position`
  - `answer_token_source=prediction_position_before_answer`

## Key Comparison (ChartQA yes/no, 48 valid samples)

### Text Instruction Token (post layers 24-47)

- Pre-fix:
  - Original: `0.7242`
  - Blind Black: `0.5984`
  - Blind Mean: `0.6324`
  - Blind Random: `0.5780`
- Post-fix:
  - Original: `0.7242`
  - Blind Black: `0.5984`
  - Blind Mean: `0.6324`
  - Blind Random: `0.5780`

Interpretation:
- Instruction-token conclusion is stable across the fix.
- Original minus Random remains strong (+0.1462 post-layer mean).

### Answer Token (post layers 24-47)

- Pre-fix:
  - Original: `1.0000`
  - Blind Black: `1.0000`
  - Blind Mean: `1.0000`
  - Blind Random: `1.0000`
- Post-fix:
  - Original: `0.6658`
  - Blind Black: `0.5309`
  - Blind Mean: `0.5769`
  - Blind Random: `0.7113`

Interpretation:
- Label leakage has been removed.
- Answer-token probe is now informative (no longer trivially perfect).

## Bootstrap Delta (Original - Blind Random)

From `results/text_probing_chartqa/phase2_probe_delta_orig_vs_random_bootstrap.json`:

- Text instruction token:
  - Pre-fix: `+0.1462`
  - Post-fix: `+0.1462`
  - 95% CI (post-fix): `[+0.1291, +0.1627]`
- Answer token:
  - Pre-fix: `0.0000`
  - Post-fix: `-0.0455`
  - 95% CI (post-fix): `[-0.0797, -0.0127]`

## Model Accuracy (Original vs Blind Random)

From `results/model_accuracy_chartqa_orig_vs_random/phase2_model_accuracy_results.json`:

- Original: `0.3125` (15/48)
- Blind Random: `0.2083` (10/48)
- Delta: `+0.1042`

This file is unchanged between the pre-fix and post-fix probing reruns.

