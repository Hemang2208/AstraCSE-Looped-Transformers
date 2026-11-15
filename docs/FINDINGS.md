# AstraCSE: Findings

Status: 2026-11-10. All 19 queued training jobs are complete. The width-1,024 loop (`d1024_loop6x2_v2`) and the baseline second seed (`baseline_v2_s1`) are done. The matched v2 pair (0.05 nats reversal) is still single-seed.

## 1. Summary

**Update (6 Nov): the matched-parameter result reverses on v2, and warmup explains about a third of the v1 penalty.** On v2, Loop-6x2 (54.0M) is 0.052 nats better than Dense-6 (54.0M) on clean tokens (2.140 against 2.192); on v1 the same pair was 1.84 nats apart the other way. The v1 warmup ablation (`loop_6x2_wu171`: 171 instead of 46 warmup steps, nothing else changed) lowers the v1 loop loss from 4.692 to 4.074 clean, recovering 0.62 of the 1.84-nat gap; 1.22 nats remain, so schedule length, data mix or document order account for the rest. Single seed for all of these.

**Update (4 Nov): the v1 gap is a training-regime effect.** On corpus v2 the compute-matched gap (Loop-6x2 minus the 12-layer baseline) is 0.08 nats on clean tokens already at step 2,400, against 1.78 nats at the end of v1 (2,302 steps). It is 0.07 nats at the end of v2 and uniform across validation shards. So the 1.7 to 1.8 nat penalty below holds for the v1 regime only, and is not a matter of training for more steps. The regimes differ in warmup (46 against 171 steps), schedule length, data mix and document order; these are not separated yet. Details: `paper/main.tex`. The points below describe the v1 regime.

1. **Weight sharing causes a large validation-loss gap at matched parameters.** Each loop model is 1.7 to 1.8 nats worse than its dense control with the same number of unique layers. The gap holds for K = 3, 4 and 6.
2. **The gap is not explained by the number of parameters.** `dense_6l` has the same unique parameters as `loop_6x2`. It scores 2.80, better than the 12-layer baseline, while `loop_6x2` scores 4.62.
3. **The gap is not explained by the peak learning rate.** Halving the peak LR for `loop_6x2` changes validation loss by 0.03 nats (4.62 to 4.59).
4. **More sharing means more loss.** Among the loop models, loss rises as K falls: 4.62 (K = 6), 4.71 (K = 4), 4.95 (K = 3).
5. **At this data budget, a dense 6-layer model matches the 12-layer baseline.** `dense_6l` (54.0M parameters, half the layer applications) scores 2.80, and the baseline (96.5M) scores 2.85. This is consistent with the baseline being limited by tokens: at 1.15 billion tokens it sees about 12 tokens per parameter, while the 54M model sees about 21. This is a hypothesis. The v2 baseline run (4.3 billion tokens) is designed to test it.
6. **MCQ accuracy is near chance and does not separate the models** (confirmed on 4 Nov with three standard conditional-likelihood rules, for v1 and v2: see `docs/RESULTS.md` §2.1). All eight models score between 24.2% and 28.6% on 650 four-choice questions. The 95% interval is about +/-3.4 points, so the differences are within noise. Validation loss is the informative metric at this scale.

## 2. Evidence

### 2.1 The main contrast

| Comparison | Params | Compute per token (layer applications) | Val loss |
|---|---:|---:|---:|
| `dense_6l` | 54.0M | 6 | 2.795 |
| `loop_6x2` | 54.0M | 12 | 4.622 |
| `baseline` | 96.5M | 12 | 2.853 |

`loop_6x2` spends twice the compute of `dense_6l` per token, with the same unique parameters, and is much worse. The loop is also much worse than the baseline, which has the same compute and 1.8 times the parameters.

### 2.2 Matched-parameter gaps

| Unique layers K | Loop val loss | Dense control val loss | Gap (nats) |
|---:|---:|---:|---:|
| 6 | 4.622 | 2.795 | 1.83 |
| 4 | 4.706 | 2.998 | 1.71 |
| 3 | 4.947 | 3.198 | 1.75 |

The gap is roughly the same at each K. The loop models follow the same pattern as the dense models: the more layers, the lower the loss. The sharing itself costs about 1.75 nats at every K tested.

### 2.3 LR check

| Config | Peak LR | Val loss |
|---|---:|---:|
| `loop_6x2` | 6e-4 | 4.622 |
| `loop_6x2_lr3e-4` | 3e-4 | 4.592 |

The pre-set threshold for changing the LR was that 3e-4 must be clearly better. It was 0.03 nats better, so the LR was kept at 6e-4 for the queued loop runs.

## 3. Interpretation

**What is established so far (within this experiment):**
- Depth-wise weight sharing, at K = 3, 4 and 6 with 12 layer applications per token, produces a 1.7 to 1.8 nat loss penalty compared with dense models with the same unique parameters.
- The penalty does not come from having fewer unique parameters (`dense_6l` vs `loop_6x2`), and a single LR change of 2x does not remove it.

**What is not established:**
- Whether the penalty is specific to this model size, data budget (1.15B tokens, one epoch) or training length (2,304 steps). The v2 runs answer part of this: at 4.30B tokens and 8,595 steps the matched gap reverses to -0.05 nats (single seed). The warmup ablation (171 steps) recovers about a third of the v1 penalty. Width 1,024 is the best model so far.
- Whether the penalty is specific to the depth-slot conditioning used here (a 9,216-parameter embedding). Other conditioning schemes were not tested.
- Whether the penalty holds across seeds. Seed 1 gives a matched gap of 1.81 nats against 1.83 for seed 0, so it does, with two seeds at this scale.

**Mechanism (hypothesis).** A shared block must serve several depths with one set of weights. Only the additive depth embedding tells the block which depth it is at. The gradient on the shared weights is the sum over all depths that reuse it. The LR check argues against the gradient-scale part of this, but not against the basic capacity or conditioning cost of sharing.

## 4. Compute-matched comparison

Among the completed runs, the only compute-matched pair is the baseline (12 layer applications, 96.5M parameters) against `loop_6x2` (12 layer applications, 54.0M parameters). The loop has 56% of the parameters, uses the same compute, and scores 1.77 nats worse. This is the strongest single comparison for the paper, because it does not depend on the dense controls.

## 5. Threats to validity

- **Seeds.** v1 has a second seed for `baseline`, `dense_6l` and `loop_6x2`. Single-model losses move 0.03 to 0.12 nats between seeds, but the matched loop-minus-dense gap moves only 0.02. The v2 and width-1,024 runs are single-seed.
- **Low-data regime.** One epoch of 1.15 billion tokens in the first regime. The v2 regime (4.30 billion tokens, 8,595 steps) changes the matched-parameter result, so the v1 conclusion is specific to that regime.
- **Tuning.** The LR was set for the baseline and reused. The loop variants got one alternative LR, not a sweep.
- **MCQ metric.** The mean-loss scoring rule (see `docs/METHODS.md` §5) is weak. MCQ results are not used to rank models.
- **Benchmark coverage.** MMLU-CS covers six of the ten training subjects. The smoke test has ten items and cannot rank models.
- **Contamination (v1).** On 2 Nov, the v1 and v2 training shards were checked against the 650-item MMLU-CS set with the 13-gram rule: 0 of 650 items flagged in either (`docs/METHODS.md` §8). The validation split was carved out before packing, but a check on 2026-11-02 found verbatim copies across the split: 7.6% of validation tokens lie in copied runs of 100 or more tokens (`docs/METHODS.md` §8). Absolute validation losses are therefore optimistic. On the clean tokens (re-score, 3 Nov) they rise by 0.06 to 0.07 nats, and the loop-minus-dense gaps move by less than 0.01 nats. The v2 new documents show almost no overlap with validation.
- **Evaluation subset.** The validation metric uses every eighth batch. This is consistent across models, so comparisons are valid, but absolute values are not the full-set values.
- **Interruptions.** Spot reclaims stopped runs several times. Each run resumed from a saved checkpoint and completed. The final numbers come from completed runs.
- **In-training validation logs.** The logged in-training `val_loss` and `best_val_loss` come from a validation loader whose position carries over between evals, so they are not comparable across evals or runs (`docs/METHODS.md` §8). Only the official evaluation (final checkpoint, fresh loader, every 8th batch) is used for the results.

## 6. What would change the conclusions

- The second v1 seed moved the matched gap by only 0.02 nats, so the v1 gap is not in question at two seeds. A third seed would be needed to put an interval on the v2 reversal (-0.05).
- If `baseline_v2` (4.3 billion tokens) closes the gap with `loop6x2_v2` on the same data, the conclusion becomes a data-budget effect instead of a sharing effect.
- If `d1024_loop6x2_v2` shows a much smaller gap than the 1.7 to 1.8 nats seen at width 768, the gap is width-dependent. This run may be cut before it finishes, so the result is uncertain.
- A better MCQ metric (conditional answer likelihood) could show a real difference that the current metric hides.

## 7. Suggested framing for the paper

- Present this as a controlled negative result for depth-wise weight sharing, not as a general claim about "looped transformers".
- Report the compute-matched comparison (baseline vs `loop_6x2`) and the matched-parameter comparisons (each loop vs its dense control) side by side. State explicitly that the dense controls use half to a quarter of the layer applications.
- Report validation loss as the primary metric. Report MCQ with the 95% interval, and say it does not separate the models at this scale.
- Report seed variance (done for v1; v2 is single-seed and should say so).
- State the data budget (1.15 billion tokens, one epoch) and the 4.3-billion-token follow-up as the scale test.
