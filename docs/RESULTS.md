# AstraCSE: Results

Status: 2026-11-10. All 19 queued training jobs are complete. Section 6 lists the last two runs (both finished and evaluated).

**Validation caveat (2026-11-02).** The validation shards share verbatim text with the v1 training shards: 7.6% of validation tokens lie in copied runs of 100 or more tokens. Every number in this file is measured on that validation set, so the absolute values are optimistic (see `docs/METHODS.md` §8 and `docs/FINDINGS.md` §5). The MMLU-CS check found no overlap with the v1 or v2 training shards (0 of 650 items flagged).

**Clean-token re-score (3 Nov).** Each completed checkpoint was re-scored on the 92.4% of validation tokens that are not inside a copied passage of 100 or more tokens. The full-set values reproduce the reported numbers for all nine runs. The clean values are 0.06 to 0.07 nats higher, and the loop-minus-dense gaps move by less than 0.01 nats, so the main finding stands.

| Run | Full set (reported) | Clean tokens (92.4%) | Copied tokens (7.6%) |
|---|---:|---:|---:|
| `baseline` | 2.8532 | 2.9142 | 2.0564 |
| `dense_6l` | 2.7953 | 2.8556 | 2.0114 |
| `dense_4l` | 2.9976 | 3.0583 | 2.2121 |
| `dense_3l` | 3.1979 | 3.2587 | 2.4111 |
| `loop_6x2` | 4.6225 | 4.6918 | 3.7172 |
| `loop_6x2_lr3e-4` | 4.5922 | 4.6603 | 3.7088 |
| `loop_4x3` | 4.7057 | 4.7740 | 3.8165 |
| `loop_3x4` | 4.9469 | 5.0099 | 4.1203 |
| `loop_6x2_wu171` | 4.0061 | 4.0737 | 3.1192 |
| `baseline_s1` | 2.8868 | 2.9483 | 2.0898 |
| `dense_6l_s1` | 2.6970 | 2.7573 | 1.9086 |
| `loop_6x2_s1` | 4.5024 | 4.5713 | 3.5986 |
| `baseline_v2` (v2 corpus) | 2.0145 | 2.0722 | 1.2604 |
| `loop6x2_v2` (v2 corpus) | 2.0837 | 2.1403 | 1.3447 |
| `dense_6l_v2` (v2 corpus, 54.0M, matched-parameter control) | 2.1353 | 2.1923 | 1.3922 |
| `loop_6x2_wu171` (v1, warmup ablation: 171 instead of 46 warmup steps) | 4.0061 | 4.0737 | 3.1192 |
| `d1024_v2` (v2 corpus, 164.8M) | 1.9274 | 1.9863 | 1.1596 |
| `d1024_ctx2048` (+400 steps at 2,048 context, evaluated at 1,024) | 1.9240 | 1.9831 | 1.1533 |

**v2 training curve (4 Nov).** Periodic checkpoints of the two v2 runs, scored with the official protocol (clean tokens):

| Step | `baseline_v2` | `loop6x2_v2` | Gap |
|---:|---:|---:|---:|
| 400 | 4.451 | 5.212 | 0.762 |
| 1,200 | 2.694 | 2.807 | 0.113 |
| 2,400 | 2.479 | 2.560 | 0.081 |
| 4,000 | 2.278 | 2.345 | 0.067 |
| 6,000 | 2.141 | 2.204 | 0.063 |
| 8,000 | 2.078 | 2.145 | 0.067 |
| 8,595 (final) | 2.072 | 2.140 | 0.068 |
| v1 end (2,302), for comparison | 2.914 | 4.692 | 1.778 |

At a similar step count the v2 gap is 0.08 nats and the v1 gap is 1.78, so the v1 penalty is not explained by the number of steps (see `docs/FINDINGS.md`, update of 4 Nov).

Matched-parameter gaps, full set and clean tokens: K=6 1.83 and 1.84; K=4 1.71 and 1.72; K=3 1.75 and 1.75. The v1-to-v2 gain for the baseline is 0.84 nats on both measures. `loop6x2_v2` was re-scored in a second one-off pause on 3 Nov: clean 2.1403 against baseline_v2 clean 2.0722, a gap of 0.07 nats, the same as on the full set (0.07). Per shard and per position decile, the gap stays between +0.06 and +0.08 nats (`scripts/overlap/results/subject_check.json`), so it is not driven by one part of the validation text.

Raw numbers: `paper/data/results/<run>.mmlu_cs.json` (validation loss, perplexity, MCQ-650 with per-subject counts) and `paper/data/results/<run>.benchmark.json` (10-item smoke test). Method details: `docs/METHODS.md`. Interpretation: `docs/FINDINGS.md`.

All runs use the v1 corpus (1,151,073,280 tokens, one epoch = 2,302 steps), the same tokenizer, and the same evaluation protocol.

## 1. Final results

| Run | Config | Params | Layers (unique x loops) | Peak LR | Val loss | Perplexity | MCQ-650 correct | Benchmark (10) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `baseline` | `configs/model.yaml` | 96,473,856 | 12 x 1 | 6e-4 | 2.8532 | 17.34 | 26.9% (175/650) | 3/10 |
| `dense_6l` | `configs/model_dense_6l.yaml` | 53,997,312 | 6 x 1 | 6e-4 | **2.7953** | 16.37 | 25.5% (166/650) | 3/10 |
| `dense_4l` | `configs/model_dense_4l.yaml` | 39,838,464 | 4 x 1 | 6e-4 | 2.9976 | 20.04 | 26.3% (171/650) | 3/10 |
| `dense_3l` | `configs/model_dense_3l.yaml` | 32,759,040 | 3 x 1 | 6e-4 | 3.1979 | 24.48 | 27.5% (179/650) | 4/10 |
| `loop_6x2` | `configs/model_loop_6x2.yaml` | 54,006,528 | 6 x 2 | 6e-4 | 4.6225 | 101.75 | 28.6% (186/650) | 2/10 |
| `loop_6x2_lr3e-4` | `configs/model_loop_6x2_lr3e-4.yaml` | 54,006,528 | 6 x 2 | 3e-4 | 4.5922 | 98.71 | 24.2% (157/650) | 2/10 |
| `loop_4x3` | `configs/model_loop_4x3.yaml` | 39,847,680 | 4 x 3 | 6e-4 | 4.7057 | 110.57 | 26.6% (173/650) | 1/10 |
| `loop_3x4` | `configs/model_loop_3x4.yaml` | 32,768,256 | 3 x 4 | 6e-4 | 4.9469 | 140.73 | 27.8% (181/650) | 4/10 |

Notes:
- "Layers (unique x loops)": a dense model has one application per layer. A loop model has `K` unique blocks applied `T` times, so it runs `K x T` layer applications per token. Every loop model runs 12 applications, the same as the baseline.
- Each config is trained for 2,304 steps as configured. The v1 epoch ends at step 2,302, and the run stops there.
- Validation loss is the mean token cross-entropy on the strided validation subset (every 8th batch). Perplexity is `exp(loss)`.

## 2. Per-subject MMLU-CS (correct / total)

The MMLU-CS set has 650 questions across six subjects. The columns are the eight v1 runs. Later runs are in section 2.1 and section 6.

| Subject (total) | baseline | dense_6l | dense_4l | dense_3l | loop_6x2 | loop_6x2_lr3e-4 | loop_4x3 | loop_3x4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| computer_security (100) | 23 | 20 | 19 | 22 | 25 | 18 | 22 | 29 |
| college_computer_science (95) | 25 | 28 | 24 | 25 | 27 | 26 | 24 | 33 |
| high_school_computer_science (100) | 23 | 23 | 28 | 25 | 26 | 23 | 24 | 21 |
| machine_learning (110) | 28 | 20 | 19 | 24 | 31 | 18 | 32 | 21 |
| electrical_engineering (145) | 47 | 44 | 52 | 56 | 47 | 42 | 45 | 48 |
| abstract_algebra (100) | 29 | 31 | 29 | 27 | 30 | 30 | 26 | 29 |
| **Total (650)** | **175** | **166** | **171** | **179** | **186** | **157** | **173** | **181** |

Chance level is 25% (four choices). No configuration is clearly above chance on any subject; see the statistical notes below.

### 2.1 MMLU-CS with conditional-likelihood scoring (4 Nov)

Same 650 questions, scored by `scripts/overlap/mcq_cond_box.py` with rules fixed before any result was seen. CF-norm (primary) = cloze log-likelihood of the choice text given "Question: ... Answer:", per character; CF-sum = the same in total; MCF = log-likelihood of the answer letter with the four choices listed. "A share" = how often MCF answers A.

| Run | Old rule | CF-norm | CF-sum | MCF | MCF A share |
|---|---:|---:|---:|---:|---:|
| `baseline` | 26.9% | 26.9% | 24.3% | 26.3% | 91% |
| `dense_6l` | 25.5% | 24.3% | 23.2% | 26.8% | 86% |
| `dense_4l` | 26.3% | 27.2% | 22.6% | 25.4% | 82% |
| `dense_3l` | 27.5% | 23.8% | 23.8% | 26.3% | 98% |
| `loop_6x2` | 28.6% | 26.3% | 24.3% | 26.2% | 100% |
| `loop_6x2_lr3e-4` | 24.2% | 23.7% | 24.6% | 26.0% | 100% |
| `loop_4x3` | 26.6% | 22.9% | 25.5% | 25.8% | 100% |
| `loop_3x4` | 27.8% | 24.8% | 24.8% | 26.0% | 100% |
| `baseline_v2` | 25.7% | 27.4% | 24.6% | 25.2% | 91% |
| `loop6x2_v2` | 27.5% | 26.8% | 24.2% | 25.4% | 94% |
| `dense_6l_v2` | 28.2% | 25.5% | 25.8% | 26.2% | 89% |
| `loop_6x2_wu171` | 25.4% | 24.8% | 25.8% | 25.7% | 98% |
| `d1024_v2` | 27.7% | 26.0% | 22.3% | 25.1% | 90% |
| `d1024_ctx2048` | 28.5% | 26.2% | 23.1% | 24.9% | 92% |
| `d1024_loop6x2_v2` | 26.5% | 26.5% | 24.9% | 25.5% | 94% |
| `baseline_v2_s1` | 28.3% | 27.8% | 25.8% | 25.4% | 98% |

Every model is at chance under every rule. The answer key is unbalanced (always C = 28.8%, always A = 26.0%). Paired on the same questions, v1 and v2 do not differ (McNemar chi-squared 0.03 for both the baseline and Loop-6x2), although validation loss improved by 0.84 and 2.55 nats. MMLU-CS stays in the evaluation as the domain's standard benchmark.

## 3. Smoke-test benchmark (10 questions)

`eval/benchmark.json` has one question per subject, written by hand. It is too small to rank models (one question is 10 percentage points). It is kept only as a sanity check. Correct answers: baseline 3, dense_6l 3, dense_4l 3, dense_3l 4, loop_6x2 2, loop_6x2_lr3e-4 2, loop_4x3 1, loop_3x4 4.

## 4. Matched-parameter comparison (validation loss)

Each loop model is paired with the dense model that has the same number of unique layers and no sharing. The unique parameter counts match except for the loop's depth-conditioning embedding (`n_layer x d_model` = 9,216 parameters).

| Pair | Loop (sharing) | Params | Val loss | Dense control | Params | Val loss | Gap (nats) |
|---|---|---:|---:|---|---:|---:|---:|
| K = 6 | `loop_6x2` (6 x 2) | 54.01M | 4.622 | `dense_6l` (6 x 1) | 54.00M | 2.795 | 1.83 |
| K = 4 | `loop_4x3` (4 x 3) | 39.85M | 4.706 | `dense_4l` (4 x 1) | 39.84M | 2.998 | 1.71 |
| K = 3 | `loop_3x4` (3 x 4) | 32.77M | 4.947 | `dense_3l` (3 x 1) | 32.76M | 3.198 | 1.75 |

The dense controls are matched on parameters, not on compute. A dense control applies each layer once, so it runs `K` layer applications per token. Each loop runs `K x T = 12`. The compute-matched comparison is the baseline (12 applications) against the loop variants (12 applications):

| Compute-matched pair | Params | Val loss |
|---|---:|---:|
| `baseline` (12 x 1) | 96.47M | 2.853 |
| `loop_6x2` (6 x 2) | 54.01M | 4.622 |

## 5. Statistical notes

- **MCQ-650.** At an accuracy near 27%, the standard error for 650 questions is about 1.7 percentage points, so the 95% interval is about +/-3.4 points. A difference between two configurations needs about 4.8 points to be significant. All MCQ differences in this table are inside that range. The MCQ column does not rank the models.
- **Validation loss.** Second seeds are done for `baseline`, `dense_6l` and `loop_6x2` (section 1). A second seed moved the single-model losses by 0.03 nats (dense-12), 0.10 (dense-6) and 0.12 (loop-6x2). The loop-minus-dense gap moved by only 0.02 nats (1.83 and 1.81), but the compute-matched gap moved from 1.78 to 1.62. The v2 runs are still single-seed.
- **In-training numbers are not comparable.** In-training validation values were logged from a 50-batch unstrided prefix. They are for monitoring only. Use the table in section 1 for every reported number.

## 6. Later runs (v2, width 1,024, seeds)

Status at 2026-11-07, about 12:00 IST. MCQ values are the old mean-loss rule (section 1); the conditional-likelihood values are in section 2.1.

| Job | Config | Corpus | Status |
|---|---|---|---|
| `baseline_v2` | `configs/model.yaml` | v2 (4.30B tokens, 8,595 steps) | Done 2 Nov 06:43 IST. Val 2.0145 (clean 2.0722), perplexity 7.50, MCQ-650 167/650 (25.7%), smoke 2/10. One earlier disk-full failure at step 5,087 (incident 11), resumed from its checkpoint |
| `loop6x2_v2` | `configs/model_loop_6x2.yaml` | v2 | Done 3 Nov 05:33 IST. Val 2.0837 (clean 2.1403), perplexity 8.03, MCQ-650 179/650 (27.5%) |
| `dense_6l_v2` | `configs/model_dense_6l.yaml` (54.0M, matched-parameter control) | v2 | Done 5 Nov 18:29 IST. Val 2.1353 (clean 2.1923), perplexity 8.46, MCQ-650 183/650 (28.2%) |
| `d1024_v2` | `configs/model_d1024.yaml` (164.8M) | v2 | Done 5 Nov 00:45 IST. Val 1.9274 (clean 1.9863), perplexity 6.87, MCQ-650 180/650 (27.7%) |
| `d1024_ctx2048` | `configs/model_d1024_ctx2048.yaml` (2,048 context, 400 steps from `d1024_v2`) | v2 | Done 5 Nov 02:49 IST. Val 1.9240 (clean 1.9831) at 1,024 tokens, perplexity 6.85, MCQ-650 185/650 (28.5%) |
| `loop_6x2_wu171` | `configs/model_loop_6x2_wu171.yaml` (171 warmup steps instead of 46) | v1 | Done 6 Nov 01:38 IST. Val 4.0061 (clean 4.0737), perplexity 54.93, MCQ-650 165/650 (25.4%) |
| `baseline_s1`, `dense_6l_s1`, `loop_6x2_s1` | v1 configs, `--seed 1` | v1 | Done 6 Nov (08:55, 19:57 and 15:58 IST). Val 2.8868, 2.6970 and 4.5024 (clean 2.9483, 2.7573 and 4.5713) |
| `d1024_loop6x2_v2` | `configs/model_d1024_loop6x2.yaml` (90.1M) | v2 | **Done 8 Nov 14:45 IST.** Val 1.9990 (clean 2.0571), perplexity 7.38, MCQ-650 173/650 (26.6%), CF-norm 26.5% |
| `baseline_v2_s1` | `configs/model.yaml`, `--seed 1` | v2 | **Done 9 Nov 16:15 IST.** Val 2.0094 (clean 2.0674), perplexity 7.46, MCQ-650 184/650 (28.3%), CF-norm 27.8% |

Clean-token and conditional-likelihood results for every run above are in section 2.1 and in `paper/data/`.

## 7. Where to find the raw files

- Results: `paper/data/results/` (the official `<run>.mmlu_cs.json` and `<run>.benchmark.json` files for every run) and `paper/data/` (clean re-score and conditional MCQ files).
- Checkpoints: not included in this repository.
- Config files: `configs/`. Parameter counts are checked with `scripts/verify_param_count.py`.
