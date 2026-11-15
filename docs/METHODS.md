# AstraCSE: Methods

This file describes how the experiments were run, so another person can reproduce or check them. Results are in `docs/RESULTS.md`. Interpretation is in `docs/FINDINGS.md`. Architecture and design rationale are in `Architecture.md`.

## 1. Model

Decoder-only transformer, implemented in `src/model.py`.

| Component | Setting |
|---|---|
| Position encoding | RoPE, base theta = 10,000 |
| Normalization | RMSNorm, eps = 1e-5, no bias |
| MLP | SwiGLU (three matrices), no bias |
| Attention | Causal multi-head, fused scaled-dot-product attention |
| Embeddings | Tied input embedding and LM head |
| Residual scaling | GPT-2 style on the output projections |
| Vocabulary | 15,000 (byte-level BPE, frozen) |
| Baseline shape | d_model 768, 12 heads (d_head 64), d_ff 2,048, 12 layers, context 1,024 |

**Loop variant.** Setting `n_unique_layers = K < n_layer` makes the model apply `K` unique blocks cyclically (`block = unique[depth_slot % K]`). Each depth slot has its own KV cache entry. A learned depth-conditioning embedding of shape `n_layer x d_model` (initialized with std 0.02) is added to the residual stream before each depth slot. The embedding is 9,216 parameters at d_model 768.

**Width variant.** `model_d1024*.yaml` uses d_model 1024, 16 heads, d_ff 2,688 (164.8M parameters). Its loop variant has K = 6 and 90.1M parameters.

Parameter counts are asserted by `scripts/verify_param_count.py` against each config's `expected_param_count`. All 11 configs pass.

## 2. Tokenizer

Byte-level BPE with a 15,000-token vocabulary. It was trained on the v1 cleaned corpus and frozen. The vocabulary hash starts with `539570c884f6e1b5`. Corpus v2 reuses the same tokenizer, so v1 and v2 token IDs are compatible.

Special tokens: `<bos>`, `<eos>`, `<pad>`, `<doc>`. Documents are separated by `<doc>` inside a packed stream.

## 3. Data

### v1 (used by all the completed runs)

- 275 training shards and 6 validation shards, stored as `uint16` memmaps, sequence length 1,024.
- Training tokens: 1,151,073,280.
- Sources: Wikipedia CS, arXiv CS, The Stack (permissive licenses only), plus small permissively licensed sources (RFCs, OWASP, Python / PostgreSQL / SQLite docs, OSTEP, Open Data Structures). Full ledger: `data/SOURCES.md`.
- Deduplication: exact and MinHash/LSH near-duplicate removal.
- Decontamination: 13-gram overlap against the 20-item smoke-test benchmark, checked before packing (0 of 20 items found). v1 and v2 training shards were both checked against the 650-item MMLU-CS set on 2 Nov, with the repo's 13-gram rule: 0 of 650 items flagged in either (see section 8).
- **Train–validation overlap (checked 2026-11-02, not removed):** 19.6% of validation tokens lie inside a verbatim 13-token match with the training shards, and 7.6% lie inside verbatim runs of 100 or more tokens (copied whole files). The split was made by document before packing, but copies across the split were not removed. See section 8.

### v2 (scaling runs)

- Built by `scripts/expand_corpus.py` and `scripts/run_corpus_v2.sh`. Four openly licensed public datasets were filtered by a CS keyword score. Details and licenses are in `Architecture.md` §11.5 and `data/SOURCES.md`.
- 1,025 training shards (750 new, plus the 275 v1 shards copied server-side), 4,296,117,248 tokens.
- Held-out `val_new`: 4 shards, 15,826,944 tokens. These documents never appear in any training shard.
- Decontamination: a document is dropped if it shares any 13-gram with a benchmark item, or if 20% or more of its 13-grams appear in the v1 validation split.

### Epoch definition

One epoch is `floor((tokens - 1) / 1024)` sequences per shard, summed over shards, then divided by 488 sequences per step. This gives 2,302 steps for v1 (matching the completed baseline) and 8,595 steps for v2.

## 4. Training

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 (matrices only) |
| Gradient clipping | Global norm 1.0 |
| LR schedule | Linear warmup over the first 2% of steps, cosine decay to 10% of peak |
| Peak LR | 6e-4 (all baseline, dense and loop configs); 3e-4 (`loop_6x2_lr3e-4`); 5e-4 (`d1024*`); 1e-4 (`d1024_ctx2048` continuation) |
| Effective batch | 499,712 tokens per step (= 488 sequences of 1,024 tokens) |
| Micro-batch | 8 for d_model 768; 4 for d_model 1,024 |
| Gradient accumulation | Computed by `train.py` as `target_effective_batch_tokens // (micro_bs x seq_len)`: 61 (micro 8) or 122 (micro 4) |
| Precision | BF16 autocast; FP32 weights, gradients and optimizer state |
| Steps | 2,304 configured for v1 (epoch ends at 2,302); 8,595 for v2 epoch; 400 for the context extension |
| Compilation | `torch.compile` for the later runs: `loop_4x3` from step 1,078 onward, and every job after it (1.67x faster). The same-batch loss matched eager to 4 decimals. `baseline` and `loop_6x2` ran eager |
| Checkpoints | Periodic every 200 steps; `sigterm-step-N.pt` on SIGTERM; `step-*-final.pt` at the end |
| Early stopping | Configurable (`--early-stop-patience`). Never triggered in the v1 runs. Disabled (0) for the one-epoch v2 runs |
| Safety | NaN/Inf guard (halt after 3 consecutive), OOM handling (halve micro-batch and recompute accumulation) |

**Interruption handling.** Spot reclaims and service restarts send SIGTERM. `train.py` saves a `sigterm-step-N.pt` checkpoint and exits. The queue leaves the job unmarked and resumes it from the newest checkpoint. All the interruptions so far resumed from the saved step.

**Seeds.** Every run records its seed as `seed=N` in `train.log`. The first two runs (`baseline`, `loop_6x2`) predate seed logging, so their seeds are not recorded. Later runs use `--seed` where it is given; otherwise the seed is random and recorded.

## 5. Evaluation protocol

Implemented in `eval/run_eval.py`.

**Validation loss and perplexity.** The mean token cross-entropy over every 8th batch of the v1 validation set (batch size 4, sequence length 1,024). This covers about one eighth of the validation set. Perplexity is `exp(loss)`. Every batch in the stride is used, so the subset spans all the validation subjects. An earlier version used the first 200 batches, which cover only the first subjects, and was replaced.

**MCQ-650 (`eval/mmlu_cs_benchmark.json`).** 650 four-choice questions from six MMLU subjects. For each choice, the model computes the mean token cross-entropy of the full text `question + "\n" + choice`. The lowest value wins, and ties go to the first choice. Accuracy is correct / 650.

*Limitation.* The loss is averaged over all tokens, including the question tokens that are shared across choices. The rule is therefore not the standard conditional likelihood of the answer given the question. It is a weak signal for four-choice questions at this scale, which is why MCQ results sit near chance (see `docs/FINDINGS.md`).

**Conditional-likelihood MCQ (4 Nov, all runs).** The same 650 questions were scored with three likelihood rules fixed before any result was seen, in `scripts/overlap/mcq_cond_box.py`. CF-norm (primary) takes the log-likelihood of the choice text after "Question: ... Answer:", divided by its character count. CF-sum takes the same log-likelihood without dividing. MCF scores the answer letter after the four choices are listed. Results: `paper/data/mcq_cond.json`, `mcq_rescore_d1024.json`, `mcq_rescore_wu171.json`, `mcq_rescore_seeds.json`. Every model is at chance under every rule.

**Smoke-test benchmark (`eval/benchmark.json`).** Ten hand-written multiple-choice items, one per subject. Used as a sanity check only.

**In-training validation.** The training loop logs a value from the first 50 unstrided validation batches. This is a monitoring proxy. It is biased toward the first subjects and is not the reported metric.

## 6. Environment

- Instance: one NVIDIA L4 GPU (24 GB) on a cloud spot instance (AWS EC2 `g6.xlarge`). A persistent spot request relaunched the instance after each reclaim. The orchestration is described in `Architecture.md` §14.4.
- Queue: jobs ran in sequence with checkpointing and resume. The orchestration scripts are not part of this release (see `Architecture.md` §14.4).

## 7. Decisions and reasons

| Date | Decision | Reason |
|---|---|---|
| 2026-10-28 | One epoch of v1 (2,302 steps) for every controlled comparison | Equal data and steps across configs |
| 2026-10-28 | Same peak LR (6e-4) for the dense and loop configs | Matched hyperparameters for the headline comparison |
| 2026-10-29 | Dense controls with `K` unique layers and no sharing | Isolates parameter count from sharing |
| 2026-10-30 | `torch.compile` for all jobs from `loop_4x3` step 1,078 | 1.67x speedup at identical loss |
| 2026-10-30 | Corpus v2 built as a separate pipeline on a data box, reusing the v1 tokenizer | No tokenizer change mid-project (Architecture §21.6) |
| 2026-10-30 | v2 steps set to the full epoch (8,595) | Calibrated on v1, where the formula gives the observed 2,302 |
| 2026-10-30 | Stack Exchange used from the Common Pile re-release, not by scraping Stack Overflow | Licensing already resolved upstream; see `data/SOURCES.md` |
| 2026-10-30 | Micro-batch 4 for the d_model 1,024 jobs | Memory safety on a 24 GB L4 (not measured) |
| 2026-10-31 | LR check (`loop_6x2_lr3e-4`) did not change the loop configs' LR | The change was 0.03 nats, below the threshold set in advance |
| 2026-10-31 | v1 seeds moved ahead of `d1024_loop6x2_v2` in the queue | Error bars for the headline result (user approval) |
| 2026-10-31 | Diagnostic volume, the old AMI and its snapshots, and the diagnostic snapshot deleted; fixed AMI and its snapshot kept | Clutter and storage; the fixed AMI is what the persistent spot request relaunches from (user approval) |
| 2026-11-04 | Conditional-likelihood MCQ (CF-norm primary, CF-sum, MCF) added as an evaluation-only step; training code frozen | The old mean-loss rule is weak; the three rules were fixed before any result was seen |
| 2026-11-04 | In-training validation fix rejected; code frozen for the rest of the run | The official evaluation does not use it; logged values are a monitoring signal only |
| 2026-11-05 | `dense_6l_v2` (matched-parameter control on v2) kept before the seeds | Completes the v2 matched comparison early |
| 2026-11-05 | Warmup ablation `loop_6x2_wu171` queued after `dense_6l_v2`; no job removed | Tests whether the short v1 warmup causes the v1 penalty |
| 2026-11-06 | `d1024_loop6x2_v2` failed because its config was missing on the GPU box; config copied to the box (checksum verified), failed marker removed, queue restarted | The committed config is unchanged; the job now runs from step 0 |

## 8. Known gaps in the method

- **Train–validation overlap (v1), found 2026-11-02.** A full scan of all 275 training shards against all 6 validation shards (every 13-token window, exact match) found verbatim copies across the split. 7.6% of validation tokens lie in copied runs of 100 or more tokens, and 19.6% lie in any 13-token match. The longest copies are long source files, including a 15,020-token C driver. The v1 validation numbers reported so far (all v1 runs, and `baseline_v2`) are therefore optimistic. Re-scoring on the clean tokens (3 Nov, `docs/RESULTS.md`) raises them by 0.06 to 0.07 nats, and leaves the loop-minus-dense gaps unchanged (within 0.01 nats). The v2 new documents are clean on a 10% sample: 0.1% of their windows match validation, and the longest shared run is 346 tokens (placeholder and license text). The cause has not been traced. Done on 3 Nov for the nine completed checkpoints and for `loop6x2_v2` (`docs/RESULTS.md`). The width-1,024 runs (`d1024_v2`, `d1024_ctx2048`) were re-scored on 5 Nov. `d1024_loop6x2_v2` finished on 8 Nov 2026 and was re-scored on the same day (`docs/RESULTS.md`). Scripts: `scripts/overlap/`.
- **Contamination (v1).** The v1 decontamination check covered only the 20-item smoke-test benchmark. On 2 Nov the v1 and v2 training shards were both checked against the 650-item MMLU-CS set (all 275 and all 1,025 shards) with the repo's 13-gram rule: no item is flagged, and no benchmark 13-gram occurs in either corpus. During v2 construction, 135 documents that contained benchmark 13-grams were dropped.
- **In-training validation (known issue, found 3 Nov).** `train.py` builds one validation DataLoader and reuses it for every eval. Its position carries over between evals, so each eval reads a different slice of the validation set. The logged in-training `val_loss` values, and the `best_val_loss` used to pick the "best" checkpoint, are therefore not comparable across evals or runs. The official numbers are not affected: they use the final checkpoint and a fresh loader (`eval/run_eval.py`). Checked with `scripts/overlap/subject_check_box.py` (3 Nov): the same first 400 sequences give 3.05 (loop) and 2.97 (baseline) on the final checkpoints. A code fix (a fresh validation loader per eval) was considered and rejected on 4 Nov: the code stays frozen for the rest of the run, and the logged values are used only as a qualitative tracker.
- Seeds: the v1 runs `baseline`, `dense_6l` and `loop_6x2` have a second seed (`--seed 1`). Every other configuration has one seed.
- LR: tuned only for the baseline schedule. The loop variants were checked at one alternative LR (`loop_6x2_lr3e-4`), not swept.
- MCQ scoring: done 4 Nov. The 650 MMLU-CS questions were re-scored with three conditional-likelihood rules fixed before any result was seen (`scripts/overlap/mcq_cond_box.py`): cloze per character (CF-norm, primary), cloze total (CF-sum) and answer letter (MCF). Every model is at chance under every rule (CF-norm 22.9% to 27.4%; best 1.40 standard errors above 25%). MCF is pure position bias: models answer "A" on 82% to 100% of questions. The answer key is unbalanced (C 28.8%, A 26.0%, B 23.1%, D 22.2%). Results: `paper/data/mcq_cond.json`, `paper/tables/mcq.tex`.
- Benchmark coverage: MMLU-CS covers six of the ten subjects in the training corpus.
- Evaluation subset: the validation metric uses every eighth batch, which is a valid comparison across models but not the full validation set.
