# AstraCSE

[![Paper DOI](https://img.shields.io/badge/Paper%20DOI-10.5281%2Fzenodo.24392658-blue)](https://doi.org/10.5281/zenodo.24392658) [![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24413831.svg)](https://doi.org/10.5281/zenodo.24413831)

Small decoder-only language models trained from scratch on computer-science text. The project studies **depth-wise weight sharing** ("loop" models, which apply a few unique transformer blocks several times) against dense models with the same number of unique parameters and against compute-matched dense baselines.

**Authors:** Abhay Tripathi (NyayaVeda AI and University of Lucknow), Ayush Kumar Tiwari, Ankit Gupta, Aaftab Ahmad, Abhinav Tripathi and Zeeshan Ali Siddiqui (University of Lucknow). Contact: abhay@nyayaveda.co.in
**Status:** all planned training runs are complete (19 of 19). The IEEE-format paper draft is in [`paper/`](paper/).

---

## Paper & Citation

- **Paper:** When Is Depth-Wise Weight Sharing Costly? A Controlled Study in Small Computer-Science Language Models
- **Authors:** Abhay Tripathi, Ayush Kumar Tiwari, Ankit Gupta, Aaftab Ahmad, Abhinav Tripathi, Zeeshan Ali Siddiqui
- **Preprint DOI:** https://doi.org/10.5281/zenodo.24392658
- **Software release DOI (v1.0.0):** https://doi.org/10.5281/zenodo.24413831

BibTeX (preprint):

```bibtex
@misc{tripathi2026depthwise,
  title     = {When Is Depth-Wise Weight Sharing Costly? A Controlled Study in Small Computer-Science Language Models},
  author    = {Tripathi, Abhay and Tiwari, Ayush Kumar and Gupta, Ankit and Ahmad, Aaftab and Tripathi, Abhinav and Siddiqui, Zeeshan Ali},
  year      = {2026},
  month     = nov,
  publisher = {Zenodo},
  note      = {Preprint},
  doi       = {10.5281/zenodo.24392658},
  url       = {https://doi.org/10.5281/zenodo.24392658}
}
```

---

## Headline results

Validation loss in nats (lower is better) on the held-out computer-science validation split. *Clean* excludes validation tokens that are verbatim copies of training text (7.6% of tokens). MCQ is accuracy on 650 MMLU computer-science questions, scored by conditional likelihood (chance = 25%).

| Model | Corpus | Params | Val (full) | Val (clean) | MCQ (CF-norm) |
|---|---|---:|---:|---:|---:|
| Dense-12 (baseline) | v1 | 96.5M | 2.853 | 2.914 | 26.9% |
| Dense-6 (matched to loop-6×2) | v1 | 54.0M | 2.795 | 2.856 | 24.3% |
| Loop-6×2 | v1 | 54.0M | 4.622 | 4.692 | 26.3% |
| Loop-6×2, 171 warmup steps | v1 | 54.0M | 4.006 | 4.074 | 24.8% |
| Dense-12 | v2 | 96.5M | 2.015 | 2.072 | 27.4% |
| Loop-6×2 | v2 | 54.0M | 2.084 | 2.140 | 26.8% |
| Dense-6 (matched to loop-6×2) | v2 | 54.0M | 2.135 | 2.192 | 25.5% |
| Dense-12, width 1,024 | v2 | 164.8M | **1.927** | 1.986 | 26.0% |
| Loop-6×2, width 1,024 | v2 | 90.1M | 1.999 | 2.057 | 26.5% |

The complete table (all 19 runs, including second seeds, the warmup ablation and the context extension) is in [`docs/RESULTS.md`](docs/RESULTS.md).

### Findings

1. **First regime (corpus v1: 1.15B tokens, 2,302 steps, 46 warmup steps).** At matched unique parameters, weight sharing costs about 1.8 nats (gap 1.83 on seed 0, 1.81 on seed 1). Parameter count and peak learning rate do not explain it.
2. **Warmup explains part of it.** Lengthening the warmup to 171 steps recovers about a third of the gap (0.62 of 1.84 nats). The remainder is attributed to other differences between the two regimes, which are not yet separated.
3. **Second regime (corpus v2: 4.30B tokens, 8,595 steps, 171 warmup steps).** The matched-parameter comparison reverses: the loop model is 0.05 nats better than its dense control, and 0.07 nats behind the 96.5M dense baseline at equal compute. These v2 results are single-seed.
4. **Width.** The width-1,024 dense model is the best model (1.927). The width-1,024 loop model reaches 1.999 with 55% of its parameters.
5. **MMLU-CS.** No model separates from chance under three conditional-likelihood scoring rules. The benchmark is kept as the standard domain test; at this scale validation loss is the informative metric.

Details, threats to validity and what would change these conclusions are in [`docs/FINDINGS.md`](docs/FINDINGS.md).

---

## Method

- **Model.** Decoder-only transformer: RoPE, RMSNorm, SwiGLU, causal attention, tied embeddings, 15,000-token byte-level BPE vocabulary, context 1,024.
- **Loop variant.** `n_unique_layers = K < n_layer` applies K unique blocks cyclically over the depth slots. A learned depth embedding is added before each application, so a shared block knows its depth. Figure 1 of the paper (`paper/figures/architecture.tex`) illustrates the design.
- **Controls.** Each loop model is compared with a dense model that has the same unique blocks and no sharing (matched parameters), and with a 12-layer dense model (matched compute).
- **Training.** AdamW, cosine learning-rate schedule with linear warmup, BF16 autocast with FP32 weights, effective batch of 499,712 tokens (488 sequences of 1,024). One epoch per run.
- **Evaluation.** Validation loss on every 8th batch of the v1 validation split, computed from the final checkpoint with a fresh loader. Clean-token re-scoring and the conditional-likelihood MCQ are described in [`docs/METHODS.md`](docs/METHODS.md).

The design is specified in [`Architecture.md`](Architecture.md). Exact settings for every run are in [`configs/`](configs/) and [`docs/METHODS.md`](docs/METHODS.md).

---

## Repository structure

```
AstraCSE/
├── configs/              YAML configuration per model (12 model configs)
├── src/                  model.py, data.py, train.py, infer.py
├── eval/                 run_eval.py (official evaluation), MMLU-CS benchmark (650), smoke test (10)
├── scripts/              corpus pipelines, training queue, evaluation and analysis scripts
├── scripts/overlap/      validation-overlap scan, clean-token re-score, conditional MCQ
├── tests/                unit tests
├── tokenizer/            frozen BPE tokenizer
├── data/                 license ledger for every training source (SOURCES.md)
├── docs/                 RESULTS, METHODS, FINDINGS, OPERATIONS
├── paper/                IEEE-format paper: main.tex, generated tables, figures, references
├── Architecture.md       design specification
├── Plan.md, Approach.md, Roadmap.md
└── Training_Run_Log.md   dated log of the experiments
```

---

## Getting started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the unit tests
pytest tests/ -q

# Check a configuration's parameter count against its declared value
python scripts/verify_param_count.py configs/model.yaml
```

### Corpus pipeline

```bash
# v1 (local; large sources are resumable)
python scripts/download_parts.py --status
python scripts/clean_text.py --raw-dir data/raw --out-dir data/cleaned
python scripts/dedup.py
python scripts/train_tokenizer.py
python scripts/pack_shards.py
python scripts/check_contamination.py

# v2 (see Architecture.md section 11.5)
bash scripts/run_corpus_v2.sh
```

### Training and evaluation

Runs are executed by a job queue (`scripts/run_queue.sh`) that reads its job list, trains each configuration with checkpointing and resume, evaluates it, and archives the results. Operating notes are in [`docs/OPERATIONS.md`](docs/OPERATIONS.md). All experiments were run on a single NVIDIA L4 GPU (AWS `g6.xlarge`).

### Paper

The paper is built from `paper/main.tex`. Build instructions and the list of generated files are in [`paper/README.md`](paper/README.md). Tables and references are regenerated with `paper/make_tables.py`, `paper/make_bib.py` and `paper/check_tex.py`.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | Final numbers for all 19 runs: validation loss, clean-token loss, MCQ under each scoring rule, per-subject counts |
| [`docs/METHODS.md`](docs/METHODS.md) | Model, data, training, evaluation protocol, decisions and known gaps |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | Interpretation, threats to validity, suggested framing |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | Operating the training queue, adding or retrying jobs |
| [`Architecture.md`](Architecture.md) | Design specification |
| [`Training_Run_Log.md`](Training_Run_Log.md) | Dated log of runs, incidents and decisions |
| [`data/SOURCES.md`](data/SOURCES.md) | License ledger for the training data |

---

## Limitations

- **Single seed for corpus v2.** The second-regime results, including the matched-parameter reversal, are single-seed. Corpus v1 has a second seed for the three main configurations.
- **Small scale.** Models have 32.8M to 164.8M parameters, trained for one epoch on 1.15B (v1) or 4.30B (v2) tokens.
- **Domain.** Computer-science text only.
- **Benchmark.** MMLU-CS is at chance for all models at this scale, so it does not rank them.
- **Validation overlap.** 7.6% of validation tokens are verbatim copies of training text. All results are reported on both the full and the clean validation tokens.
- **In-training validation logs.** The values logged during training come from a loader whose position carries over between evaluations, so they are not comparable across evaluations. Only the official final evaluation is used for results.

---

## Status

- Training: 19 of 19 planned runs complete.
- Paper: IEEE conference-format draft with tables and figures. The repository link will be added at publication.
- Open items: a second seed for the v2 matched pair, and a v1 ablation with source-shuffled shards.

---

## License

Code, configurations, tests and documentation are released under the MIT License (see [`LICENSE`](LICENSE)). The training data keeps its own licenses, listed per source in [`data/SOURCES.md`](data/SOURCES.md).
