# Paper draft (IEEE conference format)

Working title: *When Is Depth-Wise Weight Sharing Costly? A Controlled Study in Small Computer-Science Language Models*.

Authors: Abhay Tripathi (NyayaVeda AI and University of Lucknow), Ayush Kumar Tiwari, Ankit Gupta, Aaftab Ahmad, Abhinav Tripathi and Zeeshan Ali Siddiqui (University of Lucknow). Contact: abhay@nyayaveda.co.in.

**Status (2026-11-10):** third draft (6 pages). All pending markers are filled. All 19 runs are in the tables (official, clean-token and conditional MCQ). The only open item is the repository URL, to be added at publication.

## Build

No LaTeX is installed on the working PC. Use Overleaf:

1. New Project, Upload Project, upload this `paper/` folder as a zip.
2. Compiler: pdfLaTeX. Main document: `main.tex`. `IEEEtran.cls` and `IEEEtran.bst` are built into Overleaf.

Locally, with a TeX distribution: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

Local preview: Tectonic 0.17.0, a single portable exe on the PATH (official GitHub release, SHA-256 checked against GitHub's published digest). From this folder: `tectonic.exe --outdir <build dir> main.tex`. It runs BibTeX itself. The first run downloads the TeX files it needs; later runs take about 20 seconds. Tectonic uses XeTeX, which has no Times for IEEEtran, so `main.tex` loads the TeX Gyre Termes clone under XeTeX only; pdfLaTeX builds (Overleaf, IEEE) are unaffected. Preview PDFs are kept outside the repo.

Length: the abstract is about 230 words, which is within the usual IEEE limit. The paper is 6 pages.

## Files

| File | What it is |
|---|---|
| `main.tex` | The paper |
| `references.bib` | 28 references. Generated, do not edit by hand |
| `tables/*.tex` | All result tables (models, v1 results, matched gaps, v2 results, v2 curve, v2 shards, MCQ). Generated, do not edit by hand |
| `make_tables.py` | Builds `tables/` from `configs/` and `data/`, and prints the derived numbers used in the text |
| `make_bib.py` | Builds `references.bib` from `data/arxiv_metadata.json` |
| `check_tex.py` | Static checks: citations, labels, inputs, braces, begin/end; lists the pending markers |
| `data/results/<run>.mmlu_cs.json`, `<run>.benchmark.json` | Official evaluation files for every run, copied from `s3://<your-s3-bucket>/results/<run>/` |
| `data/rescore_*.json` | Clean-token re-scores (v1 and baseline_v2, loop6x2_v2, d1024 runs, wu171, seeds) |
| `data/mcq_cond.json`, `mcq_rescore_d1024.json`, `mcq_rescore_wu171.json`, `mcq_rescore_seeds.json` | Conditional-likelihood MCQ results (CF-norm, CF-sum, MCF) |
| `data/curve_v2.json` | Clean validation loss of the v2 runs at periodic checkpoints |
| `data/subject_check.json`, `data/val_coverage.json` | Per-shard and per-decile check, and the overlap audit |
| `data/arxiv_metadata.json` | Titles, authors and years of the arXiv references, fetched from the arXiv API |

Run all scripts from the repo root:

```bash
python paper/make_tables.py
python -X utf8 paper/make_bib.py
python paper/check_tex.py
```

## Where the numbers come from

Every number in the tables is computed by `make_tables.py` from the files above. Numbers in the text were taken from its printed output (gaps, changes, ranges, tokens per parameter). If a result file changes, rerun the script and update the text to match.

References: arXiv titles and author lists come from the arXiv API, not from memory. Venues are given only where the arXiv record states them or they are well established; the rest cite the arXiv preprint. Three entries are not on arXiv (GPT-2 report, Broder 1997, SmolLM-Corpus dataset) and are written by hand in `make_bib.py`.

## Pending items (red in the draft)

| Pending item | Where it is in the draft | Status |
|---|---|---|
| Seed spread of the v1 losses | Section "Width and seeds" | Done (10 Nov): table `tab:seeds` |
| Loop-6x2 at width 1,024 (`d1024_loop6x2_v2`, 90.1M) | Section "Width and seeds" | Done (10 Nov): val 1.999 (clean 2.057) |
| GPU hours | Section "Hardware" | Done (10 Nov): about 250 active GPU-hours from the run logs |
| Repository URL | Section "Reproducibility" | Open: to be added when the repository is made public (owner's decision) |
| Final sentence of the conclusion | Section "Conclusion" | Done (10 Nov) |

Done since the last draft: the matched-parameter control on v2 (`dense_6l_v2`), the warmup ablation (`loop_6x2_wu171`), the width-1,024 results (`d1024_v2`, `d1024_ctx2048`), the MMLU-CS conditional-likelihood results, and the second v1 seeds.

## Claims to keep honest

- The two training regimes differ in warmup, schedule length, data mix and document order at once. The warmup ablation isolates only the warmup: it recovers about a third of the v1 penalty. The remaining explanations are hypotheses.
- The v2 results (including the matched-parameter reversal of -0.05 nats) are single-seed. The v1 matched gap is stable across two seeds (1.83 and 1.81); the compute-matched gap is not (1.78 and 1.62).
- MCQ accuracy is at chance for every model under every rule and is not used to rank them.
- Absolute validation losses are reported on both the full and the clean tokens because of the train-validation overlap in v1.
