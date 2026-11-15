# Data Source License Ledger

Every source used in the training corpus must have a row here with a
recorded license check **before** it enters the pipeline
(Architecture.md §11.2). No source with `decision: pending-review`
or `decision: exclude` is used for tokenizer training or model training.

Candidates below come from the dataset shortlist drawn up during planning. This table is
the actual gate — the shortlist is a starting point, not a pre-approval.

| Source name | URL | License | Scope of use | Date checked | Decision |
|---|---|---|---|---|---|
| Open Data Structures (Pat Morin) | https://opendatastructures.org/ | CC-BY 2.5 CA | training corpus, non-commercial academic project | 2026-09-07 | include |
| OSTEP (Operating Systems: Three Easy Pieces) | https://pages.cs.wisc.edu/~remzi/OSTEP/ | Free for non-commercial educational use (explicit grant on book site) | training corpus, non-commercial academic project | 2026-09-07 | include |
| IETF RFCs | https://www.rfc-editor.org/ | IETF Trust Legal Provisions — permits reproduction for educational/informational use | training corpus, non-commercial academic project | 2026-09-07 | include |
| OWASP documentation | https://owasp.org/ | CC-BY-SA 4.0 | training corpus, non-commercial academic project | 2026-09-07 | include |
| Wikipedia (CS categories) | https://dumps.wikimedia.org/ | CC-BY-SA 4.0 | training corpus, non-commercial academic project | 2026-09-07 | include |
| Wikibooks (CS categories) | https://dumps.wikimedia.org/ | CC-BY-SA 4.0 | training corpus, non-commercial academic project | 2026-09-07 | include |
| GitHub repos (license:mit / license:apache-2.0 / license:bsd-2-clause / license:bsd-3-clause) | https://github.com (filtered via search API) | MIT / Apache-2.0 / BSD-2-Clause / BSD-3-Clause (per repo, verified via API license field) | code/pseudocode training data | 2026-09-07 | include |
| Python official documentation | https://docs.python.org/ | PSF License (permits reproduction with attribution) | training corpus, non-commercial academic project | 2026-09-07 | include |
| PostgreSQL documentation | https://www.postgresql.org/docs/ | PostgreSQL License (BSD-style permissive) | training corpus, non-commercial academic project | 2026-09-07 | include |
| SQLite documentation | https://www.sqlite.org/docs.html | Public domain | training corpus, non-commercial academic project | 2026-09-07 | include |
| OpenDSA (Virginia Tech) | https://opendsa-server.cs.vt.edu/ | CC-BY-SA 4.0 | training corpus, non-commercial academic project | 2026-09-07 | include |
| MIT OpenCourseWare — 6.006, 6.046 lecture notes | https://ocw.mit.edu/ | CC-BY-NC-SA 4.0 | training corpus, non-commercial academic project only | 2026-09-07 | include |

## Attribution requirements for CC-BY / CC-BY-SA sources

Sources with CC-BY or CC-BY-SA licenses require attribution in any public
write-up or publication derived from this project. The following attribution
block covers all CC-licensed sources above:

> Training data includes material from: Wikipedia (CC-BY-SA 4.0), Wikibooks
> (CC-BY-SA 4.0), OWASP documentation (CC-BY-SA 4.0), Open Data Structures
> by Pat Morin (CC-BY 2.5 CA), OpenDSA (CC-BY-SA 4.0), and MIT OpenCourseWare
> lecture notes (CC-BY-NC-SA 4.0). Redistribution of the raw data is subject
> to the respective license terms.

## Process

1. Visit the source, confirm the license terms currently stated (terms can
   change — check at time of ingestion, not from memory of what a license
   "usually" is).
2. Update `Date checked` and flip `Decision` to `include` or `exclude`.
3. If `include`, note any attribution requirement (CC-BY / CC-BY-SA sources
   require attribution — track this separately if the model's output or a
   published write-up needs to carry it).
4. Only `include` rows may be pulled into `data/raw/` (not committed to git
   — see `.gitignore`).

## Excluded by default (Architecture.md §11.3, category 6)

Textbook PDFs, scanned lecture slides, and proprietary course material are
excluded unless an explicit, recorded permission grant from the rights
holder exists. None are currently on this ledger.

## Excluded sources (evaluated and rejected)

| Source name | Reason for exclusion |
|---|---|
| CLRS (Cormen et al.) | Copyright MIT Press — no redistribution license. Excluded per §11.3 category 6. |
| Sipser "Introduction to the Theory of Computation" | Copyright Cengage — no redistribution license. Excluded per §11.3 category 6. |
| Tanenbaum "Computer Networks" | Copyright Pearson — no redistribution license. Excluded per §11.3 category 6. |
| Stack Overflow content | CC-BY-SA 4.0 BUT Stack Exchange terms of service add restrictions on bulk use for ML training; legal ambiguity — excluded to avoid risk. |

## Corpus v2 sources (added 2026-10-30)

Used for `data/shards_v2` (Architecture.md §11.5). The license column is what the pipeline recorded in `manifest_new.json`. Upstream terms have not been re-checked by hand for this ledger, so re-check before any public release of the corpus.

| Source | Access | License (as recorded) | Scope | Date checked | Decision |
|---|---|---|---|---|---|
| FineWeb-Edu dedup | Hugging Face `HuggingFaceTB/smollm-corpus`, config `fineweb-edu-dedup` | ODC-By 1.0; CommonCrawl terms | training, non-commercial academic | 2026-10-30 | include (1.19B train tokens after caps) |
| Stack Exchange | Hugging Face `common-pile/stackexchange_filtered` | CC BY-SA (per-document license in metadata) | training | 2026-10-30 | include (1.19B train tokens after caps). This is Common Pile's re-release. It is not a scrape of Stack Overflow, which stays excluded (see the row above) |
| Cosmopedia v2 | Hugging Face `HuggingFaceTB/smollm-corpus`, config `cosmopedia-v2` | ODC-By 1.0 (synthetic) | training | 2026-10-30 | include (0.72B train tokens after caps) |
| arXiv papers | Hugging Face `common-pile/arxiv_papers_filtered` | open license per paper | training | 2026-10-30 | include (0.03B train tokens after caps) |
| LibreTexts | Hugging Face `common-pile/libretexts_filtered` | open license per page | training | 2026-10-30 | include (0.5M train tokens) |

The v2 decontamination removes any document that shares a 13-gram with `eval/benchmark.json` or `eval/mmlu_cs_benchmark.json`, or that overlaps 20% or more with the v1 validation split (Architecture.md §11.5).
