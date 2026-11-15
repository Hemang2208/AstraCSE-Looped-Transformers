# Production Architecture Specification: AstraCSE-100M

**Project:** AstraCSE-100M
**Project Type:** Academic / College Research Project (IEEE-general-paper-level contribution)
**Document Type:** Production Architecture, Data, Cost, and Operations Specification
**Architecture Class:** Decoder-Only Autoregressive Transformer (dense and weight-shared "loop" variants, Section 31)
**Domain:** Core Computer Science — 10 Subject Areas (training), 6 subjects covered by the 650-question MMLU-CS eval set
**Target Model Size:** 96,473,856 Trainable Parameters for the baseline (verified derivation, Section 3); the family spans 32.8M–164.8M across 12 configs (Section 31.4)
**Primary Training Infrastructure:** AWS EC2 Spot, account `astracse`, `g6.xlarge` (1x NVIDIA L4)
**Training Compute:** one NVIDIA L4 GPU, about 250 active GPU-hours for all 19 runs and their evaluations (Section 19).

**Status (2026-11-11):** Sections 0–21 and 25–27 describe the implemented
system as it actually runs. Sections 22–24 and 28–31 are updated in place
with measured results where training has produced them; anything still
describing a plan rather than a measurement says so explicitly. This document is the architecture and design record of the study. The infrastructure sections (Sections 14 and 18) describe the queue, watchdog, launcher and stop-handler scripts used to run the experiments on cloud hardware. Those scripts, their tests and the operations runbook are not part of this release; the training and evaluation code is.

---

## 0. Revision Note — Bug Fix From Previous Draft

The previous revision of this document specified `vocab_size = 32,000` and
`d_ff = 3,072`. With those numbers and the stated `d_model=768`, `n_layer=12`
config, the actual parameter count is:

```text
Embedding:        32,000 × 768              = 24,576,000
Per-layer attn:   4 × 768 × 768              =  2,359,296
Per-layer MLP:    3 × 768 × 3,072            =  7,077,888
Per-layer norms:  2 × 768                    =      1,536
Per-layer total:                             =  9,438,720
12 layers:                                   = 113,264,640
Final norm:                                  =        768
─────────────────────────────────────────────────────────
TOTAL                                        = 137,841,408  (~137.8M)
```

That is **~43% over** the declared ~96.5M target — a real spec bug, not a
rounding difference. It would have surfaced mid-implementation as a
parameter-count mismatch, or worse, as a silent VRAM miscalculation
carried through the whole throughput model.

**Fix applied in this revision:** `vocab_size = 15,000`, `d_ff = 2,048`,
all linear layers and norms use no bias term. This yields exactly:

```text
Embedding:        15,000 × 768              = 11,520,000
Per-layer attn:   4 × 768 × 768              =  2,359,296
Per-layer MLP:    3 × 768 × 2,048            =  4,718,592
Per-layer norms:  2 × 768                    =      1,536
Per-layer total:                             =  7,079,424
12 layers:                                   = 84,953,088
Final norm:                                  =        768
─────────────────────────────────────────────────────────
TOTAL                                        = 96,473,856  (~96.5M) ✓
```

Every parameter figure, VRAM estimate, and throughput estimate in this document
uses the corrected config. See Section 3 for the full breakdown and Section
10 for why a 15,000-token vocabulary is still sufficient for this domain.

---

## 1. Architecture Overview

AstraCSE-100M is a compact, decoder-only, autoregressive Transformer designed
for domain-specialized Computer Science knowledge modeling.

The architecture is optimized for:

- Core Computer Science knowledge representation
- Technical question answering
- Concept explanation
- Code and pseudocode generation
- Multi-step technical reasoning
- MCQ and examination-oriented knowledge retrieval
- Efficient training on a single cloud GPU
- Low-memory local inference deployment

The model follows the general architectural principles used in modern
decoder-only LLMs, including:

- Pre-normalization
- RMSNorm (no bias)
- Rotary Position Embeddings (RoPE)
- Multi-Head Self-Attention (MHA), causal
- SwiGLU feed-forward blocks
- Weight tying between token embeddings and the language-model head
- No bias terms on any linear projection (modern practice — fewer params,
  comparable or better stability than biased projections)
- Autoregressive causal masking
- Mixed-precision training (BF16 compute, FP32 optimizer state)

### High-Level Architecture

```text
Input Text
    │
    ▼
┌──────────────────────────────┐
│ Custom Byte-Level BPE        │
│ Vocabulary: 15,000 Tokens    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Token IDs                    │
│ [Batch, Sequence Length]     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Token Embedding              │
│ [15,000 × 768]                │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 12 × Transformer Blocks      │
│                              │
│  RMSNorm                     │
│      ↓                       │
│  RoPE + Causal MHA            │
│      ↓                       │
│  Residual Connection         │
│      ↓                       │
│  RMSNorm                     │
│      ↓                       │
│  SwiGLU MLP                   │
│      ↓                       │
│  Residual Connection         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Final RMSNorm                │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ LM Head                      │
│ Weight-Tied with Embeddings  │
└──────────────┬───────────────┘
               │
               ▼
         Next-Token Logits
```

---

## 2. Model Configuration

### 2.1 Primary Hyperparameter Matrix

| Category | Parameter | Value | Purpose |
|---|---|---:|---|
| Model Capacity | Total Parameters | 96,473,856 | Compact domain-specialized capacity |
| Architecture | Model Type | Decoder-Only Transformer | Autoregressive generation |
| Context | Maximum Sequence Length | 1,024 tokens | Efficient technical text and code context |
| Vocabulary | Vocabulary Size | 15,000 tokens | Domain-scoped byte-level BPE vocabulary |
| Hidden State | `d_model` | 768 | Internal representation width |
| Depth | Transformer Layers | 12 | Hierarchical representation depth |
| Attention | Number of Heads | 12 | Parallel attention subspaces |
| Head Dimension | `d_head` | 64 | `768 / 12` |
| MLP | Feed-Forward Dimension (`d_ff`) | 2,048 | SwiGLU expansion (3-matrix) |
| Activation | MLP Activation | SwiGLU | Gated nonlinear representation |
| Bias | Linear/Norm Bias | Disabled | Fewer params, modern-LLM convention |
| Normalization | Normalization | RMSNorm | Stable pre-normalized training |
| Normalization Epsilon | `eps` | `1e-5` | Numerical stability |
| Position Encoding | RoPE | Enabled | Relative positional representation |
| Attention Type | Causal MHA | Enabled | Autoregressive masking |
| Weight Tying | Input/Output Embeddings | Enabled | Parameter reduction |
| Training Precision | Mixed Precision | BF16 compute / FP32 master | Reduced memory, stable updates |
| KV Cache | Inference | Enabled | Efficient autoregressive decoding |

### 2.2 Why 15,000 Vocabulary Is Sufficient

A 32,000-token vocabulary is calibrated for general-purpose multilingual
text. This model targets a single language (English) and a narrow technical
domain (10 CS subjects: natural-language explanation + code/pseudocode). A
byte-level BPE tokenizer at 15,000 tokens covers this domain efficiently
because:

- Byte-level fallback guarantees zero `<unk>` tokens regardless of vocab size
  (any input decomposes to bytes in the worst case).
- Domain text has a narrower symbol distribution (recurring CS terminology,
  a handful of programming languages) than general web text, so fewer merge
  rules are needed to reach good compression.
- A smaller vocabulary directly reduces the single largest parameter
  block (the tied embedding/LM-head matrix — 11.5M of the 96.5M total), which
  is the primary lever that keeps this model at ~96.5M instead of ~138M.

This must still be validated empirically (Section 22): measure the
tokenizer's average bytes-per-token and out-of-domain fallback rate on a
held-out sample before freezing the vocabulary size.

---

## 3. Parameter Budget

### 3.1 Full Breakdown

```text
Total Model Parameters: 96,473,856
│
├── Token Embedding Matrix
│   └── 15,000 × 768 = 11,520,000    (11.94%)
│
├── Transformer Blocks (× 12)                84,953,088    (88.06%)
│   ├── Attention Projections (Wq,Wk,Wv,Wo)
│   │   └── 4 × 768 × 768 = 2,359,296 / layer  (28,311,552 total)
│   ├── SwiGLU MLP (W_gate, W_up, W_down)
│   │   └── 3 × 768 × 2,048 = 4,718,592 / layer (56,623,104 total)
│   └── RMSNorm scale params (× 2 per layer)
│       └── 2 × 768 = 1,536 / layer            (18,432 total)
│
├── Final RMSNorm
│   └── 768                                     (0.0008%)
│
└── LM Head — weight-tied with Token Embedding
    └── No additional independent [V × d_model] matrix
```

Non-embedding parameter count: **84,953,856** (embedding is ~12% of total —
low enough that further vocabulary tuning has limited leverage; the
transformer stack dominates the budget, as intended).

### 3.2 Weight Tying

The input token embedding matrix is reused by the output language-model
head.

```text
Token IDs
    │
    ▼
Embedding Matrix [V × d_model]
    │
    ▼
Transformer
    │
    ▼
Hidden States
    │
    ▼
Transpose(Embedding Matrix)
    │
    ▼
Vocabulary Logits
```

For vocabulary 15,000 and hidden dimension 768:

```text
15,000 × 768 = 11,520,000 parameters saved
```

versus maintaining an independent output projection. Weight tying is
mandatory in this design, not optional — without it the model would sit at
~108M params, breaking the size budget.

### 3.3 Initialization Strategy

- Embedding and all linear weights: normal init, `std = 0.02`.
- Output projections inside each block (`Wo` in attention, `W_down` in MLP):
  scaled by `1 / sqrt(2 * n_layer)` at init (GPT-2-style residual-scaling
  trick) to prevent residual-stream variance growth across 12 stacked
  blocks.
- RMSNorm scale parameters: initialized to 1.0.
- No bias parameters exist, so no bias initialization is needed.

This is a standard, low-risk initialization scheme for a model this size —
it does not require tuning before the Phase 0 debug run (Section 22), but
the debug run's loss curve (should drop smoothly from `ln(vocab_size) ≈
9.62` nats at step 0) is the checkpoint that confirms it's wired correctly.

---

## 4. Token Embedding Layer

### 4.1 Input

```text
Input Shape: [batch_size, sequence_length]
Example:     [B, 1024]
```

Each integer is a token ID from the tokenizer vocabulary (range `[0, 15000)`).

### 4.2 Embedding Matrix

```text
Embedding Weight Shape: [V, d_model] = [15,000, 768]
```

```text
Token ID → Embedding Lookup → 768-Dimensional Vector
```

Resulting tensor: `[B, 1024, 768]`.

---

## 5. Positional Representation: Rotary Position Embeddings

The architecture uses **Rotary Position Embeddings (RoPE)** instead of
learned absolute position embeddings or sinusoidal encodings. RoPE is
applied to Query and Key representations inside every attention layer.

```text
Q ──────────────┐
                ├──► RoPE Rotation ───► Q'
K ──────────────┘
                └──► RoPE Rotation ───► K'
```

### 5.1 Purpose

RoPE encodes relative positional relationships directly into the attention
dot-product via rotation, rather than through additive position vectors.

**Base frequency (θ): 10,000** — the standard value used by GPT-NeoX,
LLaMA, and most RoPE implementations (`src/model.py::_build_rope_cache`:
`theta = 1 / (10000 ** (arange(0, d_head, 2) / d_head))`). Not made
configurable: NTK-aware / YaRN rescaling of θ is a technique for
*extending* a model past its trained context length, which this project
explicitly does not attempt (§5.2) — there is no scenario in this spec
where a different θ would be used, so exposing it as a tunable would be a
knob nobody turns, not a real degree of freedom.

### 5.2 Context-Length Constraint

```text
Trained maximum context length: 1,024 tokens
```

Production inference must stay within this trained boundary. RoPE
extrapolation beyond 1,024 tokens (e.g. via NTK-aware scaling or YaRN) is
**not enabled by default** and must not be assumed to work — it requires a
dedicated validation pass (perplexity on long-context held-out data) before
being exposed in any deployed inference path. Treat >1024-token input as an
explicit truncate-or-reject case in the inference server (Section 25), not
as silently-degraded behavior.

---

## 6. Transformer Block Architecture

```text
12 Transformer Blocks, each:

Input x
  │
  ▼
RMSNorm
  │
  ▼
RoPE + Causal MHA
  │
  ▼
Residual Add
  │
  ▼
RMSNorm
  │
  ▼
SwiGLU MLP
  │
  ▼
Residual Add
  │
  ▼
Output
```

Mathematically:

```text
h'    = h + Attention(RMSNorm(h))
h_out = h' + MLP(RMSNorm(h'))
```

This is a **Pre-Norm Transformer** — pre-norm is chosen over post-norm
specifically because it is more stable to train without a learning-rate
warmup tuned to the exact model depth, which matters for a project with a
single-shot compute budget and no room for repeated failed-training-run
costs.

---

## 7. Multi-Head Self-Attention

### 7.1 Configuration

```text
Hidden Dimension: 768
Attention Heads:  12
Head Dimension:   64      (768 / 12)
```

### 7.2 Projection Matrices (no bias)

| Projection | Shape |
|---|---|
| Query `Wq` | `[768, 768]` |
| Key `Wk` | `[768, 768]` |
| Value `Wv` | `[768, 768]` |
| Output `Wo` | `[768, 768]` |

```text
Input [B, 1024, 768]
      ├──► Q: [B, 1024, 768]
      ├──► K: [B, 1024, 768]
      └──► V: [B, 1024, 768]

Reshaped: [B, 12, 1024, 64]
```

### 7.3 Causal Attention

```text
Token 1 → attends to Token 1
Token 2 → attends to Tokens 1–2
...
Token N → attends to Tokens 1–N
```

```text
Attention(Q, K, V) = softmax( (QKᵀ / √d_head) + CausalMask ) V
```

### 7.4 Implementation Note

Use a fused scaled-dot-product-attention kernel (PyTorch
`torch.nn.functional.scaled_dot_product_attention` with the causal flag, or
FlashAttention if available for the target GPU/PyTorch version) rather than
a manual `QKᵀ` + mask + softmax implementation. For a 1,024-token context
this is not strictly required for memory (quadratic cost at this length is
small — `1024² × 12 heads` is trivial for 24GB VRAM), but it materially
reduces wall-clock training time and therefore cost, at zero numerical-risk
cost since these kernels are causal-mask-aware and well-tested.

---

## 8. Feed-Forward Network: SwiGLU

```text
Input
  │
  ├──────────────► Linear (W_gate) ──► SiLU
  │
  └──────────────► Linear (W_up)
                          │
                          ▼
                     Elementwise ×
                          │
                          ▼
                  Linear (W_down) → Output
```

```text
SwiGLU(x) = (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down
```

```text
d_ff = 2,048     (not 4× d_model — SwiGLU's 3-matrix structure means the
                  effective compute-equivalent expansion at d_ff=2048 is
                  already close to a standard 4×-d_model 2-matrix MLP;
                  sizing d_ff at a plain 4×768=3072 while also using the
                  3-matrix SwiGLU is what caused the original param-budget
                  overshoot in Section 0.)
```

No bias on any of `W_gate`, `W_up`, `W_down`.

---

## 9. Normalization Strategy

```text
RMSNorm → Attention → Residual Add
RMSNorm → MLP       → Residual Add
```

```text
Normalization: RMSNorm (scale-only, no bias, no mean-subtraction)
Epsilon:       1e-5
```

RMSNorm is used for reduced normalization overhead vs LayerNorm and
compatibility with modern decoder-only architectures (Llama-family
convention).

---

## 10. Tokenizer Design

- **Algorithm:** Byte-level BPE (guarantees no `<unk>` — any input, including
  unusual code symbols or non-English text accidentally present in a source
  document, decomposes to bytes in the worst case).
- **Vocabulary size:** 15,000 (see Section 2.2 for sizing rationale).
- **Special tokens (reserved IDs, not counted against the merge budget):**
  `<bos>`, `<eos>`, `<pad>`, `<doc>` (document-boundary marker used during
  packing, Section 13). 4 reserved slots is enough; do not add more without
  re-deriving the parameter count in Section 3.
- **Training corpus for the tokenizer itself:** a representative sample of
  the *final* cleaned/deduplicated corpus (Section 11), not a generic
  English corpus — the tokenizer must be fit on the actual domain
  distribution (code + technical prose) to hit good compression at 15,000
  tokens.
- **Code-awareness:** preserve leading whitespace/indentation as part of
  token boundaries (standard practice for code-inclusive tokenizers,
  e.g. GPT-2/GPT-Neo style byte-level BPE) rather than normalizing it away —
  indentation is semantically meaningful in Python and pseudocode, which is
  a chunk of the target domain.
- **Validation gate before freezing:** measure bytes-per-token on a held-out
  slice of each of the 10 subject domains individually. If any one subject
  compresses noticeably worse than the others (indicating the tokenizer
  under-serves it), that is a signal to rebalance the tokenizer training
  sample, not to silently proceed.
- **Measured (small-corpus tokenizer, `tokenizer/bytes_per_token.json`):**
  3.29–4.57 bytes/token across the 8 small-corpus subjects present so far
  (postgresql/ods lowest at ~3.3, ietf_rfcs/wikipedia highest at ~4.57 —
  expected, since RFC/Wikipedia prose has longer natural-language runs than
  terse docs). **Fertility rate (tokens/word)**, measured directly on a
  CS-terminology-dense sample: **1.32 tokens/word**. Common single-concept
  terms the tokenizer has learned as one token: `asynchronous`, `malloc`,
  `struct`, `deadlock`. Morphologically composed terms still split:
  `polymorphism` → 4 tokens, `semaphore`/`scheduler` → 2 tokens each. This
  will be re-measured once the tokenizer is retrained on the full large
  corpus (§0 revision note) — the number above is from the current small-
  corpus tokenizer, not the final one.

---

## 11. Data Sourcing and Licensing

**Current status: resolved for both corpora.** v1 (1.151B tokens: Wikipedia
CS, ArXiv CS, The Stack, plus the small permissively-licensed sources) is
collected, deduplicated, contamination-checked, packed, and has trained
three full runs (`baseline`, `loop_6x2`, `loop_4x3`). Corpus v2 (4.296B
tokens, Section 11.5) is collected, filtered, decontaminated, packed, and
queued for the scaling runs. This section remains the required process —
the gate it describes is what both corpora were actually run through, not
a hypothetical.

### 11.1 Why This Section Exists

Textbooks and lecture notes are copyrighted by default. "It's for an
academic project" is not, by itself, a license. Before any document enters
the training corpus, it must pass through a recorded license check.
Getting this wrong doesn't just create legal risk for the project — for a
public GitHub repo it's also reputationally visible in a way a private
training bug is not.

### 11.2 Required License Ledger

Every source must be logged, before ingestion, in a simple ledger (a CSV or
markdown table checked into the repo under `data/SOURCES.md` once
implementation starts):

| Field | Example |
|---|---|
| Source name | "Open Data Structures (Pat Morin)" |
| URL | (source URL) |
| License | CC-BY / CC-BY-SA / MIT / public domain / permission-granted |
| Scope of use | "training corpus, non-commercial academic project" |
| Date checked | 2026-09-06 |
| Decision | include / exclude / pending-review |

No source is added to the tokenizer-training or model-training pipeline
without a `decision: include` row backed by a license that permits this use.
When in doubt, exclude — the corpus does not need to be large to be useful
for a 96.5M-parameter domain model (see Section 16.3 on token budget vs.
corpus size).

### 11.3 Candidate Source Categories (by license risk, lowest first)

These are categories to evaluate, not a pre-approved list — each specific
source still goes through the ledger in 11.2.

1. **Openly-licensed existing pretraining corpora, pre-filtered by domain**
   — e.g. permissively-licensed code corpora that already do license
   filtering per-file (only ingest the already-permissive subset), and
   open-access academic paper corpora (CC-BY / ODC-BY abstracts and papers)
   filtered to CS categories. Lowest risk — licensing was already resolved
   upstream, and it's just a matter of confirming the redistribution terms
   for a derived-training-data use.
2. **Official language/tool documentation** with an explicit open license
   (many official docs are released under permissive terms — verify
   per-project, do not assume).
3. **Wikipedia / Wikibooks** (CC-BY-SA) — attribution and share-alike terms
   apply; fine for training-data use, note the license in the ledger.
4. **Permissively-licensed open-source code** (MIT/BSD/Apache-tagged
   repositories only) for the code/pseudocode generation objective — filter
   by repository license metadata before inclusion, not by manual spot-check.
5. **Q&A / community content with explicit content licenses** (e.g.
   CC-BY-SA-licensed dumps) — attribution requirements apply.
6. **Textbooks, scanned lecture slides, proprietary course material** —
   **excluded by default.** Only include with an explicit, recorded
   permission grant from the rights holder. Do not scrape or bulk-download
   textbook PDFs on the assumption that academic use is automatically fair
   use — it is not a blanket exemption, and the ledger decision here should
   default to `exclude` unless a specific permission is obtained.

### 11.4 Deduplication and Contamination

- **Near-duplicate removal:** hash-based exact-dedup first (cheap), then
  MinHash/LSH near-duplicate removal across documents — technical
  documentation and code samples in particular tend to contain large
  near-duplicate blocks (boilerplate, repeated examples) that inflate
  token counts without adding signal.
- **Eval contamination check:** before finalizing the training corpus,
  n-gram overlap check against the held-out evaluation set (Section 24) —
  if benchmark questions or their source passages leak into training data,
  the MCQ/eval numbers in Section 23 become meaningless. This check must run
  and be recorded as passed before a full training run starts, not after.

### 11.5 Corpus v2 — data scaling (implemented, `scripts/expand_corpus.py` + `scripts/run_corpus_v2.sh`)

v1's 1.151B tokens is small for the 164.8M-parameter scaling point (Section
31.4) and limits how many epochs the baseline/loop runs can take before the
early-stopping guard (Section 21.5) fires. Corpus v2 is a second, independent
pipeline built to add data without touching v1 or re-running the tokenizer
(Section 21.6 forbids a vocabulary change mid-project; v2 reuses the frozen
v1 tokenizer).

**Sources** (all openly licensed, pulled via the Hugging Face `datasets`
library, extracted on a CPU/data box so the GPU instance is never blocked on
this work):

| Source | Train tokens | Cap | License |
|---|---:|---:|---|
| FineWeb-Edu dedup (`HuggingFaceTB/smollm-corpus`) | 1.194B | 1.2B | ODC-By (CommonCrawl terms) |
| Stack Exchange (`common-pile/stackexchange_filtered`) | 1.194B | 1.2B | CC BY-SA (per-document license in metadata) |
| Cosmopedia v2 (`HuggingFaceTB/smollm-corpus`) | 0.722B | 0.8B | ODC-By (synthetic) |
| arXiv papers (`common-pile/arxiv_papers_filtered`) | 0.032B | 0.2B | Open license per paper |
| LibreTexts (`common-pile/libretexts_filtered`) | 0.5M | uncapped | Open license per page |

**Not a contradiction with `data/SOURCES.md`'s Stack Overflow exclusion:**
that v1 decision (Section 11.3 category 6 reasoning) excluded *directly
scraping* Stack Overflow, because Stack Exchange's own terms of service add
bulk-use restrictions beyond the CC-BY-SA license text, and resolving that
ambiguity per-document was out of scope for this project. `stackexchange_filtered`
is Common Pile's own re-release of Stack Exchange dumps, filtered and
redistributed by that project under CC BY-SA with per-document attribution
already resolved — exactly Section 11.3 category 1 ("licensing was already
resolved upstream"), not category 4/6. The v1 ledger entry stays as written
(this project still does not scrape Stack Overflow directly); v2 uses a
different, lower-risk upstream source for a related but distinct category
of content.

**Filtering.** Every candidate document is scored against a CS keyword
lexicon (`CS_TERMS` plus `WEAK_TERMS`, the latter capped at 2 points so a
document cannot pass purely on a handful of generic words) via a trie-based
regex, with a cheap anchor-term prefilter before the full score is computed.
Each source has its own acceptance threshold (fineweb 6, cosmopedia 6,
stackexchange 0 — already pre-filtered by site, arxiv 12, libretexts 6).
Cosmopedia additionally rejects story/dialogue-formatted documents and
children/young-children/alien-audience documents, and explicitly accepts
anything with a `"COMPUTERS / "` BISAC label. Stack Exchange is restricted to
a fixed list of CS-relevant sites (`SE_SITES`), with `stackoverflow.com`
itself downsampled (kept 1-in-6) since it would otherwise dominate the
source. FineWeb additionally excludes Wikipedia-sourced pages (already
covered by v1). A document below `MIN_DOC_TOKENS` (64) or `MIN_WORDS` (150)
is dropped regardless of score.

**Decontamination.** A kept document is dropped if it shares any 13-gram
with a benchmark item (`eval/benchmark.json` or `eval/mmlu_cs_benchmark.json`
— the same contamination check as Section 11.4, applied to a different
corpus), or if 20% or more of its 13-grams overlap the v1 validation split
(`VAL_OVERLAP_MAX`) — this second check exists specifically because v2 is
*additive* to v1 and must not quietly leak into the validation set that v1's
own runs are scored against.

**Packing.** Exact-duplicate documents are dropped. Per-source token caps
(table above) are applied in a random permutation of each source's
documents, so a cap does not systematically favor whichever document
happened to be extracted first. All sources are concatenated, globally
shuffled, and packed into fixed-length shards with the frozen v1 tokenizer;
0.5% of the newly-kept tokens are held out as `val_new` before packing, not
sampled after, so `val_new` cannot share a shard with anything used in
training. The v1 train shards are copied in unchanged (server-side S3 copy,
no re-tokenization) so `shards_v2/train` is a strict superset of
`shards/train`.

**Result (run completed 2026-10-30):** `data/shards_v2/train` holds 1025
shards, 4,296,117,248 tokens (750 new shards at 3.145B tokens, plus the 275
v1 shards at 1.151B). `val_new` is 4 shards, 15,826,944 tokens. At 488
sequences of 1024 tokens per training step (Section 16.3's effective batch),
one full epoch is 8,595 steps — the same formula gives 2,302 steps for v1,
which matches the step count the `baseline` run actually stopped at,
confirming the formula against a real run rather than trusting it
unverified.

The whole pipeline is resumable (state kept under
a work prefix in the bucket) and runs unattended: the data box
extracts each source in turn, packs, uploads `shards_v2/train`, `val_new`,
and `manifest_new.json`, writes a done marker, and
terminates itself. 16 tests (`tests/test_expand_corpus.py`) cover the
filtering, decontamination, and packing logic directly.

---

## 12. Data Processing and Tokenization Pipeline

```text
┌─────────────────────────────┐
│ Source Documents            │
│ (license-cleared only,      │
│  Section 11)                │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Extraction & Normalization  │
│ (strip markup, normalize    │
│  whitespace, keep code      │
│  indentation intact)        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Dedup & Contamination Filter│
│ (Section 11.4)              │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Byte-Level BPE Tokenizer    │
│ Vocabulary: 15,000          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Continuous Token Stream     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Fixed-Length Packing        │
│ Sequence Length: 1,024      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Flat Binary Shards (memmap) │
│ [uint16 token ids]          │
└─────────────────────────────┘
```

Token IDs are stored as `uint16` (vocabulary 15,000 fits comfortably under
the 65,536 ceiling), memory-mapped at train time. This avoids re-tokenizing
on every run and keeps the dataloader CPU/RAM footprint low — which is the
basis for right-sizing the training instance down in Section 14.

---

## 13. Token Packing and Document Boundaries

```text
Document A → <doc> → Document B → <doc> → Document C → <doc> → ...
    │
    ▼
Concatenated Token Stream
    │
    ▼
[Token 1 ... Token 1024]
[Token 1025 ... Token 2048]
...
```

- A `<doc>` boundary token is inserted between documents. This is a
  deliberate choice, not the "alternative" mentioned loosely in prior
  drafts: pure continuous packing without boundaries lets the model learn
  to blend unrelated documents' endings/beginnings, which is undesirable for
  a QA/explanation-oriented model where document identity matters.
- Sequences may span a document boundary (`<doc>` appears mid-sequence) —
  this is expected and fine; it is exactly how GPT-2/GPT-3-style packing
  works and is why the boundary token exists rather than padding to
  document edges (padding would waste a meaningful fraction of tokens at
  this sequence length).
- The last partial sequence in the corpus (a leftover chunk shorter than
  1,024 tokens) is dropped from the packed shards rather than padded — with
  a corpus that is likely small relative to typical LLM pretraining corpora,
  losing one final partial chunk per shard is negligible.

---

## 14. Training Infrastructure

### 14.1 Actual Compute Target (as implemented)

```text
AWS EC2 G6 Spot Instance
Instance type: g6.xlarge   (1x NVIDIA L4, 4 vCPU, 16 GB RAM)
GPU:           NVIDIA L4, 24 GB VRAM
```

The original draft specified `g5.xlarge` (1x A10G). The implementation uses
`g6.xlarge` (1x L4) instead: at the time training actually started, L4 spot
capacity was available in us-east-1 under the project's approved spot G
vCPU quota (4), where A10G capacity (`g5`) was not reliably available
across AZs (see §29, risks 1 and 7, and the incidents in the run log (not in this release)). Both GPUs are in the
same tier for this workload; the instance-right-sizing argument from the
original draft (dataloader is pre-tokenized `uint16` memmap, never the
bottleneck, so extra vCPU/RAM from a larger instance buys nothing) applies
identically to `g6.xlarge` vs `g6.2xlarge`. Measured throughput on this GPU
(Section 19.1): 17.2 s/step eager, 10.3 s/step with `torch.compile` (1.67x),
for the 39.8M `loop_4x3` config at the standard 499,712-token effective
batch — the actual replacement for the formula-based estimates in Section
19.2.

### 14.2 Infrastructure Layout (as implemented)

```text
┌──────────────────────────────────────┐
│ AWS EC2 g6.xlarge Spot Instance       │
│ (us-east-1d, persistent spot request) │
│                                       │
│ NVIDIA L4 — 24 GB VRAM                │
│ PyTorch 2.14 + torch.compile/Triton   │
│ BF16 Mixed Precision                  │
│                                       │
│ astra-resume.service (systemd)        │
│  -> queue script         (Section     │
│     14.4): reads jobs.json, trains    │
│     each pending job, evals, archives │
└──────────────────┬────────────────────┘
                   │ per-job checkpoint, train.log, eval sync
                   ▼
┌───────────────────────────────────────┐
│ Amazon S3 (s3://your-bucket-name)│
│ • data/shards/, data/shards_v2/        │
│ • checkpoints/<run>/ (periodic/best/   │
│   final; baseline unprefixed)          │
│ • results/<run>/ (val loss, perplexity,│
│   MCQ, per-subject breakdown)          │
│ • job list, queue-state file,          │
│   done and failed markers              │
└───────────────────────────────────────┘
```

### 14.3 EBS Volume

A single `gp3` root volume, sized to (packed token shards + OS + checkpoint
scratch space). Given the corpus is expected to be small relative to typical
LLM pretraining corpora (Section 11), 50 GB `gp3` is a safe starting size;
this is a small line item and not a place to
over-provision "just in case."

### 14.4 Operational infrastructure (as implemented)

The implemented system is three independent pieces, because two failure
modes need covering: a laptop that goes to sleep, and a spot instance
that gets reclaimed:

- **Queue v3** (the queue script, run by `astra-resume.service` on
  every boot): reads its job list from the bucket, a flat list of
  `{name, config, steps, micro_bs, train_shards, eval, extra_args}` jobs.
  Re-reads the list before each job, so jobs can be appended mid-run. Skips
  a job already marked `done` or `failed` (S3 markers, checked before the
  older local markers), runs it via the bootstrap script (not in this release), retries a
  non-zero exit once, then marks it `failed` and moves to the next job. A
  job that exits via SIGTERM (spot reclaim, reboot, an explicit stop) is
  left unmarked and resumes from its newest S3 checkpoint next boot — this
  is the mechanism, not early-stopping or a Lambda, that makes the queue
  tolerant of spot interruption. When the list is exhausted it records
  `done` and shuts the instance down; while idle (list temporarily empty)
  it polls for up to `idle_limit_minutes` before giving up, so a job
  appended a few minutes late is still picked up without a manual restart.
  `DRY_RUN=1` prints the next job without running it, for validating a
  freshly-edited `jobs.json` before trusting it. Covered by 7 tests
  (queue tests, not in this release) that run the real script under `bash` against
  fakes of `aws`/the bootstrap/the evaluator.
- **Watchdog** (a watchdog function, source not in this release,
  EventBridge-scheduled every 20 minutes, capped at 30 restarts): starts the
  target instance if it is stopped and the queue has not recorded `done` or
  `failed`. This is what lets the laptop sleep — nothing
  on the laptop is in the loop between a spot reclaim and the next restart.
- **Launcher** (a launcher function, source not in this release,
  EventBridge-scheduled every 5 minutes with a fixed deadline): retries a
  spot launch when capacity is briefly unavailable in every AZ, repoints the
  watchdog at the new instance once a launch succeeds, clears
  the queue-state file, and disables its own schedule — so it runs itself out
  of a job instead of polling forever.
- **Stop handler** (source not in this release, and
  a budget-stop function): an
  optional AWS-side stop for instances tagged `Project=AstraCSE`. When it
  fires, it disables the watchdog and launcher rules first, writes
  `budget_stop` to the queue-state file (the queue never overwrites that
  state), and stops each tagged instance, so one failure does not block
  the others. Its alarm thresholds are account-specific and are not
  recorded here.

---

## 15. Precision and Memory Strategy

```text
Compute precision:      BF16 (PyTorch native AMP / autocast)
Optimizer master state: FP32 (AdamW moments + master weights)
Framework:              PyTorch
```

### 15.1 Memory Budget (training, rough order-of-magnitude)

| Component | Approx. size |
|---|---|
| Model weights (BF16) | ~193 MB (96.47M × 2 bytes) |
| Model weights (FP32 master, for optimizer) | ~386 MB |
| Gradients (FP32) | ~386 MB |
| AdamW moments (2×, FP32) | ~772 MB |
| Activations (batch/seq-dependent) | variable — the actual lever for batch size |

Fixed (non-activation) memory is under ~1.8 GB — trivial against 24 GB
VRAM. This means batch size is essentially activation-memory-bound, not
weight/optimizer-bound, and there is comfortable headroom to use a large
micro-batch before needing gradient accumulation. Exact safe micro-batch
size must still be found empirically (start high, halve on OOM) in Phase 0
(Section 22) rather than calculated in the abstract, since activation memory
depends on the exact attention-kernel implementation in use.

**Verified against the actual implementation** (`src/train.py`): the model
is moved to device with no dtype cast (`AstraCSE(cfg).to(device)`), so
weights and gradients live in FP32 throughout; `torch.autocast(dtype=
torch.bfloat16)` casts activations/matmul inputs to BF16 *on the fly* inside
the forward/backward context only — it does not hold a second permanent
BF16 copy of every weight. The "Model weights (BF16) ~193MB" line above is
therefore a transient, not a standing, cost — the ~1.8GB figure is already
a safe overestimate, not an underestimate. (A different implementation that
explicitly cast the whole model to BF16 for compute while keeping a
separate FP32 master copy for the optimizer would need both lines
simultaneously — that is not what this codebase does.)

Same accounting for the loop-transformer variants (§31), scaled by their
unique-parameter count instead of 96.47M:

| Config | Unique params | Fixed memory (weights+grads+AdamW, FP32) |
|---|---|---|
| Baseline (12×1) | 96,473,856 | ~1.44 GB |
| 6×2 | 54,006,528 | ~0.81 GB |
| 4×3 | 39,847,680 | ~0.59 GB |
| 3×4 | 32,768,256 | ~0.49 GB |

All four remain trivial against 24GB VRAM — parameter efficiency does not
meaningfully change the training memory story at this scale (§31.3: the
real benefit is checkpoint/deployment footprint, not training-time VRAM,
since activation memory — which scales with effective depth, identical
across all four — dominates regardless).

### 15.2 BF16 Validation — confirmed

BF16 autocast runs on the actual instance (`g6.xlarge`, NVIDIA L4, Section
14.1) without the fallback ever being needed: three full runs
(`baseline`, `loop_6x2`, `loop_4x3`) and the Phase 0 debug run have
completed with BF16 compute / FP32 optimizer, zero NaN/Inf halts (Section
21.1) across any of them, and `torch.compile` measured a further 1.67x
speedup on top (Section 19.1) with identical loss to 4 decimals versus
eager — so neither the FP16-loss-scaler fallback nor a compile-correctness
concern has come up in practice. BF16 remains preferred over FP16 for the
reason stated below regardless.

---

## 16. Optimizer and Training Hyperparameters

This section was entirely missing from the previous draft — without it, an
implementer would be guessing at learning rate, batch size, and stopping
criteria, which is exactly the kind of gap that turns into a wasted paid
training run.

### 16.1 Optimizer

```text
Optimizer:      AdamW
betas:          (0.9, 0.95)
eps:            1e-8
weight_decay:   0.1   (applied to matrix weights only —
                        excluded from RMSNorm scale params,
                        since decaying those destabilizes normalization)
grad_clip:      global norm clip at 1.0
```

### 16.2 Learning Rate Schedule

```text
Schedule:      Linear warmup → Cosine decay
Warmup steps:  ~2% of total planned steps (recompute once total steps is set)
Peak LR:       start at 6e-4  (typical for ~100M-scale decoder-only models)
Min LR:        10% of peak (6e-5)
```

**Resolved for the dense baseline:** 6e-4 was carried into the full
`baseline`, `loop_6x2`, and `loop_4x3` runs (all converged without
divergence, loss spikes, or NaNs — Section 21.1/21.2 never fired across any
of them) and is the value in `configs/model.yaml`. It is not re-derived for
every width: `model_d1024.yaml` (d_model 1024) uses 5e-4, between GPT-3's
6e-4 (125M) and 3e-4 (350M) conventions, and `model_d1024_ctx2048.yaml`
(continued pre-training from a trained checkpoint) uses a much lower 1e-4,
since the model is no longer starting from random init.

**Open question for the loop variant:** 6e-4 was tuned against the dense
baseline's gradient scale, not the loop variant's. A shared block's gradient
is the *sum* of the gradients from every depth slot that reuses it (Section
31.1), so its effective per-step update is larger than an unshared block's
at the same nominal LR — the loop variant may need a lower peak LR purely
for this reason, independent of whether weight sharing itself hurts quality.
`configs/model_loop_6x2_lr3e-4.yaml` (peak LR halved, everything else
identical to `model_loop_6x2.yaml`) was run specifically to test this (done; see Section 31.7)
before committing a peak LR for the larger-scale loop runs on corpus v2
(`loop6x2_v2`, `d1024_loop6x2_v2`).

### 16.3 Batch Size and Token Budget

```text
Sequence length:        1,024 tokens
Target effective batch: ~0.5M tokens/optimizer-step (typical for this scale)
```

Effective batch is achieved via `micro_batch_size × gradient_accumulation_steps`.
Micro-batch size is determined empirically in Phase 0 against the 24 GB
VRAM ceiling (Section 15.1); gradient accumulation makes up the rest to
reach the target effective batch.

**Total training tokens is corpus-limited, not compute-limited, for this
project.** The available GPU time (Section 19) comfortably supports far more tokens
than a from-scratch-collected, license-cleared, 10-subject CS corpus is
likely to contain in a single pass (Section 11 status: not yet collected).
The practical plan is:

1. Measure the actual unique token count of the cleaned, packed corpus once
   Section 11 is complete.
2. If unique tokens ≥ ~2B, train for roughly one Chinchilla-style pass
   (~20 tokens/parameter ≈ 1.9B tokens) as a floor, and continue with
   additional epochs only if held-out validation loss keeps improving.
3. If unique tokens are well under 1B (the more likely outcome for a
   from-scratch, license-cleared, single-author-collected corpus), plan for
   multiple epochs (2–5×) over the corpus, with the overfitting guard in
   Section 21.5 (early stopping on held-out loss) as the hard stop — repeat
   epochs are fine until validation loss stops improving, they are not fine
   past that point.

This corpus-size unknown is the single biggest open variable in the whole
plan (tracked in Section 29, Risk 1) — the throughput model in Section 19 is
therefore expressed as a formula against measured throughput and measured
corpus size, not a fixed promised number.

**Which parameter count for the D=20N Chinchilla target, once §31 exists:**
use the baseline's 96.47M (equivalently, effective depth × per-layer cost —
identical across all four §31 configs), not each loop-variant's smaller
unique-parameter count. Chinchilla's N is a proxy for FLOPs/token, and
FLOPs/token is matched across all four configs by construction (§31.3) —
using the unique-parameter count instead would incorrectly suggest the
33M-parameter loop variant needs less data than the 96M baseline, when in
fact it does the same amount of compute per token and should be fed the
same token budget. One side-effect worth stating plainly in the eventual
write-up: at a fixed corpus size, the loop variants receive *more* training
signal per unique parameter than the baseline does, purely because they
have fewer unique parameters absorbing the same token stream — this is a
natural, expected consequence of parameter sharing, not a separate finding.

---

## 17. Spot Instance Fault Tolerance

AWS Spot instances may be interrupted with a 2-minute notice. Checkpointing
is a first-class production requirement, not an afterthought.

### 17.1 Checkpoint Frequency

At least every 1,000 training steps, containing:

```text
checkpoint/
├── model_state
├── optimizer_state
├── scheduler_state
├── global_step
├── epoch
├── rng_state          (python, numpy, torch, cuda)
├── dataloader_state    (shard index + intra-shard offset)
└── training_metadata   (config hash, tokenizer version, timestamp)
```

`dataloader_state` matters specifically because resuming without it
silently restarts the data stream from the beginning of the current epoch —
harmless for loss correctness (packed shards are order-agnostic-ish over a
full epoch) but it does distort the token-budget accounting in Section 16.3
if interruptions are frequent, so it should be tracked even though it is not
safety-critical.

### 17.2 Checkpoint Synchronization

```text
Training Process → Local Checkpoint → Atomic Upload → Amazon S3
```

- Write to a temporary S3 key first, verify upload integrity (checksum),
  then atomically update the `latest` pointer only after verification
  succeeds. A checkpoint that fails upload must never become `latest`.

```text
checkpoints/
├── step-001000/
├── step-002000/
├── step-003000/
└── latest/          (pointer, updated only post-verification)
```

**As implemented:** there is no separate `latest/` pointer object — an
earlier version of resume read one and it was never written, so resume
silently restarted the wrong way (the run log (not in this release) incidents, fixed in commit
`e2c28c8`). The actual mechanism is simpler and has no second object to get
out of sync: resume lists everything under the run's S3 checkpoint prefix
and picks the checkpoint with the highest step number
(`newest_final_key`/the `step-(\d{7})\.pt` pattern in `src/train.py`),
preferring a `-final` checkpoint only at the end of a run. Periodic
checkpoints beyond `--keep-local` are pruned locally (not in S3) once
superseded, so a long run does not fill the instance's disk.

---

## 18. Interruption Recovery Protocol

```text
AWS Spot Interruption Notice (2-min warning)
          │
          ▼
SIGTERM Listener
          │
          ▼
Stop New Training Step (finish current step, do not start another)
          │
          ▼
Save Emergency Checkpoint
          │
          ▼
Upload to S3 (best-effort within the 2-minute window)
          │
          ▼
Graceful Shutdown
```

On restart:

```text
New EC2 Instance → Bootstrap Script → Download Latest Verified Checkpoint
       │
       ▼
Restore: Model, Optimizer, Scheduler, RNG State, Global Step, Dataloader State
       │
       ▼
Resume Training
```

The recovery process must be idempotent and must log the resumed
`global_step` explicitly on startup — a resume that silently restarts from
step 0 is a cost-and-correctness bug, and the startup log line is the cheap
guard that makes it immediately visible if it happens.

**Edge case — emergency checkpoint fails to finish uploading within the
2-minute window:** the bootstrap script must fall back to the last fully
verified `latest` checkpoint (potentially up to ~1,000 steps stale) rather
than blocking on or trusting a partial upload. Losing up to 1,000 steps of
progress on an interruption is an acceptable, budgeted cost; resuming from a
corrupted or partial checkpoint is not.

**As implemented and tested:** `astra-resume.service` runs
the queue script (not in this release) with `Type=simple`, `KillMode=control-group` under
systemd, and the queue traps `TERM` itself (`trap 'INTERRUPTED=1; exit 143'
TERM`) rather than relying only on `train.py`'s own handler — a service
restart, a spot reclaim, and a reboot all deliver the same signal path.
`train.py` saves a `sigterm-step-<N>.pt` checkpoint on `TERM` and exits 143;
the queue recognizes exit 143 specifically and leaves the job unmarked
(neither `done` nor `failed`) so it resumes rather than retrying from
scratch. This path has been exercised many times in practice, not just in
the kill-drill below: every one of `loop_6x2`'s and `loop_4x3`'s systemd
restarts resumed correctly from its newest `sigterm-*` checkpoint (three
such resumes logged for `loop_4x3` alone by 2026-10-30), and the original
Oct 28 debug-run kill-and-resume test (Section 22.1 item 4) passed cleanly.
The one real incident was at the infrastructure layer, not the checkpoint
layer: terminating an instance without first cancelling its *persistent*
spot request let AWS re-fulfil the request with the original (stale)
launch spec, producing an unexpected second instance that held the spot
vCPU quota (the run log (not in this release) incidents) — the lesson recorded there is to
cancel the request before ever terminating its instance.

---

## 19. Compute Throughput and Usage

### 19.1 Measured Throughput (replaces the formula estimate)

The formula `FLOPs ≈ 6 × N_params × D_tokens` and the MFU-scenario table
below were the pre-training planning input; Phase 0 (Section 22) and the
full runs since have replaced them with measurements on the actual
`g6.xlarge` / NVIDIA L4 instance (Section 14.1):

| Config | Params | Precision/mode | Measured speed |
|---|---:|---|---:|
| `baseline` (12 layers) | 96.47M | BF16, eager | measured in the full run; see the run log (not in this release) |
| `loop_6x2` (6×2) | 54.01M | BF16, eager | ~17.4 s/step (effective batch 499,712 tokens) |
| `loop_4x3` (4×3) | 39.85M | BF16, eager | 17.2 s/step |
| `loop_4x3` (4×3) | 39.85M | BF16, `torch.compile` | 10.3 s/step — **1.67x faster**, same-batch loss identical to 4 decimals vs eager (`scripts/bench_compile.py`, and cross-checked live at the eager/compiled boundary of the real run) |

Every job queued after this measurement runs with `--compile`. The 1.67x
factor is the real replacement for an assumed MFU percentage — it is not a
FLOPs-utilization claim, just a measured wall-clock ratio on this exact
model/GPU/PyTorch-2.14/Triton-3.8 combination.

### 19.2 Historical Illustrative Scenarios (superseded by 19.1, kept for context)

The original planning table assumed `g5.xlarge` (A10G, ~125 TFLOPS dense
BF16 peak) and a range of MFU guesses, since no measurement existed yet:

| MFU scenario | Effective TFLOPS | Est. tokens/sec | Time for 2B tokens | Time for 4B tokens |
|---|---:|---:|---:|---:|
| Conservative (15%) | 18.75 | ~32,000 | ~17.4 hrs | ~34.7 hrs |
| Moderate (25%) | 31.25 | ~54,000 | ~10.3 hrs | ~20.6 hrs |
| Optimistic (35%) | 43.75 | ~75,500 | ~7.4 hrs | ~14.7 hrs |

The instance that actually ran training is `g6.xlarge` (L4), not `g5.xlarge`
(Section 14.1), so this table's TFLOPS basis no longer applies; Section
19.1's measured s/step numbers are the authoritative figures for planning the
remaining queue.

### 19.3 GPU Time Used

All 19 runs and their evaluations used one NVIDIA L4 GPU for about 250 active
GPU-hours in total, from the run logs. Interruptions (spot reclaims and restarts)
add wall-clock time but no GPU time, because a stopped instance does not run. The
step counts and the measured speeds in Section 19.1 give the expected duration of
each run; the run log (not in this release) records the actual timeline.

---

## 20. Training Objective

Autoregressive next-token prediction. Given `x₁, ..., xₙ`, the model learns
`P(xₜ | x₁, ..., xₜ₋₁)` via causal language-model cross-entropy:

```text
L = -Σ log P(xₜ | x₁, ..., xₜ₋₁)
```

The model attends only to previous tokens (Section 7.3).

---

## 21. Training Failure Modes and Edge Cases

This is the section most conspicuously missing from the previous draft.
Production training on an unattended Spot instance needs explicit handling
for these, or a failure silently burns GPU-hours producing a
broken checkpoint.

### 21.1 Loss Divergence: NaN/Inf

- **Detection:** check `loss.isfinite()` every step, before the optimizer
  step is applied.
- **Response:** skip the optimizer step for that batch, log the step index
  and a hash of the offending batch, increment a consecutive-NaN counter.
- **Escalation:** if NaN/Inf occurs on 3 consecutive steps, halt training
  and trigger the emergency-checkpoint path (Section 18) rather than
  continuing to burn compute on a diverged run — this is a case where
  automated recovery should stop and surface the problem, not "retry
  forever."
- BF16 (vs FP16) is chosen specifically because it removes the
  loss-scale-underflow/overflow failure mode that FP16 training is prone to
  (Section 15.2) — this is a preventative measure for this exact risk.

### 21.2 Loss Spikes (non-fatal but degrading)

- **Detection:** loss > 3× the trailing 100-step moving average.
- **Response:** log and continue by default (transient spikes are normal),
  but if spikes recur more than 5 times in any 500-step window, that is a
  signal to lower the learning rate on resume from the last good
  checkpoint, not something to silently ride out to the end of the run.

### 21.3 Out-of-Memory (OOM)

- **Detection:** catch `torch.cuda.OutOfMemoryError` around the training
  step.
- **Response:** clear the CUDA cache, halve the micro-batch size (increasing
  gradient-accumulation steps to compensate and keep the effective batch
  size in Section 16.3 constant), log the adjustment, and resume from the
  last checkpoint rather than crashing the whole job.
- **Prevention:** Phase 0 (Section 22) determines a safe micro-batch size
  empirically with headroom, specifically so this path is a rare safety net
  and not the normal operating mode.

### 21.4 Dataloader / S3 Flakiness

- Memmapped local shards (Section 12) mean the training loop itself has no
  network dependency per-step — S3 is only touched at checkpoint
  boundaries, which minimizes the surface area for this failure mode by
  design.
- Checkpoint upload failures: retry with exponential backoff (e.g. 3
  attempts); if all attempts fail, keep the local checkpoint on the EBS
  volume and continue training (do not halt training over a checkpoint
  upload failure) — the next successful checkpoint will still cover for it,
  and EBS is durable within an availability zone even across a Spot
  reclaim of the instance's *compute*, though not if the volume itself is
  deleted, so the bootstrap script should attempt to reattach the prior
  run's EBS volume before falling back to S3.

### 21.5 Overfitting on a Small/Narrow Corpus

Directly tied to the corpus-size unknown in Section 16.3 and Section 29:

- **Detection:** held-out validation loss (a slice never seen during
  training, carved out before packing) evaluated at every checkpoint.
- **Response:** if validation loss increases for 2 consecutive evaluation
  points while training loss keeps decreasing, that is the early-stopping
  trigger — stop the epoch-repetition plan there, keep the best checkpoint
  by validation loss (not the latest checkpoint), and treat further epochs
  as not worth the additional compute cost.

### 21.6 Tokenizer/Vocabulary Drift

- The tokenizer must be frozen (versioned, hashed) before training starts.
  If the corpus is later expanded (e.g. more sources cleared through the
  Section 11 ledger after the tokenizer was fit), do not silently retrain
  the tokenizer mid-project — a changed vocabulary invalidates all existing
  checkpoints. Treat a tokenizer change as a decision to restart training
  from step 0 with a new run, made explicitly, not as a drop-in swap.

### 21.7 Checkpoint Corruption

- Verify checkpoint integrity (checksum) both on upload (Section 17.2) and
  on download during resume, before attempting to load it into the model —
  a corrupted checkpoint should fail loudly and fall back to the next-older
  verified checkpoint, not crash the training process or silently load
  garbage weights.

---

## 22. Validation-Before-Scale Protocol (Phase 0) — PASSED

**No full-scale, paid training run started before this phase passed.** It
converted the unknowns (real throughput, safe batch size, reasonable LR, a
correctly-wired training loop) into measured facts using small, cheap runs
before committing budget to the long ones that followed.

### 22.1 Phase 0 Checklist — results

1. **Overfit-a-tiny-batch sanity check: passed.** Covered as an automated
   test rather than a one-off manual run (part of the 125-test suite,
   Section 30) — loss on a small fixed batch drops toward zero over a few
   hundred steps.
2. **Short real-data run: passed**, on a debug instance (`<instance-id>`,
   2026-10-28, 50 steps) before any full run launched. Confirmed loss starts
   near `ln(15,000) ≈ 9.62` nats and decreases visibly (Section 3.3 init is
   correctly wired), and gave the first real throughput number, later
   superseded by the full-run measurements in Section 19.1.
3. **LR mini-sweep: resolved at 6e-4** for the dense family (Section 16.2);
   a second, targeted LR check for the loop variant specifically
   (`model_loop_6x2_lr3e-4.yaml`) is queued rather than swept from scratch,
   since the dense value already has three full runs of evidence behind it
   and only the loop variant's shared-gradient effect is in question.
4. **Checkpoint/resume drill: passed**, twice over. The original debug-run
   drill (2026-10-28, resumed cleanly from `sigterm-step-0000456.pt`) and,
   far beyond the minimum bar, dozens of real resumes during the actual
   full runs — every systemd restart and spot interruption since has
   resumed from the newest `sigterm-*` or `step-*` checkpoint without a
   discontinuity (Section 18, "as implemented").
5. **Compute estimate recomputed: done**, using Section 19.1's measured s/step
   rather than an MFU guess. Go/no-go for each subsequent job is now
   implicit in the queue (Section 14.4) rather than a manual per-run
   approval step.

All five passed before the `baseline` full run started on 2026-10-28.

---

## 23. Evaluation Targets and Measured Results

### 23.1 Language Modeling Metrics — target vs. measured

| Metric | Original target | `baseline` (96.47M, v1, 2302 steps) | `loop_6x2` (54.01M, v1, 2304 steps) |
|---|---:|---:|---:|
| Validation Loss | ~1.6–2.1 | 2.853 | 4.622 |
| Perplexity | ~1.8–2.5 | 17.34 | 101.75 |
| MCQ accuracy (650 MMLU-CS) | 60%–70% | 26.9% (175/650) | 28.6% (186/650) |
| Inference Throughput | >95 tokens/sec | not yet measured end-to-end | — |
| Inference VRAM | <850 MB target (Section 26) | not yet measured end-to-end | — |

Both runs land well short of the original validation-loss and MCQ targets.
This was expected once the actual corpus size was known (Section 29, Risk 1
— ~1.15B tokens total, not the multi-billion-token scale the 1.6–2.1 target
implicitly assumed) rather than a sign of a broken run: training loss curves
are smooth with no spikes/NaNs/OOMs in either run (the run log (not in this release)), and
MCQ accuracy sitting close to the 25% four-choice chance level is the
expected outcome for a 32–96M model trained on low single-digit billions of
tokens, not evidence against the implementation. The original targets stay
in this table as the un-walked-back goal; the scaling runs now queued
(corpus v2 at 4.3x the tokens, and the 164.8M `d1024` width increase) are
the planned path toward them, not a redefinition of the target.

These numbers also depend on final dataset size/quality, tokenizer
efficiency, absence of eval contamination (Section 11.4), and total
training tokens (Section 16.3) — track them, don't treat them as a spec to
force by overfitting the eval set.

---

## 24. Domain Evaluation

```text
1. Computer Programming
2. Data Structures
3. Algorithms
4. Operating Systems
5. Computer Networks
6. Database Systems
7. Computer Architecture
8. Theory of Computation
9. Software Engineering
10. Cybersecurity / Related Core Domain
```

The exact subject taxonomy must be frozen before benchmark construction —
and the benchmark question set must be built (or verified as free of
overlap, Section 11.4) *before* the corpus is finalized, so contamination
checking is actually possible.

### 24.1 Evaluation Categories (per subject)

```text
Conceptual Questions → Definition Understanding → Application Questions
→ Code / Pseudocode Tasks → Multi-Step Technical Reasoning → MCQ Evaluation
```

Initial MCQ target: **60%–70%**. The benchmark must distinguish
memorization/retrieval from conceptual understanding, reasoning, and code
correctness — e.g. include paraphrased variants of the same underlying
question to catch memorization vs. genuine understanding.

---

## 25. Inference Architecture

```text
Autoregressive Decoding → KV Cache → Incremental Token Generation
```

```text
Without KV cache: every new token recomputes the full previous sequence.
With KV cache:     previous K/V states are cached; only the new token
                   requires new attention computation.
```

Expose `use_cache=True` in the inference runtime.

**Degeneration guard:** small from-scratch LMs are prone to greedy-decoding
loops (repeating a token or short sequence indefinitely — common inside
generated code, e.g. `x++; x++; x++;`). `src/infer.py::generate()` supports
`repetition_penalty` (CLI default 1.15, library default 1.0/disabled):
already-generated tokens have their logit divided (if positive) or
multiplied (if negative) by the penalty before sampling, matching the
standard HF `generate()` formula. Verified directly in
`tests/test_infer.py::test_repetition_penalty_breaks_greedy_loop` — a
forced-favourite token that would otherwise be picked every single step
under greedy decoding gets suppressed after its first appearance once
penalty > 1.0.

### 25.1 Input-Length Edge Case

Per Section 5.2, inputs exceeding 1,024 tokens are **not silently truncated
by default** — the inference server must explicitly reject or explicitly
truncate-with-a-warning, as a deliberate configuration choice, since silent
truncation of a technical question could quietly drop the part of the
prompt that mattered.

---

## 26. Deployment, Quantization, and Memory Budget

The previous draft stated a `<850 MB` VRAM target without a plan for how to
reach it. At BF16, the raw model alone is ~193 MB — the remaining budget is
consumed by the KV cache, activations, and runtime overhead, all of which
scale with context length and batch size. The concrete plan:

### 26.1 Export Path

```text
Trained checkpoint (BF16, FP32 optimizer state)
        │
        ▼
Strip optimizer/scheduler state — keep model weights only
        │
        ▼
Export for inference:
  • FP16/BF16 weights  (~193 MB) — default deployment target
  • INT8 weight-only quantization (e.g. via bitsandbytes or a
    GGUF/llama.cpp-style conversion) — (~100 MB) — for the tightest
    memory-constrained deployment targets (Section 26.2)
```

### 26.2 Memory Budget Formula

```text
VRAM = f(model weights, precision, context length, batch size, KV cache, runtime overhead)
```

The `<850 MB` target is a claim about a *specific* configuration (single
request, batch size 1, full 1,024-token context, FP16/INT8 weights), not an
unconditional guarantee across all batch sizes and context lengths. Any
reported VRAM number in project documentation or the eventual write-up must
state the exact `(precision, context length, batch size)` triple it was
measured under — an unqualified "<850MB" claim is not a valid benchmark
result.

### 26.3 Deployment Targets

- University servers / local development: FP16/BF16 export, no
  quantization needed (VRAM is not the binding constraint there).
- Small-GPU or client-side/CPU deployment: INT8 export via GGUF +
  llama.cpp-style runtime — this is also the path that removes any ongoing
  GPU-serving cost entirely for a demo/portfolio deployment, which matters
  for the same cost-minimization goal that drives the training-side
  decisions in this document.

---

## 27. Repository and Artifact Structure (Planned)

This section documents the intended repository layout for when
implementation starts. **Nothing under this section has been created yet**
— it is a plan to build against, not a description of current repo state.

```text
AstraCSE/
├── README.md
├── Architecture.md
├── configs/
│   └── model.yaml         (single source of truth for the Section 2 hyperparameters —
│                            code reads this, does not hardcode magic numbers)
├── data/
│   ├── SOURCES.md          (license ledger, Section 11.2)
│   └── (raw/ and processed/ — not committed to git; large-file storage rules
│        added to .gitignore before any data lands here)
├── tokenizer/
│   └── (trained tokenizer artifact + version hash)
├── src/
│   ├── model.py            (architecture, Sections 2–9)
│   ├── data.py             (packing/memmap loader, Sections 12–13)
│   ├── train.py            (training loop, Sections 16–21)
│   └── infer.py            (Section 25–26)
├── scripts/
│   ├── (queue and bootstrap scripts: not in this release)
│   └── (watchdog, launcher and stop handler: not in this release)
├── tests/
│   └── (Phase 0 sanity checks as automated tests, Section 22)
└── eval/
    └── (10-subject benchmark set, Section 24 — built/frozen before
         final corpus assembly, per Section 11.4 contamination requirement)
```

`configs/model.yaml` existing as the single source of truth for
hyperparameters is a deliberate choice: it is what prevents a repeat of the
Section 0 bug, where the documented config and the (hypothetical)
implemented config could silently drift apart. Parameter-count verification
(Section 3) should be a small script run against this config file, not a
one-off manual calculation redone from memory each time.

---

## 28. Production Readiness Checklist — all items passed before the full runs

This checklist gated the transition from planning to any paid cloud
compute. Every box below was checked before `baseline` started
(2026-10-28) or before the corresponding later feature (corpus v2, queue
v3, torch.compile) started being used in a paid run.

**Data & Legal**
- [x] Every training-data source logged in the Section 11.2 ledger with a
      recorded license and `include` decision (`data/SOURCES.md`, v1; the
      `sources` block of `manifest_new.json`, v2 — Section 11.5)
- [x] Deduplication (exact + near-duplicate) run on the assembled corpus
- [x] Eval-set contamination check run and passed against the final corpus
      (both v1 and v2; v2 additionally checked against the v1 val split)
- [x] Held-out validation split carved out *before* packing (`shards/val`,
      `shards_v2/val_new`)

**Tokenizer**
- [x] Tokenizer trained on the actual cleaned corpus (not a generic corpus)
- [x] Bytes-per-token measured per subject domain, no major outlier subject
- [x] Tokenizer version hashed and frozen (reused unchanged for corpus v2,
      per Section 21.6)

**Model & Config**
- [x] Every `configs/*.yaml` matches its own Section 2/31.4 specification
- [x] Parameter count verified programmatically against each config's
      `expected_param_count` (`scripts/verify_param_count.py`) — not
      eyeballed, for all 11 configs
- [x] Initialization scheme (Section 3.3) implemented

**Training Loop**
- [x] NaN/Inf guard implemented and tested (Section 21.1)
- [x] OOM auto-recovery implemented and tested (Section 21.3)
- [x] Checkpoint save/upload/verify path implemented (Section 17)
- [x] SIGTERM emergency-checkpoint handler implemented and tested
      (Section 18) — against real spot reclaims and service restarts, not
      only the drill
- [x] Resume path tested via an actual kill-and-resume drill (Section 22.1
      item 4), then proven repeatedly in production
- [x] Early-stopping on held-out validation loss implemented and made
      configurable (`--early-stop-patience`, Section 21.5) — disabled for
      the one-epoch corpus-v2 runs, where it would not be meaningful

**Stop Controls**
- [x] Stop handler for tagged instances and the queue stop state wired and
      tested before the full runs (Section 14.4); bugs found in testing were
      fixed before relying on it
- [x] Spot price checked live at each launch, not assumed; the instance
      family itself changed from the originally-assumed `g5.xlarge` to
      `g6.xlarge` based on live capacity (Section 14.1)

**Phase 0 (Section 22) — all 5 steps passed, see Section 22.1 for results**

**Operational infrastructure beyond the original checklist (Section 14.4)**
- [x] Watchdog Lambda restarts the instance after an unplanned stop, so the
      operator's laptop does not need to stay on
- [x] Launcher Lambda retries a spot launch on a schedule when capacity is
      briefly unavailable
- [x] Queue v3 runs jobs from an S3-hosted list, re-read before each job,
      so new work can be appended without redeploying anything

---

## 29. Open Risks and Decisions Log

Explicit unresolved items — the point of this section is that nothing below
is silently assumed true elsewhere in this document.

1. **Corpus size — RESOLVED, then extended.** v1 finished at exactly
   1,151,073,280 tokens (Wikipedia + ArXiv CS + The Stack, matching the
   ~1.15B target) and trained three full runs. Being on the small side for
   the 164.8M scaling point, it was extended rather than replaced: corpus
   v2 (Section 11.5) adds 3.145B new CS-filtered tokens from four more
   sources on top of v1, for 4.296B tokens total. Early stopping (Section
   21.5) remains the overfitting guard, now configurable per job and
   disabled for the one-epoch v2 runs where it would not apply.
2. **Real GPU throughput for this exact model/config — RESOLVED, and the
   GPU itself changed.** The instance ended up being `g6.xlarge` (NVIDIA
   L4), not the originally-assumed `g5.xlarge` (A10G) — see Section 14.1
   for why. Measured: 17.2 s/step eager, 10.3 s/step with `torch.compile`
   (1.67x), at the standard 499,712-token effective batch (Section 19.1).
3. **Peak learning rate (6e-4) — RESOLVED for the dense family,** carried
   through three full runs without divergence. Still open specifically for
   the loop variant (Section 16.2) — a shared block's summed gradient may
   need a lower LR, under test in `model_loop_6x2_lr3e-4.yaml`.
4. **Safe micro-batch size — RESOLVED.** 8 (d_model 768) and 4 (d_model
   1024, 1024 or 2048 context) run without OOM on the L4's 24GB, with
   gradient accumulation making up the rest of the 499,712-token effective
   batch (Section 16.3).
5. **Long-context (>1,024 token) behavior is explicitly out of scope** for
   this version — RoPE extrapolation is not enabled and not validated
   (Section 5.2). Any future decision to support longer context needs its
   own validation pass, not an assumption that it "just works."
6. **Data sources — RESOLVED.** All sources verified and recorded in
   data/SOURCES.md with license decisions. 12 sources approved (include),
   4 excluded. License ledger gate satisfied.
7. **Spot pricing is not hard-coded anywhere in this document on purpose**
   — it fluctuates by region/AZ/time, and the queue's stop-and-resume logic
   (Section 14.4) is the real safeguard, not a pricing assumption made months before
   training actually runs. This mattered in practice: `g5` (A10G) capacity
   was unreliable across every us-east-1 AZ when launches were attempted,
   while `g6` (L4) capacity was available, which is why the instance family
   actually used (Section 14.1) differs from the original plan.
8. **Context length (1,024 tokens) vs. attention mechanism — DECIDED,
   revisit only with a reason.** Considered switching to Grouped-Query
   Attention or doubling context to 2,048. Not done: at this parameter
   scale (30–96M) and this context length, per-sequence KV-cache is
   ~18–36MB (12 layers × 2 × 1,024 × 12 heads × 64 d_head × 2 bytes,
   halved for the 6-unique-layer §31 variant's still-12-deep cache — cache
   size depends on effective depth, not unique params) — already inside
   the <850MB inference target (Section 23) with huge headroom. GQA solves
   a KV-cache memory problem that does not exist yet at this scale; adding
   it now would mean re-deriving every RoPE/KV-cache correctness test in
   Section 22/`tests/test_rope.py` for a real-world saving of a few MB.
   1,024 → 2,048 is a legitimate, zero-parameter-cost future option
   (doubles the attention term's compute, which §7's own note already
   observes is not the dominant cost at this d_model) but requires
   re-packing shards (Section 13) — deferred until corpus collection
   (Section 11) is finished, so it is a one-time decision made once instead
   of twice.
9. **Eval benchmark size — PARTIALLY RESOLVED.** The original
   `eval/benchmark.json` (20 questions, 2/subject) was indeed too small for
   a meaningful cross-config comparison; the 650-question MMLU-CS set
   (`eval/mmlu_cs_benchmark.json`) is now the primary metric used for the
   `baseline`/`loop_6x2` comparison (Section 23.1) and gives enough
   questions per subject to not be dominated by 1-2 flipped answers. It
   does not fully resolve Section 31.5's concern, though: MMLU-CS only
   covers 6 of AstraCSE's 10 training subjects (no dedicated category for
   data structures, operating systems, computer networks, database
   systems, or software engineering), and at the current near-chance
   accuracy level (Section 23.1) the benchmark cannot yet distinguish the
   models it was meant to compare — that distinguishing power is expected
   to emerge (or not) as the scaling runs land, not from the benchmark
   itself changing size again.

---

### 29.1 Risks added 2026-11-01

10. **The v1 contamination check is incomplete.** The v1 13-gram check covered
    only the 20-item smoke-test benchmark (0 of 20 items found). It was not
    run against the 650-item MMLU-CS set. v2 was checked against both. Re-run
    the v1 check against `eval/mmlu_cs_benchmark.json` before the paper, since
    an overlap would inflate the MCQ numbers. **Status: done 2 Nov (0 of 650 items flagged in the v1 and v2 training shards).**
11. **Stop-handler reporting lag.** The stop handler reacts to a reporting
    value that lags real activity by about a day, so it can act later than
    the reading suggests. **Status: accepted; the queue completed all 19 jobs.**
12. **Spot stops.** On 2026-10-31 there were five AWS-side stops (02:07 to
    10:20 UTC), and `baseline_v2` has had five SIGTERM stops (21:44 to 06:53
    IST). Each one resumed from a `sigterm-step-*.pt` checkpoint, with 8 to 25
    minutes of downtime. The cause of the `baseline_v2` stops is assumed to be
    the same as the others, but not verified. **Status: mitigated by checkpoint and resume;
    schedule effects are tracked in the run log (not in this release).**
13. **Seed variance is unmeasured.** Most configurations ran one seed. The v1
    seed reruns ran before `d1024_loop6x2_v2` (second v1 seed done; see the 10 Nov update). **Status: open until
    the reruns complete.**
14. **Queue stop before the last jobs.** The stop handler could halt the
    queue before its last jobs finished. **Status: resolved on 10 Nov 2026;
    all 19 queued jobs are done.**

**Updates on 2026-11-07:**
- Item 10 is closed: the v1 and v2 training shards were checked against the 650-item MMLU-CS set on 2 Nov (0 flagged).
- Item 13 is partly measured: a second v1 seed of `baseline`, `dense_6l` and `loop_6x2` is done (Section 31.9). The matched loop-minus-dense gap is 1.81 nats on seed 1 and 1.83 on seed 0. The v2 and width-1,024 runs are single-seed.
- Item 14: resolved on 10 Nov 2026. All 19 queued jobs are done.
- New item 15: `d1024_loop6x2_v2` failed once on 6 Nov, because `configs/model_d1024_loop6x2.yaml` was missing on the GPU box. The committed file was copied (checksum verified) and the job restarted from step 0. Procedure: `the operations runbook (not in this release)`, retry a failed job.

---

## 30. Implementation Status (as of 2026-11-07)

This section tracks actual implementation progress against the spec. The
2026-09-19 revision of this section described a pre-cloud-GPU plan; every
item it marked pending has since run on real AWS infrastructure, and the
corpus, infrastructure, and test suite have all grown substantially beyond
that revision. History is kept in `git log` and the run log (not in this release)
rather than duplicated here.

### Completed

**Phase 0 — Data Sourcing (both corpora):**
- v1: all sources in `data/SOURCES.md` verified and approved with license
  decisions; small corpus (257 files, 2.82M tokens) plus the large download
  (Wikipedia CS, ArXiv CS, The Stack) finished at 1,151,073,280 tokens
- v2 (Section 11.5): `scripts/expand_corpus.py` + `scripts/run_corpus_v2.sh`
  built from scratch — CS-lexicon filtering, dual decontamination (eval
  benchmarks + v1 val split), exact dedup, per-source caps, packing. Ran to
  completion 2026-10-30: 4.296B tokens total (3.145B new), `val_new` held
  out at 15.8M tokens

**Phase 1 — Data Processing:**
- `scripts/clean_text.py`, `extract_pdfs.py` (pymupdf), `dedup.py`
  (MinHash/LSH), `train_tokenizer.py` (byte-level BPE, vocab 15,000, frozen
  hash), `pack_shards.py` (uint16 memmap shards, seq_len 1024, `<doc>`
  boundaries, val split), `check_contamination.py` (13-gram overlap,
  passed 0/20 benchmark items) — all as in the 09-19 revision, unchanged
  and still in use; v2 (above) reuses the same frozen tokenizer rather than
  retraining one (Section 21.6)
- Eval benchmarks: `eval/benchmark.json` (20 questions × 10 subjects,
  original smoke test) plus `eval/mmlu_cs_benchmark.json` (650 MCQ
  questions, 6 subjects, built by `scripts/download_mmlu.py`) — the latter
  is now the primary metric (Section 23.1)

**Phase 2 — Model Implementation (gate passed):**
- `src/model.py` — full architecture per §2–9, plus the loop-transformer
  weight-sharing mode (§31, generalized to any `d_model`/`n_head`/`d_ff`,
  not just the baseline width — see the `d1024` configs)
- `src/data.py`, `src/train.py` — as in the 09-19 revision, plus since
  added: `torch.compile` support (measured 1.67x, Section 19.1), a
  recorded/seedable RNG (`--seed`), weight init from a local path or an S3
  path/prefix (`--init-from`, used for context extension), configurable
  checkpoint pruning (`--keep-local`) and early-stop patience
  (`--early-stop-patience`), and a per-run `train.log` archived to S3
- `src/infer.py` — unchanged from the 09-19 revision
- `scripts/verify_param_count.py` — now checked against all 11 configs
  (Section 31.4), not just the baseline
- **125 tests passing** (up from 71): the original suite plus
  `test_train_phases.py`, `test_checkpoint_pruning.py`,
  `test_expand_corpus.py` (16 tests, corpus v2 filtering/decontamination/
  packing), queue tests (7, not in this release; they run the real queue script under `bash`
  against fakes), watchdog and budget-handler tests (8, not in this release),
  `test_pack_shards.py`, `test_check_contamination.py`

**Phase 3 — Debug run and validation: passed.** See Section 22.1 for the
five results.

**Phase 4 — Full Training: 19 of 19 jobs done (10 Nov 2026).**
(the queue script, Section 14.4, not in this release) has run, in order, on a single
`g6.xlarge` spot instance:

| Run | Params | Result |
|---|---:|---|
| `baseline` | 96.47M | Done. val 2.853, ppl 17.34, MCQ 26.9% |
| `loop_6x2` | 54.01M | Done. val 4.622, ppl 101.75, MCQ 28.6% |
| `loop_4x3` | 39.85M | Done. val 4.706, ppl 110.57, MCQ 26.6% |
| `loop_3x4` | 32.77M | Done. val 4.947, ppl 140.73, MCQ 27.8% |
| `dense_6l`, `dense_4l`, `dense_3l`, `loop_6x2_lr3e-4` | 32.8M–54.0M | Done (see §31.8 and `docs/RESULTS.md`) |
| `baseline_v2` | 96.47M | Failed on corpus v2 at step 5,087 (1 Nov 2026, 15:07 IST): local disk likely full. See the run log (not in this release) incident 11 |
| Later jobs (4 to 10 Nov) | 32.8M–164.8M | Done: `loop6x2_v2`, `d1024_v2`, `d1024_ctx2048`, `dense_6l_v2`, `loop_6x2_wu171`, the three v1 second seeds, `d1024_loop6x2_v2` (8 Nov) and `baseline_v2_s1` (9 Nov). Nothing is running or paused |

**Phase 5 — Evaluation:** running per-job as part of the queue
(`run_eval.py` against both benchmark files, uploaded to
`results/<run>/`), not a separate pass at the end. `baseline` and
`loop_6x2` are evaluated; later jobs will add to the same results tree,
including a `val_new`-based eval once the corpus-v2 runs land.

**Infrastructure beyond the 09-19 revision (Section 14.4):**
- Stop handler (source not in this release) —
  stop handler for tagged instances (a budget-stop function),
  bugs found in testing and fixed (Section 14.4)
- Watchdog (source not in this release) — a watchdog function
- Launcher (source not in this release) —
  a launcher function
- `scripts/bench_compile.py` — eager-vs-compiled benchmark

### Pending

**Phase 4 (complete, 10 Nov 2026):** 19 of 19 jobs are done. `d1024_loop6x2_v2` (width 1,024, loop, 90.1M) finished on 8 Nov (val 1.999, clean 2.057). `baseline_v2_s1` finished on 9 Nov (val 2.009, clean 2.067). `d1024_loop6x2_v2` failed once on 6 Nov because its config file was missing on the GPU box; the committed file was copied (checksum verified) and the job restarted from step 0. The equal-parameter dense controls for K = 3, 4 and 6 are complete (Section 31.8), and so is the v2 matched-parameter control (Section 31.9).

**Phase 5 (continued):** eval on `val_new`; the full parameter-count vs.
validation-perplexity vs. MCQ-accuracy comparison across all configs once
the sweep finishes.

**Phase 6 — Export:** `src/infer.py export` ready; not yet run against a
final trained checkpoint end-to-end (inference throughput/VRAM targets in
Section 23.1 are accordingly still unmeasured).

**Cleanup (tracked in the run log (not in this release), not architectural):** a
diagnostic EBS volume and an old AMI/snapshot pair from the Oct 29 incident
(the run log (not in this release) incidents), and pruning the owner-IP `/32` security-group
rules accumulated across the project.

---

## 31. Loop-Transformer Variant — Parameter-Efficiency Study

Added to support an IEEE-general-paper-level contribution beyond "trained a
small LM": a controlled comparison of parameter efficiency at matched
effective depth and matched per-token compute.

### 31.1 Mechanism

Standard transformer: `n_layer` depth-slots, each with its own weights.
Loop-transformer: only `n_unique_layers` (K) distinct blocks exist; the
forward pass still runs `n_layer` (= K × T) layer-applications, but cycles
through the K unique blocks T times (`block = unique_blocks[depth_slot % K]`).

This is a depth-slot indexing change only — RoPE, KV-cache, causal masking,
and weight tying are completely unaffected, because:
- The KV-cache is a list indexed by **depth-slot** (0..n_layer-1), not by
  weight identity. Two depth-slots that reuse the same block still get their
  own independent cache entries (tests/test_loop_transformer.py — `
  test_kv_cache_per_slot_entries_differ_when_looped`).
- RoPE position offset logic (§5, the KV-cache offset bug fixed earlier)
  operates per depth-slot's attention call exactly as before; looping does
  not change how positions are computed.
- PyTorch's `.parameters()` de-duplicates by tensor identity (the same
  mechanism already relied on for embed/lm_head weight tying, §3.2), so a
  shared block is counted once, and autograd correctly **sums** its gradient
  contributions across all T depth-slots that used it in the forward pass —
  verified directly in `test_shared_gradient_equals_sum_of_unrolled_gradients`
  against an independently-constructed "unrolled" reference model.

### 31.2 Depth conditioning

A shared block has no intrinsic signal telling it which of its T loop
iterations is executing. Following the Universal Transformer (Dehghani et
al., 2018) pattern, a learned per-depth-slot vector `depth_embed[n_layer,
d_model]` is added to the residual stream immediately before each layer
application. This only exists when `n_unique_layers < n_layer`
(`AstraCSE.is_looped`); the non-looped baseline allocates nothing extra and
is byte-for-byte unchanged (`test_baseline_not_looped`).

### 31.3 What this buys — and what it does not

**Reduces:** unique parameter count and checkpoint/memory footprint (only K
blocks are ever stored, vs. n_layer).

**Does not reduce:** training or inference FLOPs/latency. The forward pass
still executes n_layer layer-applications regardless of how many are weight
-shared; wall-clock training cost per token is the same as the n_layer,
non-shared baseline. The claim under test is parameter efficiency at matched
compute, not speed.

### 31.4 Configs and the efficiency-frontier sweep

| Config | K (unique) | T (loops) | Params | vs. baseline |
|---|---|---|---|---|
| `configs/model.yaml` (baseline) | 12 | 1 | 96,473,856 | 100% |
| `configs/model_loop_6x2.yaml` | 6 | 2 | 54,006,528 | 56.0% |
| `configs/model_loop_4x3.yaml` | 4 | 3 | 39,847,680 | 41.3% |
| `configs/model_loop_3x4.yaml` | 3 | 4 | 32,768,256 | 34.0% |

All four share identical `n_layer=12` effective depth, identical corpus,
identical training steps — so training wall-clock cost is ~equal across all
four (§31.3). Total GPU time for the sweep is therefore about four times a single
run's (§19).

The paper's headline artifact is the resulting curve: parameter count vs.
validation perplexity vs. eval-benchmark accuracy across these four points,
cross-checked against equal-parameter dense controls and repeated at a
second width — the full sweep actually queued goes well beyond the
original four points:

| Config | Role | Params | Status |
|---|---|---:|---|
| `baseline` | K=12, T=1 (no sharing) | 96,473,856 | Done — val 2.853, MCQ 26.9% |
| `loop_6x2` | K=6, T=2 | 54,006,528 | Done — val 4.622, MCQ 28.6% |
| `loop_4x3` | K=4, T=3 | 39,847,680 | Done — val 4.706, MCQ 26.6% |
| `loop_3x4` | K=3, T=4 | 32,768,256 | Done — val 4.947, MCQ 27.8% |
| `dense_6l` | equal-param control for `loop_6x2` (6 real layers, T=1) | 53,997,312 | Done — val 2.795, MCQ 25.5% |
| `dense_4l` | equal-param control for `loop_4x3` | 39,838,464 | Done — val 2.998, MCQ 26.3% |
| `dense_3l` | equal-param control for `loop_3x4` | 32,759,040 | Done — val 3.198, MCQ 27.5% |
| `model_loop_6x2_lr3e-4` | LR check: does lowering the peak LR close part of the `loop_6x2` gap? | 54,006,528 | Done — val 4.592, MCQ 24.2% (0.03 nats better; §31.7) |
| `loop6x2_v2` | `loop_6x2` on corpus v2 (4.3x the tokens) | 54,006,528 | Done — val 2.084 (clean 2.140), MCQ 27.5% |
| `dense_6l_v2` | matched-parameter control for `loop6x2_v2` on corpus v2 | 53,997,312 | Done — val 2.135 (clean 2.192), MCQ 28.2% |
| `d1024_v2` | the dense 12-layer model at d_model 1024, corpus v2 | 164,807,680 | Done — val 1.927 (clean 1.986), MCQ 27.7% |
| `d1024_ctx2048` | `d1024_v2` continued for 400 steps at 2,048 context (evaluated at 1,024) | 164,807,680 | Done — val 1.924 (clean 1.983), MCQ 28.5% |
| `loop_6x2_wu171` | `loop_6x2` with 171 warmup steps instead of 46 (v1) | 54,006,528 | Done — val 4.006 (clean 4.074), MCQ 25.4% |
| `d1024_loop6x2_v2` | the same K=6,T=2 sharing at d_model 1024, on corpus v2 | 90,096,640 | Done — val 1.999 (clean 2.057), MCQ 26.5% CF-norm. Failed once on 6 Nov (config missing on the box), fixed |

The `dense_*` controls are not in the original four-point sweep; they were
added once the first result (§31.6) came in, specifically to separate two
things `loop_6x2` changes at once relative to `baseline` — depth-wise weight
sharing, and simply having fewer unique parameters. `dense_6l` has the same
unique-parameter count as `loop_6x2` but no sharing (6 real layers, run
once, not looped); if it scores close to `loop_6x2`, the measured gap is
mostly a parameter-count effect, not a sharing-mechanism effect. Result
(2026-10-31): `dense_6l` scored 2.795, so the gap is not a parameter-count
effect (§31.7).

### 31.5 Eval benchmark size — partially resolved

The original `eval/benchmark.json` (20 questions, 2/subject) was indeed too
small to produce a statistically meaningful accuracy comparison across
models — noise from 1-2 flipped answers would dominate any real signal.
The 650-question MMLU-CS set (`eval/mmlu_cs_benchmark.json`) is now used as
the primary comparison metric (§23.1) and has enough questions per subject
to not be dominated by a couple of flipped answers. It does not fully
resolve the concern: MMLU-CS covers only 6 of AstraCSE's 10 training
subjects, and at the current near-chance accuracy level (§31.6) it cannot
yet distinguish the configs it is meant to compare. That is expected to
change as the scaling runs (corpus v2, `d1024`) land, not something to fix
by growing the benchmark again.

### 31.6 First measured result (2026-10-30)

`loop_6x2` (54.0M, 56% of baseline's parameters) is far worse than
`baseline` (96.5M) at the identical 2304-step, identical-corpus budget:
validation loss 4.622 vs 2.853, perplexity 101.75 vs 17.34. MCQ accuracy is
close to chance for both (28.6% vs 26.9%, chance = 25% on 4-choice
questions) and does not yet distinguish them (§31.5). The training loss
curve shows the same gap from early in training, growing as the LR decays
(in-training val loss at step 500: baseline 6.27, loop_6x2 7.08; by step
2000: baseline 4.04, loop_6x2 5.36) — not a late divergence, and no
spikes/NaNs/OOMs in either run.

This is a clear negative result for weight sharing at this scale and this
step budget, but at the time it was measured it was not yet a clean
mechanism result: `loop_6x2` differs from `baseline` in two confounded ways
(depth-wise weight sharing, and having fewer unique parameters absorbing
the same per-token compute), and the LR used (6e-4) was tuned against the
dense baseline, not against the loop variant's larger effective per-step
update (§16.2). `dense_6l` and `loop_6x2_lr3e-4` were queued specifically
to separate these explanations.

### 31.7 Mechanism isolated (2026-10-31): it is the sharing, not params or LR

Both follow-ups have since finished, and together they resolve the
confound:

| Config | Params | Mechanism | Val loss |
|---|---:|---|---:|
| `baseline` | 96.47M | 12 real layers | 2.853 |
| `dense_6l` | 54.00M | 6 real layers, no sharing | **2.795** |
| `loop_6x2` | 54.01M | 6 unique blocks, looped x2, LR 6e-4 | 4.622 |
| `loop_6x2_lr3e-4` | 54.01M | 6 unique blocks, looped x2, LR 3e-4 | 4.592 |

`dense_6l` has the identical unique-parameter count to both loop configs
and no sharing at all — it scores *better* than the 96.5M baseline despite
44% fewer parameters, which rules out "too few unique parameters" as the
cause of the loop variants' much worse loss (a 54M-parameter model clearly
has enough capacity to reach baseline-level loss at this corpus size and
step budget; it just needs those parameters applied without the depth-wise
reuse). `loop_6x2_lr3e-4` halves the peak LR and moves the loss by only
0.03 nats (4.592 vs 4.622) — far short of a result that would justify
calling 6e-4 "too high" as the explanation, so the §16.2 hypothesis about
the shared block's summed gradient is mostly ruled out too.

**What is left standing as the explanation is the sharing mechanism
itself** — reusing the same weights at multiple depth slots (with only a
learned per-slot additive embedding distinguishing them, §31.2) — not a
confound this experiment failed to control for. This is now a clean,
paper-worthy negative result for Universal-Transformer-style depth-wise
weight sharing at this parameter scale and training budget, rather than
one data point toward an eventual conclusion.

**Decision this resolves:** `loop6x2_v2` and `d1024_loop6x2_v2`
(Architecture.md §31.4) keep their already-queued peak LRs (6e-4, 5e-4) —
the LR check did not clear the bar set in advance (§16.2: "if 3e-4 is
clearly better") for changing them, so no `jobs.json` edit was made.
`loop_3x4`/`dense_4l`/`dense_3l` (running/queued) extend the same
isolation to K=3 and K=4; if they show the same pattern, the finding holds
across the whole K∈{3,4,6} sweep, not just one point on it.

### 31.8 Matched-parameter results, complete for K = 3, 4, 6 (2026-11-01)

| Pair | Loop (sharing), val loss | Dense control (same unique params, no sharing), val loss | Gap (nats) |
|---|---|---|---:|
| K = 6 | `loop_6x2`: 4.622 | `dense_6l`: 2.795 | 1.83 |
| K = 4 | `loop_4x3`: 4.706 | `dense_4l`: 2.998 | 1.71 |
| K = 3 | `loop_3x4`: 4.947 | `dense_3l`: 3.198 | 1.75 |

At every K, sharing costs 1.7 to 1.8 nats relative to the dense control with
the same unique parameters. Among the loop models, loss rises as sharing
increases (K = 6 < K = 4 < K = 3). The dense controls apply each layer once,
so they run T times fewer layer applications per token than their loop
counterparts. They are matched on parameters, not compute. The compute-matched
comparison (12 applications per token) is `baseline` (96.5M, val 2.853)
against `loop_6x2` (54.0M, val 4.622).

MCQ accuracies in this table are 24% to 29% on 650 four-choice questions.
Their 95% interval is about ±3.4 points, so they do not separate the models
(`docs/RESULTS.md` §5). Full numbers, per-subject counts and the threats to
validity are in `docs/RESULTS.md` and `docs/FINDINGS.md`. The v1 seed reruns
are done (Section 31.9): the matched gap is 1.81 nats on the second seed, against 1.83 on the first.

### 31.9 Results after 2026-11-01 (updated 2026-11-07)

Numbers are validation loss in nats on the same v1 validation set (full set, with clean-token values in brackets; `docs/RESULTS.md`). MCQ is the conditional-likelihood CF-norm rule, unless stated.

**Matched parameters on corpus v2 (the sign reverses).** `loop6x2_v2` (54.0M, 2.084 [2.140]) against `dense_6l_v2` (54.0M, 2.135 [2.192]): the loop is 0.052 nats better. On v1 the same pair was 1.83 nats apart the other way. These v2 results are single-seed, and a seed spread of about 0.1 nats is visible in v1 single models, so the reversal is provisional. The compute-matched gap on v2 (loop against the 96.5M dense baseline, 2.015 [2.072]) is +0.07 nats.

**Warmup ablation on v1 (`loop_6x2_wu171`).** The loop model with 171 warmup steps instead of 46 reaches 4.006 [4.074], against 4.622 [4.692] for the original. Against `dense_6l` (2.856 clean), the clean gap falls from 1.84 to 1.22 nats, which recovers about a third (0.62 nats) of the first-regime penalty. The rest of the penalty comes from something else in the first regime: schedule length, data mix or document order (Section 31.8 lists the candidates).

**Second seed on v1.** `baseline_s1` 2.887, `dense_6l_s1` 2.697, `loop_6x2_s1` 4.502 (full set). The matched gap is 1.805 on seed 1 and 1.827 on seed 0. The compute-matched gap is 1.616 on seed 1 and 1.769 on seed 0. The matched gap is stable across seeds; the compute-matched gap is not, and the paper reports the spread.

**Width 1,024.** `d1024_v2` (164.8M) reaches 1.927 [1.986], 0.087 nats below `baseline_v2`. Continuing it for 400 steps at a 2,048-token context changes its loss at 1,024 tokens by -0.003 nats. This evaluation does not test longer context. `d1024_loop6x2_v2` (90.1M) finished on 8 Nov (val 1.999, clean 2.057).

**MCQ.** Under the three conditional-likelihood rules every model is at chance, including the width-1,024 models (CF-norm 26.0% and 26.2%). CF-norm across all runs ranges from 22.9% to 27.4%. Full table: `docs/RESULTS.md` section 2.1.

