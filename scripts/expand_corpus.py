"""Corpus v2 — add CS text from openly licensed Hugging Face datasets to the v1 corpus.

Two stages, run on a CPU instance (not the GPU):

  extract  For one source, download its files, keep CS-relevant documents, tokenize
           them with the frozen v1 tokenizer, and drop any document that
             - shares a 13-gram with a benchmark item (eval contamination),
             - shares >= VAL_OVERLAP_MAX of its 13-grams with the v1 validation split
               (so v1 val loss stays a held-out measure),
             - is shorter than MIN_DOC_TOKENS tokens.
           Writes one .npz per input file to <workdir>/<source>/ (tokens, doc lengths,
           text hashes) plus a stats .json, and stops after --max-tokens.
  pack     Exact-dedup across all sources, apply per-source token caps, shuffle
           documents globally (so every shard mixes sources), hold out 0.5% of the new
           documents as val_new, and pack uint16 shards like scripts/pack_shards.py
           (<doc> after each document, SEQ_LEN-aligned shards of TOKENS_PER_SHARD).

The v1 shards are not re-packed: shards_v2/train = v1 train shards + the new shards,
and every run is still evaluated on the v1 val split.

Usage:
  python scripts/expand_corpus.py extract --source fineweb --max-tokens 1.1e9 \
      --workdir /data/work --val-shards data/shards/val --workers 12
  python scripts/expand_corpus.py pack --workdir /data/work --out-dir /data/shards_v2_new
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.check_contamination import NGRAM_SIZE, benchmark_item_hashes, window_hashes  # noqa: E402
from scripts.pack_shards import SEQ_LEN, TOKENS_PER_SHARD, write_shard  # noqa: E402

MIN_DOC_TOKENS = 64
MIN_WORDS = 150
VAL_OVERLAP_MAX = 0.20
SCORE_CHARS = 20_000          # lexicon scoring reads only the start of long documents
VAL_NEW_MOD = 200             # 1 in 200 new documents goes to val_new

# Distinctive CS terms across the 10 subjects. Words with common non-CS meanings
# ("class", "function", "memory", "network", "tree", "process", ...) are left out on
# purpose, or appear only inside unambiguous phrases.
CS_TERMS = [
    # algorithms and data structures
    "algorithm", "algorithms", "time complexity", "space complexity", "big-o", "big o notation",
    "asymptotic", "dynamic programming", "greedy algorithm", "divide and conquer",
    "binary search", "quicksort", "merge sort", "mergesort", "heapsort", "bubble sort",
    "insertion sort", "hash table", "hash map", "hashmap", "linked list", "binary tree",
    "binary search tree", "red-black tree", "avl tree", "b-tree", "b+ tree", "trie",
    "priority queue", "adjacency list", "adjacency matrix", "depth-first search",
    "breadth-first search", "dijkstra", "bellman-ford", "minimum spanning tree",
    "topological sort", "memoization", "np-complete", "np-hard", "data structure",
    "data structures", "recursion", "recursive function",
    # theory of computation
    "turing machine", "finite automaton", "finite automata", "regular expression",
    "context-free grammar", "pushdown automaton", "undecidable", "halting problem",
    "computability", "complexity class", "polynomial time", "lambda calculus",
    # programming
    "source code", "compiler", "interpreter", "programming language", "python",
    "javascript", "typescript", "java", "c++", "c#", "golang", "rust programming",
    "object-oriented", "encapsulation", "exception handling", "syntax error",
    "runtime error", "debugger", "debugging", "unit test", "unit tests", "api", "sdk",
    "git", "github", "null pointer", "garbage collector", "integer overflow", "data type",
    "boolean", "for loop", "while loop", "if statement", "programming", "programmer",
    "source file", "stack trace", "compile time", "runtime",
    # operating systems
    "operating system", "operating systems", "kernel", "linux", "unix", "context switch",
    "virtual memory", "page table", "deadlock", "semaphore", "mutex", "race condition",
    "system call", "file system", "interrupt handler", "multithreading", "concurrency",
    "thread pool", "process scheduling", "scheduler",
    # computer architecture
    "cpu", "instruction set", "assembly language", "cache miss", "cache hit", "pipelining",
    "branch prediction", "alu", "microarchitecture", "risc", "cisc", "x86", "gpu", "dram",
    "sram", "memory hierarchy", "von neumann", "register file", "microprocessor",
    # networks
    "tcp", "udp", "ip address", "http", "https", "dns", "router", "routing protocol",
    "ethernet", "packet switching", "osi model", "tcp/ip", "subnet", "wi-fi", "socket",
    "network protocol", "firewall", "bandwidth", "latency", "lan", "wan",
    # databases
    "database", "databases", "sql", "relational database", "primary key", "foreign key",
    "database normalization", "query optimization", "acid properties", "nosql", "mysql",
    "postgresql", "sqlite", "mongodb", "database schema", "stored procedure",
    # security
    "encryption", "decryption", "cryptography", "public key", "private key", "rsa", "aes",
    "sha-256", "hash function", "malware", "phishing", "buffer overflow", "sql injection",
    "cross-site scripting", "xss", "authentication", "intrusion detection",
    "cybersecurity", "ransomware", "penetration testing", "vulnerability",
    # software engineering
    "software engineering", "software development", "agile", "scrum", "version control",
    "requirements engineering", "uml", "design pattern", "design patterns", "refactoring",
    "continuous integration", "devops", "code review", "software testing",
    "test-driven development", "microservices", "software architecture",
    # general
    "computer science", "computing", "software", "hardware", "computer", "computers",
    "machine learning", "neural network", "artificial intelligence",
]

LICENSES = {
    "fineweb": "ODC-By 1.0 (HuggingFaceTB/smollm-corpus fineweb-edu-dedup; CommonCrawl terms)",
    "cosmopedia": "ODC-By 1.0 (HuggingFaceTB/smollm-corpus cosmopedia-v2; synthetic)",
    "stackexchange": "CC BY-SA (common-pile/stackexchange_filtered; per-document license in metadata)",
    "arxiv": "open licenses per paper (common-pile/arxiv_papers_filtered)",
    "libretexts": "open licenses per page (common-pile/libretexts_filtered)",
}

# Stack Exchange sites whose questions are about computing. Meta sites are excluded.
SE_SITES = {
    "stackoverflow.com", "superuser.com", "serverfault.com", "askubuntu.com",
    "unix.stackexchange.com", "cs.stackexchange.com", "cstheory.stackexchange.com",
    "softwareengineering.stackexchange.com", "programmers.stackexchange.com",
    "codereview.stackexchange.com", "dba.stackexchange.com", "security.stackexchange.com",
    "networkengineering.stackexchange.com", "crypto.stackexchange.com",
    "reverseengineering.stackexchange.com", "retrocomputing.stackexchange.com",
    "devops.stackexchange.com", "sqa.stackexchange.com", "datascience.stackexchange.com",
    "ai.stackexchange.com", "computergraphics.stackexchange.com",
    "gamedev.stackexchange.com", "webmasters.stackexchange.com",
    "quantumcomputing.stackexchange.com", "vi.stackexchange.com", "emacs.stackexchange.com",
}
STACKOVERFLOW_KEEP_MOD = 6    # keep 1 in 6 Stack Overflow questions (it would dominate)

SOURCES = {
    # name: (repo, subdirectory or file prefix, format, lexicon threshold)
    "fineweb": ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup/", "parquet", 6),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", "parquet", 6),
    "stackexchange": ("common-pile/stackexchange_filtered", "stackexchange-", "jsonl.gz", 0),
    "arxiv": ("common-pile/arxiv_papers_filtered", "arxiv-papers-", "jsonl.gz", 12),
    "libretexts": ("common-pile/libretexts_filtered", "libretexts-", "jsonl.gz", 6),
}


# ---------------------------------------------------------------------------
# Filters (pure functions, unit-tested)
# ---------------------------------------------------------------------------

# Generic tech words that also appear in non-CS writing (photography, business,
# zoology for "python", travel for "java"). Together they add at most WEAK_CAP to a
# document's score, so they can support but never carry a CS decision.
WEAK_TERMS = {
    "computer", "computers", "computing", "software", "hardware", "programming",
    "programmer", "api", "sdk", "runtime", "latency", "bandwidth", "authentication",
    "vulnerability", "agile", "machine learning", "artificial intelligence",
    "neural network", "python", "java", "git", "router", "socket", "boolean", "lan",
    "wan", "wi-fi", "scheduler", "database", "databases", "encryption", "firewall",
}
WEAK_CAP = 2


# Cheap substring prefilter: a document with none of these cannot reach a CS score
# threshold in practice (on 1,000 sampled documents it kept all 24 that scored >= 4
# while passing 27% of documents), so the regex runs on far fewer of them.
CS_ANCHORS = ("comput", "softwar", "program", "algorithm", "linux", "unix", "kernel", "cpu",
              "protocol", "databas", "encrypt", "hardware", "cyber", "malware", "sql",
              "server", "internet")


def _trie_pattern(words: list[str]) -> str:
    """Alternation factored into a character trie. Python's re tries every branch of a
    flat alternation at each position; the trie form matched 11x faster here."""
    trie: dict = {}
    for w in words:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node[""] = True

    def build(node: dict) -> str:
        alts = [re.escape(ch) + build(node[ch]) for ch in sorted(k for k in node if k)]
        if not alts:
            return ""
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        return "(?:" + body + ")?" if "" in node else body

    return build(trie)


def build_cs_regex(terms: list[str]) -> re.Pattern:
    """Matches CS terms in lower-cased text. Boundaries are "not alphanumeric" rather
    than \\b, so terms such as "c++" and "c#" that end in punctuation still match."""
    words = sorted({t.lower() for t in terms})
    return re.compile(r"(?<![a-z0-9])(?:" + _trie_pattern(words) + r")(?![a-z0-9])")


def has_cs_anchor(low: str) -> bool:
    return any(a in low for a in CS_ANCHORS)


def cs_score(text: str, rx: re.Pattern, max_chars: int = SCORE_CHARS) -> tuple[int, int]:
    """(score, total CS-term matches) in the first max_chars of text. The score counts
    distinct specific terms plus at most WEAK_CAP distinct generic ones."""
    found = rx.findall(text[:max_chars].lower())
    distinct = set(found)
    weak = len(distinct & WEAK_TERMS)
    return len(distinct) - weak + min(weak, WEAK_CAP), len(found)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def text_hash(text: str) -> int:
    """64-bit hash of the whitespace-collapsed, lower-cased text, for exact dedup."""
    canon = " ".join(text.lower().split())
    return int.from_bytes(hashlib.md5(canon.encode("utf-8")).digest()[:8], "little")


def has_benchmark_ngram(tokens: np.ndarray, bench_sorted: np.ndarray, n: int = NGRAM_SIZE) -> bool:
    if len(bench_sorted) == 0:
        return False
    h = window_hashes(tokens, n)
    if len(h) == 0:
        return False
    idx = np.searchsorted(bench_sorted, h)
    idx[idx == len(bench_sorted)] = 0
    return bool(np.any(bench_sorted[idx] == h))


def val_overlap_fraction(tokens: np.ndarray, val_sorted: np.ndarray, n: int = NGRAM_SIZE) -> float:
    if len(val_sorted) == 0:
        return 0.0
    h = window_hashes(tokens, n)
    if len(h) == 0:
        return 0.0
    idx = np.searchsorted(val_sorted, h)
    idx[idx == len(val_sorted)] = 0
    return float(np.mean(val_sorted[idx] == h))


def keep_web_url(url: str) -> bool:
    """Wikipedia is already in the v1 corpus; leave its copies out of the web data."""
    host = (urlparse(url).hostname or "").lower()
    return not (host == "wikipedia.org" or host.endswith(".wikipedia.org"))


def keep_se_site(site: str, doc_id: str) -> bool:
    site = site.lower()
    if ".meta." in site or site.startswith("meta.") or site not in SE_SITES:
        return False
    if site == "stackoverflow.com":
        return int(hashlib.md5(str(doc_id).encode()).hexdigest(), 16) % STACKOVERFLOW_KEEP_MOD == 0
    return True


def accept(source: str, text: str, meta: dict, rx: re.Pattern) -> tuple[bool, str]:
    """Text-level decision for one document: (keep?, reason when rejected)."""
    if len(text.split()) < MIN_WORDS and source != "stackexchange":
        return False, "short"
    threshold = SOURCES[source][3]
    if source == "fineweb":
        if not keep_web_url(meta.get("url", "")):
            return False, "wikipedia"
        if int(meta.get("int_score") or 0) < 3:
            return False, "edu_score"
    elif source == "stackexchange":
        if len(text) < 400:
            return False, "short"
        if not keep_se_site(meta.get("site", ""), meta.get("id", "")):
            return False, "site"
        return True, ""
    elif source == "libretexts":
        if "computer_science" in meta.get("url", "").lower():
            return True, ""
    elif source == "cosmopedia":
        # Stories, dialogues and texts for young children or the "alien" persona teach
        # little CS; a BISAC "COMPUTERS / ..." topic in the generation prompt is a
        # precise label for computing content.
        fmt = (meta.get("format") or "").lower()
        if fmt.startswith("story") or fmt == "dialogue":
            return False, "format"
        if (meta.get("audience") or "").lower() in {"children", "young_children", "alien"}:
            return False, "audience"
        if "COMPUTERS / " in (meta.get("prompt") or ""):
            return True, ""
    if not has_cs_anchor(text[:SCORE_CHARS].lower()):
        return False, "not_cs"
    score, _ = cs_score(text, rx)
    if score < threshold:
        return False, "not_cs"
    return True, ""


# ---------------------------------------------------------------------------
# Reading source files
# ---------------------------------------------------------------------------

def iter_docs(source: str, path: Path):
    """Yield (text, meta) for each document of one downloaded file."""
    fmt = SOURCES[source][2]
    if fmt == "parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(path))
        cols = (["text", "metadata"] if source == "fineweb"
                else ["text", "prompt", "format", "audience"])
        for batch in pf.iter_batches(batch_size=2000, columns=cols):
            d = batch.to_pydict()
            n = len(d["text"])
            if source == "fineweb":
                metas = [m or {} for m in d["metadata"]]
            else:
                metas = [{"prompt": p, "format": f, "audience": a}
                         for p, f, a in zip(d["prompt"], d["format"], d["audience"])]
            for text, meta in zip(d["text"], metas[:n]):
                yield text or "", meta
    else:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                rec = json.loads(line)
                meta = dict(rec.get("metadata") or {})
                meta.setdefault("id", rec.get("id", ""))
                meta.setdefault("url", meta.get("url") or str(rec.get("id", "")))
                yield rec.get("text") or "", meta


def list_source_files(source: str) -> list[str]:
    from huggingface_hub import HfApi
    repo, prefix, fmt, _ = SOURCES[source]
    ext = ".parquet" if fmt == "parquet" else ".json.gz"
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if f.startswith(prefix) and f.endswith(ext)]
    return sorted(files)


# ---------------------------------------------------------------------------
# Extract stage (one worker process per file)
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(tok_dir: str, bench_path: str, val_path: str) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"   # one process per core already
    from tokenizers import ByteLevelBPETokenizer
    _W["tok"] = ByteLevelBPETokenizer(f"{tok_dir}/vocab.json", f"{tok_dir}/merges.txt")
    _W["rx"] = build_cs_regex(CS_TERMS)
    _W["bench"] = np.load(bench_path)
    _W["val"] = np.load(val_path, mmap_mode="r")      # shared page cache across workers


def process_file(source: str, repo_file: str, out_dir: str, dl_dir: str) -> dict:
    from huggingface_hub import hf_hub_download
    out = Path(out_dir) / (Path(repo_file).name.split(".")[0] + ".npz")
    if out.exists():
        return json.loads(out.with_suffix(".json").read_text())
    stats = {"file": repo_file, "docs": 0, "kept": 0, "tokens": 0, "rejected": {}}
    t0 = time.time()
    for attempt in range(1, 6):   # the Hub can rate-limit or drop a large download
        try:
            path = Path(hf_hub_download(SOURCES[source][0], repo_file, repo_type="dataset",
                                        local_dir=dl_dir))
            break
        except Exception as e:
            if attempt == 5:
                raise
            print(f"  download retry {attempt}/5 for {repo_file}: {type(e).__name__}", flush=True)
            time.sleep(30 * attempt)
    tok, rx, bench, val = _W["tok"], _W["rx"], _W["bench"], _W["val"]
    token_parts, lengths, hashes = [], [], []

    def reject(reason: str) -> None:
        stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1

    def flush(texts: list[str]) -> None:
        for text, enc in zip(texts, tok.encode_batch(texts)):
            ids = np.asarray(enc.ids, dtype=np.uint16)
            if len(ids) < MIN_DOC_TOKENS:
                reject("few_tokens"); continue
            if has_benchmark_ngram(ids, bench):
                reject("benchmark_ngram"); continue
            if val_overlap_fraction(ids, val) >= VAL_OVERLAP_MAX:
                reject("val_overlap"); continue
            token_parts.append(ids)
            lengths.append(len(ids))
            hashes.append(text_hash(text))
            stats["kept"] += 1
            stats["tokens"] += len(ids)

    pending: list[str] = []
    for text, meta in iter_docs(source, path):
        stats["docs"] += 1
        text = normalize_text(text)
        ok, why = accept(source, text, meta, rx)
        if not ok:
            reject(why); continue
        pending.append(text)
        if len(pending) >= 256:
            flush(pending); pending = []
    if pending:
        flush(pending)
    path.unlink(missing_ok=True)

    tokens = np.concatenate(token_parts) if token_parts else np.empty(0, dtype=np.uint16)
    np.savez(out, tokens=tokens, lengths=np.asarray(lengths, dtype=np.uint32),
             hashes=np.asarray(hashes, dtype=np.uint64))
    stats["seconds"] = round(time.time() - t0, 1)
    out.with_suffix(".json").write_text(json.dumps(stats))
    return stats


def build_filter_arrays(tok_dir: Path, bench_paths: list[Path], val_dir: Path,
                        workdir: Path) -> tuple[Path, Path]:
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    items = benchmark_item_hashes(bench_paths, tok, NGRAM_SIZE)
    bench = np.unique(np.concatenate([it["hashes"] for it in items])) if items else np.empty(0, np.uint64)
    parts = [np.unique(window_hashes(np.fromfile(p, dtype=np.uint16), NGRAM_SIZE))
             for p in sorted(val_dir.glob("*.bin"))]
    val = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)
    workdir.mkdir(parents=True, exist_ok=True)
    bp, vp = workdir / "bench_hashes.npy", workdir / "val_hashes.npy"
    np.save(bp, bench)
    np.save(vp, val)
    print(f"filters: {len(items)} benchmark items, {len(bench):,} benchmark 13-grams, "
          f"{len(val):,} val 13-grams")
    return bp, vp


def extract(args) -> int:
    workdir = Path(args.workdir)
    bp, vp = workdir / "bench_hashes.npy", workdir / "val_hashes.npy"
    if not (bp.exists() and vp.exists()):
        bp, vp = build_filter_arrays(Path(args.tok_dir), [Path(p) for p in args.bench],
                                     Path(args.val_shards), workdir)
    files = list_source_files(args.source)
    if args.max_files:
        files = files[:args.max_files]
    out_dir = workdir / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = workdir / "download" / args.source
    print(f"[{args.source}] {len(files)} files, cap {args.max_tokens:,.0f} tokens, {args.workers} workers")

    total = sum(json.loads(p.read_text())["tokens"] for p in out_dir.glob("*.json"))
    done = {json.loads(p.read_text())["file"] for p in out_dir.glob("*.json")}
    todo = [f for f in files if f not in done]
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(args.tok_dir, str(bp), str(vp))) as ex:
        queue = iter(todo)
        running = {}
        for _ in range(args.workers):
            f = next(queue, None)
            if f is None or total >= args.max_tokens:
                break
            running[ex.submit(process_file, args.source, f, str(out_dir), str(dl_dir))] = f
        failed = []
        while running:
            fut = next(as_completed(running))
            name = running.pop(fut)
            try:
                st = fut.result()
            except Exception as e:
                # One bad file must not lose the rest: it has no output, so a rerun retries it.
                failed.append(name)
                print(f"  FAILED {name}: {type(e).__name__}: {e}", flush=True)
            else:
                total += st["tokens"]
                print(f"  {st['file']}: kept {st['kept']:,}/{st['docs']:,} docs, "
                      f"{st['tokens']:,} tokens ({st.get('seconds', 0)}s); total {total:,}",
                      flush=True)
            f = next(queue, None)
            if f is not None and total < args.max_tokens:
                running[ex.submit(process_file, args.source, f, str(out_dir), str(dl_dir))] = f
    print(f"[{args.source}] done: {total:,} tokens"
          + (f"; {len(failed)} files failed: {failed}" if failed else ""))
    return 0


# ---------------------------------------------------------------------------
# Pack stage
# ---------------------------------------------------------------------------

def load_source(out_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    tokens, lengths, hashes = [], [], []
    stats = {"files": 0, "docs": 0, "kept": 0, "tokens": 0, "rejected": {}}
    for npz in sorted(out_dir.glob("*.npz")):
        d = np.load(npz)
        tokens.append(d["tokens"]); lengths.append(d["lengths"]); hashes.append(d["hashes"])
        st = json.loads(npz.with_suffix(".json").read_text())
        stats["files"] += 1
        for k in ("docs", "kept", "tokens"):
            stats[k] += st[k]
        for k, v in st["rejected"].items():
            stats["rejected"][k] = stats["rejected"].get(k, 0) + v
    if not tokens:
        return (np.empty(0, np.uint16), np.empty(0, np.uint32), np.empty(0, np.uint64), stats)
    return np.concatenate(tokens), np.concatenate(lengths), np.concatenate(hashes), stats


def pack_documents(docs: list[np.ndarray], doc_token_id: int, out_dir: Path,
                   start_index: int) -> tuple[list[str], int]:
    """Concatenate docs with <doc> after each and write SEQ_LEN-aligned shards of
    TOKENS_PER_SHARD tokens; the last partial sequence is dropped (as pack_shards.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sep = np.array([doc_token_id], dtype=np.uint16)
    paths, total, buf, buf_len, idx = [], 0, [], 0, start_index

    def emit(chunk: np.ndarray) -> None:
        nonlocal idx, total
        p = out_dir / f"shard_{idx:04d}.bin"
        write_shard(chunk, p)
        paths.append(str(p)); total += len(chunk); idx += 1

    for d in docs:
        buf.append(d); buf.append(sep); buf_len += len(d) + 1
        if buf_len >= TOKENS_PER_SHARD:
            flat = np.concatenate(buf)
            while len(flat) >= TOKENS_PER_SHARD:
                emit(flat[:TOKENS_PER_SHARD]); flat = flat[TOKENS_PER_SHARD:]
            buf, buf_len = [flat], len(flat)
    if buf_len >= SEQ_LEN:
        flat = np.concatenate(buf)
        emit(flat[:(len(flat) // SEQ_LEN) * SEQ_LEN])
    return paths, total


def pack(args) -> int:
    from tokenizers import ByteLevelBPETokenizer
    workdir, out_dir = Path(args.workdir), Path(args.out_dir)
    tok = ByteLevelBPETokenizer(f"{args.tok_dir}/vocab.json", f"{args.tok_dir}/merges.txt")
    doc_id = tok.token_to_id("<doc>")
    caps = dict(kv.split("=") for kv in args.caps) if args.caps else {}
    rng = random.Random(args.seed)
    seen: set[int] = set()
    train_docs, val_docs, manifest_sources = [], [], {}
    for k, source in enumerate(args.sources):
        tokens, lengths, hashes, stats = load_source(workdir / source)
        cap = float(caps.get(source, "inf"))
        offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
        used, dups, n_train, n_val = 0, 0, 0, 0
        # Random order before the cap: files can be sorted by topic (Stack Exchange
        # files are sorted by site name), so a cap in file order would keep only the
        # first sites alphabetically.
        order = np.random.default_rng(args.seed + k).permutation(len(lengths))
        for i in order:
            h = int(hashes[i])
            if h in seen:
                dups += 1; continue
            if used >= cap:
                break
            seen.add(h)
            # A copy, so the source's full array (Stack Exchange alone is ~10 GB) can
            # be freed once its capped selection is made.
            doc = tokens[offsets[i]:offsets[i + 1]].copy()
            if h % VAL_NEW_MOD == 0:
                val_docs.append(doc); n_val += len(doc)
            else:
                train_docs.append(doc); n_train += len(doc)
            used += len(doc)
        del tokens, lengths, hashes, offsets
        manifest_sources[source] = {**stats, "duplicates_dropped": dups, "cap": cap,
                                    "train_tokens": n_train, "val_new_tokens": n_val,
                                    "license": LICENSES.get(source, "")}
        print(f"  {source}: {n_train:,} train + {n_val:,} val_new tokens "
              f"({dups:,} duplicates dropped)")
    rng.shuffle(train_docs)
    train_paths, train_tokens = pack_documents(train_docs, doc_id, out_dir / "train",
                                               args.start_index)
    val_paths, val_tokens = pack_documents(val_docs, doc_id, out_dir / "val_new", 0)
    manifest = {"sources": manifest_sources, "train_shards": len(train_paths),
                "train_tokens": train_tokens, "val_new_shards": len(val_paths),
                "val_new_tokens": val_tokens, "seed": args.seed, "seq_len": SEQ_LEN,
                "tokens_per_shard": TOKENS_PER_SHARD, "dtype": "uint16",
                "filters": {"min_doc_tokens": MIN_DOC_TOKENS, "min_words": MIN_WORDS,
                            "val_overlap_max": VAL_OVERLAP_MAX, "ngram": NGRAM_SIZE,
                            "benchmark_ngram": "any match drops the document"}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"packed {len(train_paths)} train shards ({train_tokens:,} tokens), "
          f"{len(val_paths)} val_new shards ({val_tokens:,} tokens)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--source", required=True, choices=sorted(SOURCES))
    e.add_argument("--max-tokens", type=float, required=True)
    e.add_argument("--max-files", type=int, default=0)
    e.add_argument("--workdir", required=True)
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    e.add_argument("--tok-dir", default=str(REPO_ROOT / "tokenizer"))
    e.add_argument("--bench", nargs="+", default=[str(REPO_ROOT / "eval" / "benchmark.json"),
                                                 str(REPO_ROOT / "eval" / "mmlu_cs_benchmark.json")])
    e.add_argument("--val-shards", default=str(REPO_ROOT / "data" / "shards" / "val"))
    p = sub.add_parser("pack")
    p.add_argument("--workdir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sources", nargs="+",
                   default=["libretexts", "stackexchange", "cosmopedia", "fineweb", "arxiv"])
    p.add_argument("--caps", nargs="*", default=[], help="source=max_tokens, e.g. fineweb=1.1e9")
    p.add_argument("--start-index", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tok-dir", default=str(REPO_ROOT / "tokenizer"))
    args = ap.parse_args()
    return extract(args) if args.cmd == "extract" else pack(args)


if __name__ == "__main__":
    sys.exit(main())
