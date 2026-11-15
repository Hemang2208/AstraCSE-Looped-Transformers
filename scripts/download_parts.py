"""Download corpus in 8 independent parts — run one at a time, track status.

Usage:
    python scripts/download_parts.py --status          # dekho kya done, kya bacha
    python scripts/download_parts.py --part 1          # Part 1 chalao
    python scripts/download_parts.py --part 2          # Part 2 chalao
    ... and so on up to --part 8

Each part saves its status to data/.parts_status.json
Resume-safe: agar beech mein ruk gaya, dobara --part N chalao.

Parts:
  Part 1 — Wikipedia CS articles        (~200M tokens, ~3 GB download)
  Part 2 — ArXiv: Algorithms + Theory   (~100M tokens, ~2 GB download)
  Part 3 — ArXiv: OS + Networks + Arch  (~100M tokens, ~2 GB download)
  Part 4 — ArXiv: DB + SE + Security    (~100M tokens, ~2 GB download)
  Part 5 — The Stack: Python            (~200M tokens, ~8 GB download)
  Part 6 — The Stack: C + C++           (~200M tokens, ~12 GB download)
  Part 7 — The Stack: Java + JavaScript (~150M tokens, ~8 GB download)
  Part 8 — The Stack: Go + Rust         (~100M tokens, ~6 GB download)
                                         ─────────────────────────────
  Total                                  ~1.15B tokens, ~43 GB download
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
STATUS_FILE = Path("data/.parts_status.json")
MIN_FREE_DISK_GB = 5
MAX_FILE_BYTES = 50 * 1024 * 1024   # 50 MB per output file
CHECKPOINT_EVERY = 2_000
BYTES_PER_TOKEN = 4.0

# ---------------------------------------------------------------------------
# Part definitions
# ---------------------------------------------------------------------------

PARTS = {
    1: {
        "name":   "Wikipedia CS articles",
        "tokens": "~200M",
        "dl_gb":  "~3 GB",
        "desc":   "English Wikipedia filtered to CS topics",
    },
    2: {
        "name":   "ArXiv: Algorithms + Theory + PL",
        "tokens": "~100M",
        "dl_gb":  "~2 GB",
        "desc":   "cs.DS, cs.CC, cs.FL, cs.PL categories",
    },
    3: {
        "name":   "ArXiv: OS + Networks + Architecture",
        "tokens": "~100M",
        "dl_gb":  "~2 GB",
        "desc":   "cs.OS, cs.DC, cs.NI, cs.AR categories",
    },
    4: {
        "name":   "ArXiv: DB + SE + Security",
        "tokens": "~100M",
        "dl_gb":  "~2 GB",
        "desc":   "cs.DB, cs.SE, cs.CR categories",
    },
    5: {
        "name":   "The Stack: Python",
        "tokens": "~200M",
        "dl_gb":  "~8 GB",
        "desc":   "MIT/Apache/BSD licensed Python code only",
    },
    6: {
        "name":   "The Stack: C + C++",
        "tokens": "~200M",
        "dl_gb":  "~12 GB",
        "desc":   "MIT/Apache/BSD licensed C and C++ code only",
    },
    7: {
        "name":   "The Stack: Java + JavaScript",
        "tokens": "~150M",
        "dl_gb":  "~8 GB",
        "desc":   "MIT/Apache/BSD licensed Java and JS code only",
    },
    8: {
        "name":   "The Stack: Go + Rust",
        "tokens": "~100M",
        "dl_gb":  "~6 GB",
        "desc":   "MIT/Apache/BSD licensed Go and Rust code only",
    },
}

# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")


def print_status() -> None:
    status = load_status()
    free_gb = shutil.disk_usage(RAW_DIR if RAW_DIR.exists() else Path(".")).free / 1024**3

    print(f"\n{'='*62}")
    print(f"  AstraCSE Corpus Download — Status")
    print(f"  Free disk: {free_gb:.1f} GB")
    print(f"{'='*62}")
    print(f"  {'Part':<6} {'Name':<36} {'Tokens':<10} {'Status'}")
    print(f"  {'─'*6} {'─'*36} {'─'*10} {'─'*12}")

    total_done = 0
    for n in range(1, 9):
        info = PARTS[n]
        pkey = f"part{n}"
        st = status.get(pkey, {})
        state = st.get("state", "pending")
        tokens_m = st.get("tokens_millions", 0)

        if state == "done":
            symbol = "[DONE]"
            tok_str = f"{tokens_m:.0f}M written"
            total_done += 1
        elif state == "running":
            symbol = "[...]"
            tok_str = f"{tokens_m:.0f}M so far"
        elif state == "partial":
            symbol = "[PART]"
            tok_str = f"{tokens_m:.0f}M (resume)"
        else:
            symbol = "[ ]"
            tok_str = info["tokens"]

        print(f"  {n:<6} {info['name']:<36} {tok_str:<18} {symbol}")

    print(f"{'─'*62}")
    print(f"  Done: {total_done}/8 parts")
    if total_done == 8:
        print(f"  ALL DONE — run pipeline: clean -> dedup -> tokenizer -> pack")
    else:
        next_part = next((n for n in range(1, 9)
                          if status.get(f"part{n}", {}).get("state") != "done"), None)
        if next_part:
            print(f"  Next: python scripts/download_parts.py --part {next_part}")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Disk guard
# ---------------------------------------------------------------------------

def check_disk(path: Path) -> bool:
    free = shutil.disk_usage(path if path.exists() else Path(".")).free
    return free > MIN_FREE_DISK_GB * 1024**3


# ---------------------------------------------------------------------------
# ChunkWriter — splits output into 50MB files
# ---------------------------------------------------------------------------

class ChunkWriter:
    def __init__(self, out_dir: Path, prefix: str):
        self.out_dir = out_dir
        self.prefix = prefix
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._idx = self._find_next()
        self._fh = None
        self._bytes = 0

    def _find_next(self) -> int:
        existing = sorted(self.out_dir.glob(f"{self.prefix}_*.txt"))
        if not existing:
            return 0
        m = re.search(r"_(\d+)\.txt$", existing[-1].name)
        return (int(m.group(1)) + 1) if m else 0

    def _open_next(self) -> None:
        if self._fh:
            self._fh.close()
        p = self.out_dir / f"{self.prefix}_{self._idx:05d}.txt"
        self._fh = open(p, "a", encoding="utf-8")
        self._bytes = p.stat().st_size if p.exists() else 0
        self._idx += 1

    def write(self, text: str) -> None:
        # NUL-byte prefix: an unambiguous per-document delimiter (never occurs
        # in valid UTF-8 text or source code) so clean_text.py can recover
        # individual documents from a chunk file — plain "\n\n" alone is not
        # unique enough (ordinary blank lines inside a document look the same).
        payload = "\x00" + text
        b = payload.encode("utf-8")
        if self._fh is None or self._bytes + len(b) > MAX_FILE_BYTES:
            self._open_next()
        self._fh.write(payload + "\n\n")
        self._bytes += len(b) + 2

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


# ---------------------------------------------------------------------------
# Wikipedia filter keywords
# ---------------------------------------------------------------------------

WIKI_CS_KEYWORDS = {
    "algorithm", "data structure", "sorting", "graph theory", "binary tree",
    "hash table", "linked list", "stack", "queue", "heap", "trie",
    "operating system", "process", "thread", "deadlock", "semaphore",
    "virtual memory", "paging", "file system", "scheduler",
    "computer network", "protocol", "tcp", "http", "dns", "router",
    "database", "sql", "relational", "query", "index", "transaction",
    "compiler", "interpreter", "programming language", "python", "java",
    "computer architecture", "cpu", "cache", "instruction set", "risc",
    "software engineering", "testing", "debugging", "version control",
    "cryptography", "encryption", "cybersecurity", "vulnerability",
    "automata", "turing machine", "complexity", "np-complete",
    "machine learning", "neural network", "deep learning",
    "computer science", "discrete mathematics", "boolean algebra",
    "big o notation", "recursion", "dynamic programming",
}

def _is_cs(title: str, text: str) -> bool:
    t = (title + " " + text[:400]).lower()
    return any(kw in t for kw in WIKI_CS_KEYWORDS)


# ---------------------------------------------------------------------------
# Allowed licenses for code
# ---------------------------------------------------------------------------

ALLOWED_LICENSES = {
    "MIT", "APACHE-2.0", "BSD-2-CLAUSE", "BSD-3-CLAUSE",
    "ISC", "UNLICENSE", "CC0-1.0", "0BSD",
}

def _license_ok(licenses) -> bool:
    """Accepts a single license string OR an iterable of them (bigcode/the-stack
    reports licenses as a list under max_stars_repo_licenses etc — there is no
    plain "license" field on this dataset)."""
    if not licenses:
        return False
    if isinstance(licenses, str):
        licenses = re.split(r"[,\s]+", licenses)
    return any(lic and lic.strip().upper() in ALLOWED_LICENSES for lic in licenses)

def _is_generated(content: str) -> bool:
    first = content.split("\n", 1)[0].lower()
    return any(kw in first for kw in
               ("auto-generated", "do not edit", "generated by", "autogenerated",
                "this file is automatically", "machine generated"))


# ---------------------------------------------------------------------------
# Part runners
# ---------------------------------------------------------------------------

def run_part1_wikipedia(status: dict) -> None:
    """Part 1: Wikipedia CS filtered."""
    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("ERROR: pip install datasets tqdm")
        return

    out_dir = RAW_DIR / "wikipedia_large"
    pkey = "part1"
    resume_from = status.get(pkey, {}).get("docs_seen", 0)

    print(f"\n[Part 1] Wikipedia CS — resuming from doc {resume_from:,}")

    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)

    tokens = 0
    seen = 0
    written = 0

    with ChunkWriter(out_dir, "wiki") as w:
        for doc in tqdm(ds, desc="Wikipedia", unit="doc", mininterval=10):
            seen += 1
            if seen <= resume_from:
                continue

            title = doc.get("title", "")
            text  = doc.get("text", "")
            if len(text) < 200 or not _is_cs(title, text):
                continue

            if not check_disk(out_dir):
                print("\nWARNING: low disk space — stopping")
                break

            w.write(f"=== {title} ===\n\n{text}")
            tokens += len(text.encode()) // int(BYTES_PER_TOKEN)
            written += 1

            if written % CHECKPOINT_EVERY == 0:
                status["part1"] = {"state": "partial", "docs_seen": seen,
                                   "tokens_millions": tokens / 1e6}
                save_status(status)
                print(f"  {written:,} articles | {tokens/1e6:.1f}M tokens")

            if tokens >= 200_000_000:
                break

    status["part1"] = {"state": "done", "docs_seen": seen,
                       "tokens_millions": tokens / 1e6}
    save_status(status)
    print(f"\n[Part 1] Done: {written:,} articles | {tokens/1e6:.1f}M tokens")


OAI_URL  = "https://export.arxiv.org/oai2"
OAI_NS   = "http://www.openarchives.org/OAI/2.0/"
ARXIV_NS = "http://arxiv.org/OAI/arXiv/"


def _oai_fetch(resumption_token: str | None, max_retry: int = 6) -> str | None:
    """One ListRecords request against arXiv's OAI-PMH endpoint.

    Cursor-based (resumptionToken), not offset-based — this is the interface
    ArXiv itself recommends for bulk metadata harvesting and does not suffer
    the deep-pagination 500s that the search API (start=N) hits past ~10k.
    Returns response body text, or None if all retries were exhausted.
    """
    if resumption_token:
        params = {"verb": "ListRecords", "resumptionToken": resumption_token}
    else:
        params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": "cs"}

    for attempt in range(max_retry):
        try:
            resp = requests.get(OAI_URL, params=params, timeout=90)
        except Exception as exc:
            wait = 15 * (2 ** attempt)
            print(f"\n  network error ({type(exc).__name__}: {exc}) — retrying in {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code == 503:
            wait = int(resp.headers.get("Retry-After", 20))
            print(f"\n  OAI rate limit (503) — waiting {wait}s as instructed")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.text

        wait = 15 * (2 ** attempt)
        print(f"\n  HTTP {resp.status_code} — retrying in {wait}s")
        time.sleep(wait)

    return None


def _parse_oai_records(xml_text: str):
    """Returns (records, next_resumption_token, error_code).

    records is a list of (title, abstract, categories_set).
    error_code is set (e.g. 'badResumptionToken') when the OAI server
    returned an <error> element instead of records.
    """
    root = ET.fromstring(xml_text)

    error_el = root.find(f"{{{OAI_NS}}}error")
    if error_el is not None:
        return [], None, error_el.get("code", "unknown")

    list_records = root.find(f"{{{OAI_NS}}}ListRecords")
    if list_records is None:
        return [], None, None

    records = []
    for rec in list_records.findall(f"{{{OAI_NS}}}record"):
        header = rec.find(f"{{{OAI_NS}}}header")
        if header is not None and header.get("status") == "deleted":
            continue

        metadata = rec.find(f"{{{OAI_NS}}}metadata")
        if metadata is None:
            continue
        meta = metadata.find(f"{{{ARXIV_NS}}}arXiv")
        if meta is None:
            continue

        title    = " ".join((meta.findtext(f"{{{ARXIV_NS}}}title") or "").split())
        abstract = " ".join((meta.findtext(f"{{{ARXIV_NS}}}abstract") or "").split())
        cats_raw = (meta.findtext(f"{{{ARXIV_NS}}}categories") or "").strip()
        records.append((title, abstract, set(cats_raw.split())))

    token_el = list_records.find(f"{{{OAI_NS}}}resumptionToken")
    next_token = None
    if token_el is not None and token_el.text and token_el.text.strip():
        next_token = token_el.text.strip()

    return records, next_token, None


def run_arxiv_categories(
    part_num: int,
    categories: set,
    out_subdir: str,
    target_tokens: int,
    status: dict,
) -> None:
    """Harvest ArXiv CS papers via OAI-PMH bulk metadata harvesting, filtered
    client-side to this part's categories.  Stops early (state=done, under
    target) if the whole `cs` archive is exhausted before target_tokens is
    reached — abstracts are short, so narrow categories may not reach ~100M
    tokens alone; that is expected and not an error."""
    try:
        from tqdm import tqdm
    except ImportError:
        print("ERROR: pip install tqdm")
        return

    out_dir = RAW_DIR / out_subdir
    pkey    = f"part{part_num}"
    st      = status.get(pkey, {})

    resumption_token = st.get("resumption_token")
    tokens   = int(st.get("tokens_millions", 0) * 1e6)
    written  = st.get("papers_written", 0)
    scanned  = st.get("records_scanned", 0)

    print(f"\n[Part {part_num}] ArXiv OAI-PMH harvest — categories {sorted(categories)}")
    print(f"  Endpoint : {OAI_URL}  (set=cs, filtered client-side)")
    print(f"  Resuming : {written:,} papers written, {scanned:,} CS records scanned, "
          f"{tokens/1e6:.2f}M tokens so far")

    with ChunkWriter(out_dir, f"arxiv_p{part_num}") as w:
        with tqdm(desc=f"ArXiv p{part_num}", unit="rec",
                  initial=scanned, mininterval=10) as pbar:

            while tokens < target_tokens:
                xml_text = _oai_fetch(resumption_token)
                if xml_text is None:
                    print("\n  OAI harvest failed after retries — saving progress and stopping")
                    break

                try:
                    records, next_token, error_code = _parse_oai_records(xml_text)
                except ET.ParseError as exc:
                    print(f"\n  XML parse error: {exc} — stopping")
                    break

                if error_code == "badResumptionToken":
                    print("\n  Resumption token expired — restarting harvest from beginning "
                          "of the cs archive (already-written papers are kept)")
                    resumption_token = None
                    continue
                elif error_code:
                    print(f"\n  OAI error '{error_code}' — stopping")
                    break

                low_disk = False
                for title, abstract, cats in records:
                    scanned += 1
                    if not (cats & categories):
                        continue

                    text = f"{title}\n\n{abstract}"
                    if len(text) < 100:
                        continue
                    if not check_disk(out_dir):
                        print("\nWARNING: low disk space — stopping")
                        low_disk = True
                        break

                    w.write(text)
                    tokens  += len(text.encode()) // int(BYTES_PER_TOKEN)
                    written += 1

                    if tokens >= target_tokens:
                        break

                pbar.update(len(records))

                status[pkey] = {
                    "state":             "partial",
                    "resumption_token":  next_token,
                    "papers_written":    written,
                    "records_scanned":   scanned,
                    "tokens_millions":   tokens / 1e6,
                }
                save_status(status)

                if low_disk or tokens >= target_tokens:
                    break

                if next_token is None:
                    print(f"\n  Entire ArXiv 'cs' archive scanned ({scanned:,} records) — "
                          f"no more data available for these categories")
                    break

                resumption_token = next_token
                time.sleep(1.0)   # polite delay between OAI batches

    status[pkey] = {
        "state":             "done",
        "resumption_token":  None,
        "papers_written":    written,
        "records_scanned":   scanned,
        "tokens_millions":   tokens / 1e6,
    }
    save_status(status)
    print(f"\n[Part {part_num}] Done: {written:,} papers | {tokens/1e6:.2f}M tokens "
          f"(scanned {scanned:,} CS records total)")


def run_stack_languages(
    part_num: int,
    languages: list,
    target_tokens: int,
    max_files_per_lang: int,
    status: dict,
) -> None:
    """Generic Stack runner for a list of languages."""
    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("ERROR: pip install datasets tqdm")
        return

    pkey = f"part{part_num}"
    status[pkey] = status.get(pkey, {})

    def _part_total_millions() -> float:
        return sum(status[pkey].get(f"{l}_tokens_millions", 0.0) for l in languages)

    for lang in languages:
        out_dir = RAW_DIR / "the_stack" / lang

        if status[pkey].get(f"{lang}_done"):
            print(f"\n[Part {part_num}] The Stack: {lang} — already done, skipping")
            continue

        resume_from = status[pkey].get(f"{lang}_seen", 0)
        # Fixed baseline captured once per resume — every checkpoint below sets
        # (does not add to) tokens_millions, so re-checkpointing mid-language
        # can never compound (the earlier bug: adding an already-cumulative
        # `tokens` onto an already-cumulative saved value, every checkpoint).
        lang_base_millions = status[pkey].get(f"{lang}_tokens_millions", 0.0)

        print(f"\n[Part {part_num}] The Stack: {lang} — resuming from {resume_from:,}")

        try:
            ds = load_dataset("bigcode/the-stack",
                              data_dir=f"data/{lang}",
                              split="train", streaming=True)
        except Exception as e:
            print(f"  FAILED to load {lang}: {e}")
            print(f"  Make sure you accepted terms at: huggingface.co/datasets/bigcode/the-stack")
            continue

        tokens = 0
        seen = 0
        written = 0
        low_disk = False

        with ChunkWriter(out_dir, f"stack_{lang}") as w:
            for doc in tqdm(ds, desc=f"Stack/{lang}", unit="file", mininterval=10):
                seen += 1
                if seen <= resume_from:
                    continue

                licenses = set(doc.get("max_stars_repo_licenses") or []) \
                         | set(doc.get("max_issues_repo_licenses") or []) \
                         | set(doc.get("max_forks_repo_licenses") or [])
                content = doc.get("content", "").strip()

                if not _license_ok(licenses):
                    continue
                if len(content) < 100 or len(content) > 50_000:
                    continue
                if _is_generated(content):
                    continue

                if not check_disk(out_dir):
                    print("\nWARNING: low disk space — stopping")
                    low_disk = True
                    break

                w.write(content)
                tokens += len(content.encode()) // int(BYTES_PER_TOKEN)
                written += 1

                if written % CHECKPOINT_EVERY == 0:
                    status[pkey]["state"] = "partial"
                    status[pkey][f"{lang}_seen"] = seen
                    status[pkey][f"{lang}_tokens_millions"] = lang_base_millions + tokens / 1e6
                    status[pkey]["tokens_millions"] = _part_total_millions()
                    save_status(status)
                    print(f"  {written:,} files | {tokens/1e6:.1f}M tokens "
                          f"({_part_total_millions():.1f}M total for this part)")

                if written >= max_files_per_lang or tokens >= target_tokens // len(languages):
                    break

        status[pkey][f"{lang}_seen"] = seen
        status[pkey][f"{lang}_tokens_millions"] = lang_base_millions + tokens / 1e6
        status[pkey]["tokens_millions"] = _part_total_millions()
        if low_disk:
            save_status(status)
            print(f"\n[Part {part_num}] Stopped early on {lang} — low disk space. "
                  f"Not marked done; resume with the same --part command.")
            return
        status[pkey][f"{lang}_done"] = True
        save_status(status)

    status[pkey]["state"] = "done"
    save_status(status)
    print(f"\n[Part {part_num}] Done: {_part_total_millions():.1f}M tokens total")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_part(part_num: int) -> None:
    status = load_status()
    pkey = f"part{part_num}"

    if status.get(pkey, {}).get("state") == "done":
        print(f"\n[Part {part_num}] Already DONE — skipping.")
        print(f"  Run: python scripts/download_parts.py --status")
        return

    info = PARTS[part_num]
    print(f"\n{'='*60}")
    print(f"Starting Part {part_num}: {info['name']}")
    print(f"  Expected tokens:  {info['tokens']}")
    print(f"  Download size:    {info['dl_gb']}")
    print(f"  Description:      {info['desc']}")
    print(f"{'='*60}")
    print(f"Press Ctrl+C to pause — progress is saved automatically.")
    print()

    try:
        if part_num == 1:
            run_part1_wikipedia(status)

        elif part_num == 2:
            run_arxiv_categories(
                part_num=2,
                categories={"cs.DS", "cs.CC", "cs.FL", "cs.PL", "cs.LG"},
                out_subdir="arxiv_algo_theory",
                target_tokens=100_000_000,
                status=status,
            )

        elif part_num == 3:
            run_arxiv_categories(
                part_num=3,
                categories={"cs.OS", "cs.DC", "cs.NI", "cs.AR"},
                out_subdir="arxiv_os_networks",
                target_tokens=100_000_000,
                status=status,
            )

        elif part_num == 4:
            run_arxiv_categories(
                part_num=4,
                categories={"cs.DB", "cs.SE", "cs.CR"},
                out_subdir="arxiv_db_se_sec",
                target_tokens=100_000_000,
                status=status,
            )

        elif part_num == 5:
            run_stack_languages(5, ["python"],         200_000_000, 500_000, status)

        elif part_num == 6:
            run_stack_languages(6, ["c", "c++"],       200_000_000, 300_000, status)

        elif part_num == 7:
            run_stack_languages(7, ["java", "javascript"], 150_000_000, 300_000, status)

        elif part_num == 8:
            run_stack_languages(8, ["go", "rust"],     100_000_000, 200_000, status)

    except KeyboardInterrupt:
        print(f"\n\n[Part {part_num}] Paused by user — progress saved.")
        print(f"Resume with: python scripts/download_parts.py --part {part_num}")

    print_status()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download corpus in 8 parts. Track status anytime.")
    ap.add_argument("--part",   type=int, choices=range(1, 9),
                    help="Which part to run (1-8)")
    ap.add_argument("--status", action="store_true",
                    help="Show current status and exit")
    ap.add_argument("--reset",  type=int, choices=range(1, 9),
                    help="Reset a part's status to pending (to re-download)")
    args = ap.parse_args()

    if args.status or (not args.part and not args.reset):
        print_status()
        return 0

    if args.reset:
        status = load_status()
        status.pop(f"part{args.reset}", None)
        save_status(status)
        print(f"Part {args.reset} reset to pending.")
        return 0

    if args.part:
        run_part(args.part)

    return 0


if __name__ == "__main__":
    sys.exit(main())
