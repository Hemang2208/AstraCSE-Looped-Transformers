"""Phase 0 — Download all approved sources into data/raw/<subject>/.

Sources: SOURCES.md approved list only.
- Resume-safe: already-downloaded files are skipped.
- Each source goes into data/raw/<subject>/ folder.
- Wikipedia/Wikibooks: uses pre-extracted plaintext dumps (much faster than HTML scraping).
- OSTEP, OpenDSA, ODS: chapter-by-chapter HTML fetch + save.
- IETF RFCs: bulk text download for networking RFCs.
- OWASP: GitHub repo clone of their docs.
- Python/PostgreSQL/SQLite docs: official doc pages.

Run:
    python scripts/download_data.py
    python scripts/download_data.py --sources wikipedia ietf ostep  # specific sources only
    python scripts/download_data.py --list   # show what will be downloaded
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path("data/raw")
HEADERS = {
    "User-Agent": "AstraCSE-DataCollector/1.0 (academic non-commercial ML project; "
                  "contact: github.com/bhargav-abhay/AstraCSE)"
}
TIMEOUT = 30


def log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def _get(url: str, stream: bool = False) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=stream)
    r.raise_for_status()
    return r


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download url → dest. Returns False if already exists (skipped)."""
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  SKIP (exists): {dest.name}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  Downloading: {desc or dest.name}")
    try:
        r = _get(url, stream=True)
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                          desc=dest.name[:40], leave=False) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        log(f"  FAILED {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def save_text(text: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  SKIP (exists): {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    log(f"  SAVED: {dest.name}  ({len(text):,} chars)")


def fetch_html_text(url: str) -> str:
    """Fetch URL, return raw HTML as text."""
    r = _get(url)
    return r.text


def _rate_limit() -> None:
    time.sleep(1.5)   # be polite to servers — 1.5s avoids Wikipedia 429


# ---------------------------------------------------------------------------
# Wikipedia CS dump (pre-extracted plaintext — much smaller than full dump)
# Uses the Kiwix/Wikipedia plaintext extraction endpoint
# ---------------------------------------------------------------------------

WIKIPEDIA_CS_ARTICLES = [
    # Data Structures
    "Array_(data_structure)", "Linked_list", "Stack_(abstract_data_type)",
    "Queue_(abstract_data_type)", "Hash_table", "Binary_search_tree",
    "AVL_tree", "Red%E2%80%93black_tree", "B-tree", "Heap_(data_structure)",
    "Graph_(abstract_data_type)", "Trie", "Bloom_filter", "Skip_list",
    # Algorithms
    "Sorting_algorithm", "Quicksort", "Merge_sort", "Heapsort",
    "Binary_search_algorithm", "Breadth-first_search", "Depth-first_search",
    "Dijkstra%27s_algorithm", "Bellman%E2%80%93Ford_algorithm",
    "Floyd%E2%80%93Warshall_algorithm", "Dynamic_programming",
    "Greedy_algorithm", "Divide-and-conquer_algorithm", "Big_O_notation",
    "Computational_complexity_theory", "NP-completeness",
    # Computer Networks
    "OSI_model", "Internet_protocol_suite", "Transmission_Control_Protocol",
    "Internet_Protocol", "Domain_Name_System", "Hypertext_Transfer_Protocol",
    "HTTPS", "Border_Gateway_Protocol", "Routing", "Network_switch",
    "Firewall_(computing)", "Virtual_private_network",
    # Operating Systems
    "Operating_system", "Process_(computing)", "Thread_(computing)",
    "Scheduling_(computing)", "Virtual_memory", "Paging", "File_system",
    "Deadlock", "Semaphore_(programming)", "Mutual_exclusion",
    "Interrupt", "System_call", "Memory_management",
    # Computer Architecture
    "Von_Neumann_architecture", "Central_processing_unit", "Cache_(computing)",
    "Instruction_set_architecture", "RISC", "CISC", "Pipeline_(computing)",
    "Memory_hierarchy", "Direct_memory_access", "GPU", "Multiprocessing",
    # Database Systems
    "Database", "Relational_database", "SQL", "Database_normalization",
    "ACID", "Database_transaction", "B-tree", "Query_optimization",
    "NoSQL", "Database_index",
    # Theory of Computation
    "Turing_machine", "Finite-state_machine", "Pushdown_automaton",
    "Regular_language", "Context-free_language", "Halting_problem",
    "Chomsky_hierarchy", "Lambda_calculus", "P_versus_NP_problem",
    "Computability_theory",
    # Software Engineering
    "Software_engineering", "SOLID", "Design_pattern_(computer_science)",
    "Agile_software_development", "Test-driven_development",
    "Version_control", "Continuous_integration", "Software_testing",
    "Object-oriented_programming", "Functional_programming",
    # Computer Programming
    "Computer_programming", "Algorithm", "Data_type", "Variable_(computer_science)",
    "Control_flow", "Recursion_(computer_science)", "Object-oriented_programming",
    "Compiler", "Interpreter_(computing)", "Debugging",
    # Cybersecurity
    "Computer_security", "Cryptography", "SQL_injection",
    "Cross-site_scripting", "Buffer_overflow", "Public-key_cryptography",
    "Transport_Layer_Security", "Firewall_(computing)",
    "Intrusion_detection_system", "OWASP",
]

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def _wiki_fetch(api_url: str, title: str) -> str:
    """Fetch Wikipedia/Wikibooks article text with retry on 429."""
    for attempt in range(4):
        try:
            r = requests.get(api_url, params={
                "action": "query", "prop": "extracts", "explaintext": True,
                "titles": title, "format": "json", "redirects": 1,
            }, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 429:
                wait = 10 * (2 ** attempt)
                log(f"    429 rate limit — waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            pages = r.json()["query"]["pages"]
            page = next(iter(pages.values()))
            return page.get("extract", "")
        except requests.HTTPError as e:
            if attempt < 3:
                time.sleep(10 * (2 ** attempt))
            else:
                raise
    return ""


def download_wikipedia(out_dir: Path) -> None:
    log("=== Wikipedia CS articles ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for article in WIKIPEDIA_CS_ARTICLES:
        if article in seen:
            continue
        seen.add(article)
        dest = out_dir / f"{article}.txt"
        if dest.exists() and dest.stat().st_size > 0:
            log(f"  SKIP: {article}")
            continue
        try:
            title = article.replace("%E2%80%93", "–").replace("%27", "'")
            text = _wiki_fetch(WIKIPEDIA_API, title)
            if text and len(text) > 200:
                dest.write_text(text, encoding="utf-8")
                log(f"  OK: {article}  ({len(text):,} chars)")
            else:
                log(f"  EMPTY/SHORT: {article}")
            _rate_limit()
        except Exception as e:
            log(f"  FAILED: {article}: {e}")


# ---------------------------------------------------------------------------
# Wikibooks CS
# ---------------------------------------------------------------------------

WIKIBOOKS_ARTICLES = [
    "Data_Structures", "Algorithm_Implementation",
    "Introduction_to_Computer_Information_Systems",
    "Computer_Programming", "Python_Programming",
    "C_Programming", "Java_Programming",
    "Data_Structures/Introduction",
    "Data_Structures/Arrays", "Data_Structures/Linked_Lists",
    "Data_Structures/Stacks_and_Queues", "Data_Structures/Trees",
    "Data_Structures/Hash_Tables", "Data_Structures/Graphs",
]

WIKIBOOKS_API = "https://en.wikibooks.org/w/api.php"


def download_wikibooks(out_dir: Path) -> None:
    log("=== Wikibooks CS ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    for article in WIKIBOOKS_ARTICLES:
        dest = out_dir / f"{article.replace('/', '_')}.txt"
        if dest.exists() and dest.stat().st_size > 0:
            log(f"  SKIP: {article}")
            continue
        try:
            title = article.replace("_", " ")
            text = _wiki_fetch(WIKIBOOKS_API, title)
            if text and len(text) > 200:
                dest.write_text(text, encoding="utf-8")
                log(f"  OK: {article}  ({len(text):,} chars)")
            else:
                log(f"  EMPTY/SHORT: {article}")
            _rate_limit()
        except Exception as e:
            log(f"  FAILED: {article}: {e}")


# ---------------------------------------------------------------------------
# OSTEP (Operating Systems: Three Easy Pieces)
# ---------------------------------------------------------------------------

# OSTEP chapter filenames as they appear at https://pages.cs.wisc.edu/~remzi/OSTEP/
# Format: (local_name, remote_filename_without_extension)
OSTEP_CHAPTERS = [
    ("intro",               "intro"),
    ("cpu-intro",           "cpu-intro"),
    ("cpu-api",             "cpu-api"),
    ("cpu-mechanisms",      "cpu-mechanisms"),
    ("cpu-sched",           "cpu-sched"),
    ("cpu-sched-mlfq",      "cpu-sched-mlfq"),
    ("cpu-sched-lottery",   "cpu-sched-lottery"),
    ("vm-intro",            "vm-intro"),
    ("vm-api",              "vm-api"),
    ("vm-mechanism",        "vm-mechanism"),
    ("vm-segmentation",     "vm-segmentation"),
    ("vm-freespace",        "vm-freespace"),
    ("vm-paging-intro",     "vm-paging-linear"),
    ("vm-tlbs",             "vm-tlbs"),
    ("vm-smalltables",      "vm-smalltables"),
    ("vm-beyondphys",       "vm-beyondphys"),
    ("vm-complete",         "vm-complete"),
    ("threads-intro",       "threads-intro"),
    ("threads-api",         "threads-api"),
    ("threads-locks",       "threads-locks"),
    ("threads-cv",          "threads-cv"),
    ("threads-sema",        "threads-sema"),
    ("threads-bugs",        "threads-bugs"),
    ("file-disks",          "file-disks"),
    ("file-raid",           "file-raid"),
    ("file-intro",          "file-intro"),
    ("file-ffs",            "file-ffs"),
    ("file-lfs",            "file-lfs"),
    ("file-journaling",     "file-journaling"),
]
OSTEP_BASE = "https://pages.cs.wisc.edu/~remzi/OSTEP"


def download_ostep(out_dir: Path) -> None:
    log("=== OSTEP chapters ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, remote in OSTEP_CHAPTERS:
        dest = out_dir / f"{name}.pdf"
        url = f"{OSTEP_BASE}/{remote}.pdf"
        download_file(url, dest, desc=f"OSTEP: {name}")
        _rate_limit()


# ---------------------------------------------------------------------------
# Open Data Structures (Pat Morin) — PDF
# ---------------------------------------------------------------------------

def download_ods(out_dir: Path) -> None:
    log("=== Open Data Structures ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Official PDF — CC-BY licensed
    download_file(
        "https://opendatastructures.org/ods-python.pdf",
        out_dir / "ods-python.pdf",
        desc="Open Data Structures (Python edition)"
    )
    download_file(
        "https://opendatastructures.org/ods-java.pdf",
        out_dir / "ods-java.pdf",
        desc="Open Data Structures (Java edition)"
    )


# ---------------------------------------------------------------------------
# IETF RFCs — key networking RFCs as plain text
# ---------------------------------------------------------------------------

KEY_RFCS = {
    # Core Internet protocols
    "rfc791":  "IPv4",
    "rfc793":  "TCP",
    "rfc768":  "UDP",
    "rfc826":  "ARP",
    "rfc1034": "DNS",
    "rfc1035": "DNS implementation",
    "rfc2460": "IPv6",
    "rfc4291": "IPv6 addressing",
    "rfc2131": "DHCP",
    "rfc3022": "NAT",
    # Application layer
    "rfc7230": "HTTP/1.1 message syntax",
    "rfc7231": "HTTP/1.1 semantics",
    "rfc7235": "HTTP/1.1 authentication",
    "rfc9110": "HTTP semantics",
    "rfc9112": "HTTP/1.1",
    "rfc9114": "HTTP/3",
    "rfc2818": "HTTP over TLS",
    "rfc7540": "HTTP/2",
    "rfc5321": "SMTP",
    "rfc1939": "POP3",
    # Security
    "rfc8446": "TLS 1.3",
    "rfc5246": "TLS 1.2",
    "rfc4251": "SSH protocol architecture",
    "rfc4253": "SSH transport layer",
    "rfc3550": "RTP",
    # Routing
    "rfc4271": "BGP-4",
    "rfc2328": "OSPF version 2",
    "rfc1058": "RIP",
}


def download_ietf_rfcs(out_dir: Path) -> None:
    log("=== IETF RFCs ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    for rfc_id, desc in KEY_RFCS.items():
        dest = out_dir / f"{rfc_id}.txt"
        url = f"https://www.rfc-editor.org/rfc/{rfc_id}.txt"
        download_file(url, dest, desc=f"RFC {rfc_id.upper()}: {desc}")
        _rate_limit()


# ---------------------------------------------------------------------------
# OWASP — clone the OWASP Testing Guide and WSTG from GitHub
# ---------------------------------------------------------------------------

def download_owasp(out_dir: Path) -> None:
    log("=== OWASP documentation ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    repos = [
        ("https://github.com/OWASP/OWASP-Testing-Guide-v4.2",
         "OWASP-Testing-Guide"),
        ("https://github.com/OWASP/wstg",
         "OWASP-WSTG"),
        ("https://github.com/OWASP/CheatSheetSeries",
         "OWASP-CheatSheets"),
        ("https://github.com/OWASP/Top10",
         "OWASP-Top10"),
    ]

    for repo_url, folder in repos:
        dest = out_dir / folder
        if dest.exists():
            log(f"  SKIP (exists): {folder}")
            continue
        log(f"  Cloning: {folder}")
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", repo_url, str(dest)],
                check=True, capture_output=True, timeout=120
            )
            log(f"  OK: {folder}")
        except subprocess.CalledProcessError as e:
            log(f"  FAILED: {folder}: {e.stderr.decode()[:200]}")
        except FileNotFoundError:
            log("  SKIP: git not found in PATH — install git and retry")


# ---------------------------------------------------------------------------
# Python official documentation (plain text via pip download of docs pkg)
# ---------------------------------------------------------------------------

PYTHON_DOC_PAGES = [
    "tutorial/index", "tutorial/introduction", "tutorial/controlflow",
    "tutorial/datastructures", "tutorial/modules", "tutorial/inputoutput",
    "tutorial/errors", "tutorial/classes", "tutorial/stdlib",
    "tutorial/stdlib2",
    "library/functions", "library/stdtypes", "library/exceptions",
    "library/collections", "library/itertools", "library/functools",
    "library/os", "library/sys", "library/pathlib", "library/re",
    "library/json", "library/csv", "library/sqlite3", "library/threading",
    "library/multiprocessing", "library/asyncio",
    "reference/expressions", "reference/simple_stmts",
    "reference/compound_stmts", "reference/datamodel",
]
PYTHON_DOCS_BASE = "https://docs.python.org/3"


def download_python_docs(out_dir: Path) -> None:
    log("=== Python documentation ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    for page in PYTHON_DOC_PAGES:
        name = page.replace("/", "_")
        dest = out_dir / f"{name}.html"
        url = f"{PYTHON_DOCS_BASE}/{page}.html"
        download_file(url, dest, desc=f"Python docs: {page}")
        _rate_limit()


# ---------------------------------------------------------------------------
# PostgreSQL documentation
# ---------------------------------------------------------------------------

PSQL_VERSION = "16"
PSQL_CHAPTERS = [
    "intro", "tutorial-sql", "tutorial-advanced",
    "sql-syntax", "dml", "queries", "datatype", "functions",
    "indexes", "mvcc", "parallel-query", "admin", "backup",
    "high-availability", "monitoring", "performance-tips",
    "ddl", "typeconv", "plpgsql",
]
PSQL_BASE = f"https://www.postgresql.org/docs/{PSQL_VERSION}"


def download_postgresql_docs(out_dir: Path) -> None:
    log("=== PostgreSQL documentation ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    for chapter in PSQL_CHAPTERS:
        dest = out_dir / f"{chapter}.html"
        url = f"{PSQL_BASE}/{chapter}.html"
        download_file(url, dest, desc=f"PostgreSQL: {chapter}")
        _rate_limit()


# ---------------------------------------------------------------------------
# SQLite documentation
# ---------------------------------------------------------------------------

SQLITE_PAGES = [
    "lang", "lang_select", "lang_insert", "lang_update", "lang_delete",
    "lang_createtable", "lang_createindex", "lang_transaction",
    "lang_explain", "queryplanner", "opcode", "datatype3",
    "nulls", "rowidtable", "withoutrowid", "json1",
    "fts5", "rtree", "limits", "pragma",
    "isolation", "wal", "inmemorydb", "uri",
    "c3ref/intro", "capi3ref",
]
SQLITE_BASE = "https://www.sqlite.org"


def download_sqlite_docs(out_dir: Path) -> None:
    log("=== SQLite documentation ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    for page in SQLITE_PAGES:
        name = page.replace("/", "_")
        dest = out_dir / f"{name}.html"
        url = f"{SQLITE_BASE}/{page}.html"
        download_file(url, dest, desc=f"SQLite: {page}")
        _rate_limit()


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

SOURCES = {
    "wikipedia":   (download_wikipedia,    RAW_DIR / "wikipedia"),
    "wikibooks":   (download_wikibooks,    RAW_DIR / "wikibooks"),
    "ostep":       (download_ostep,        RAW_DIR / "operating_systems" / "ostep"),
    "ods":         (download_ods,          RAW_DIR / "data_structures" / "ods"),
    "ietf":        (download_ietf_rfcs,    RAW_DIR / "computer_networks" / "ietf_rfcs"),
    "owasp":       (download_owasp,        RAW_DIR / "cybersecurity" / "owasp"),
    "python_docs": (download_python_docs,  RAW_DIR / "computer_programming" / "python_docs"),
    "postgresql":  (download_postgresql_docs, RAW_DIR / "database_systems" / "postgresql"),
    "sqlite":      (download_sqlite_docs,  RAW_DIR / "database_systems" / "sqlite"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Download AstraCSE training data.")
    ap.add_argument("--sources", nargs="*", choices=list(SOURCES),
                    default=list(SOURCES),
                    help="Which sources to download (default: all)")
    ap.add_argument("--list", action="store_true",
                    help="List sources and exit")
    args = ap.parse_args()

    if args.list:
        print("Available sources:")
        for name, (_, out_dir) in SOURCES.items():
            status = "EXISTS" if out_dir.exists() else "missing"
            print(f"  {name:<16s}  -> {out_dir}  [{status}]")
        return 0

    selected = args.sources or list(SOURCES)
    log(f"Downloading {len(selected)} source(s): {', '.join(selected)}")
    log(f"Output root: {RAW_DIR.resolve()}")
    print()

    for name in selected:
        fn, out_dir = SOURCES[name]
        try:
            fn(out_dir)
        except Exception as e:
            log(f"ERROR in {name}: {e}")
        print()

    log("All downloads complete.")
    log(f"Next step: python scripts/clean_text.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
