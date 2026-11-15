"""Static checks for the paper sources (no LaTeX install needed).

Checks that every \\cite key exists in references.bib, every \\ref has a \\label, every \\input file exists,
braces balance, and \\begin/\\end counts match. Lists the \\pending markers still in the text.

Run from the repo root:  python paper/check_tex.py
"""
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
BS = "\\"


def command_args(text, name):
    return re.findall(re.escape(BS + name) + r"\{([^}]*)\}", text)


def main():
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    tables = {p.name: p.read_text(encoding="utf-8") for p in sorted((PAPER / "tables").glob("*.tex"))}
    bib = (PAPER / "references.bib").read_text(encoding="ascii")
    body = tex + "".join(tables.values())

    cites = {k.strip() for grp in command_args(body, "cite") for k in grp.split(",")}
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    labels = set(command_args(body, "label"))
    refs = set(command_args(body, "ref"))
    inputs = command_args(tex, "input")
    problems = []
    if cites - keys:
        problems.append(f"cited but not in bib: {sorted(cites - keys)}")
    if refs - labels:
        problems.append(f"ref without label: {sorted(refs - labels)}")
    missing = [i for i in inputs if not (PAPER / i).exists()]
    if missing:
        problems.append(f"missing \\input files: {missing}")
    for name, text in [("main.tex", tex), *tables.items()]:
        if text.count("{") != text.count("}"):
            problems.append(f"{name}: braces unbalanced ({text.count('{')} open, {text.count('}')} close)")
        b, e = len(command_args(text, "begin")), len(command_args(text, "end"))
        if b != e:
            problems.append(f"{name}: {b} \\begin vs {e} \\end")
    pending = re.findall(re.escape(BS + "pending") + r"\{([^}]*)\}", tex)[1:]  # skip the macro definition
    print(f"citations: {len(cites)} used, {len(keys)} in bib; uncited bib entries: {sorted(keys - cites) or 'none'}")
    print(f"labels: {len(labels)}, refs: {len(refs)}, inputs: {len(inputs)}")
    print(f"pending markers: {len(pending)}")
    for p in pending:
        print(f"  - {p}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
