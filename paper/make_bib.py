"""Write paper/references.bib from verified metadata.

Titles, author lists and years for arXiv papers come from paper/data/arxiv_metadata.json (arXiv API, 2026-11-04).
Every entry carries a DOI that was checked against the registry on 2026-11-12:
- a published version is cited where Crossref holds its record (publisher DOI);
- every other arXiv paper cites its arXiv preprint with the arXiv DOI 10.48550/arXiv.<id> (checked in DataCite);
- the GPT-2 report and the SmolLM-Corpus dataset have no DOI and carry their URLs instead.

The DOI is also written in the note field as "doi: <DOI>", because the IEEEtran.bst style does not print the
doi field itself. The doi field stays for machine readers.

Run from the repo root:  python paper/make_bib.py
"""
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "paper" / "data" / "arxiv_metadata.json").read_text(encoding="utf-8"))

ICLR = "International Conference on Learning Representations (ICLR)"
NEURIPS = "Advances in Neural Information Processing Systems (NeurIPS)"
ACL = "Proc. Annual Meeting of the Association for Computational Linguistics (ACL)"
EACL = "Proc. Conference of the European Chapter of the Association for Computational Linguistics (EACL)"
SUSTAINLP = "Proc. Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)"
ICML = "International Conference on Machine Learning (ICML)"
NEUROCOMPUTING = {"journal": "Neurocomputing", "volume": "568", "pages": "127063"}

# key -> (arXiv id, venue, year). venue is None for a preprint; a dict is a journal article.
ENTRIES = {
    "vaswani2017attention": ("1706.03762", NEURIPS, "2017"),
    "dehghani2019universal": ("1807.03819", ICLR, "2019"),
    "lan2020albert": ("1909.11942", ICLR, "2020"),
    "takase2023lessons": ("2104.06022", SUSTAINLP, "2023"),
    "liu2024mobilellm": ("2402.14905", ICML, "2024"),
    "bae2025relaxed": ("2410.20672", ICLR, "2025"),
    "giannou2023looped": ("2301.13196", None, "2023"),
    "yang2024looped": ("2311.12424", ICLR, "2024"),
    "saunshi2025reasoning": ("2502.17416", ICLR, "2025"),
    "geiping2025scaling": ("2502.05171", NEURIPS, "2025"),
    "su2021roformer": ("2104.09864", NEUROCOMPUTING, "2024"),
    "zhang2019rmsnorm": ("1910.07467", NEURIPS, "2019"),
    "shazeer2020glu": ("2002.05202", None, "2020"),
    "loshchilov2019adamw": ("1711.05101", ICLR, "2019"),
    "press2017tying": ("1608.05859", EACL, "2017"),
    "hendrycks2021mmlu": ("2009.03300", ICLR, "2021"),
    "brown2020gpt3": ("2005.14165", NEURIPS, "2020"),
    "kaplan2020scaling": ("2001.08361", None, "2020"),
    "hoffmann2022chinchilla": ("2203.15556", NEURIPS, "2022"),
    "penedo2024fineweb": ("2406.17557", NEURIPS, "2024"),
    "kocetkov2022stack": ("2211.15533", None, "2022"),
    "lee2022dedup": ("2107.06499", ACL, "2022"),
    "sennrich2016bpe": ("1508.07909", ACL, "2016"),
    "kandpal2025commonpile": ("2506.05209", NEURIPS, "2025"),
    "paszke2019pytorch": ("1912.01703", NEURIPS, "2019"),
}

# Publisher DOIs, each checked against the Crossref record for that DOI (title and year).
PUBLISHER_DOI = {
    "takase2023lessons": "10.18653/v1/2023.sustainlp-1.5",
    "geiping2025scaling": "10.52202/085713-1380",
    "su2021roformer": "10.1016/j.neucom.2023.127063",
    "press2017tying": "10.18653/v1/e17-2025",
    "hoffmann2022chinchilla": "10.52202/068431-2176",
    "penedo2024fineweb": "10.52202/079017-0970",
    "kandpal2025commonpile": "10.52202/085713-1918",
    "lee2022dedup": "10.18653/v1/2022.acl-long.577",
    "sennrich2016bpe": "10.18653/v1/p16-1162",
}

MANUAL = r"""
@techreport{radford2019language,
  author      = {Alec Radford and Jeffrey Wu and Rewon Child and David Luan and Dario Amodei and Ilya Sutskever},
  title       = {{Language Models are Unsupervised Multitask Learners}},
  institution = {OpenAI},
  year        = {2019},
  note        = {Available: \url{https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf}}
}

@inproceedings{broder1997resemblance,
  author    = {Andrei Z. Broder},
  title     = {{On the Resemblance and Containment of Documents}},
  booktitle = {Proc. Compression and Complexity of SEQUENCES},
  pages     = {21--29},
  year      = {1997},
  note      = {doi: 10.1109/sequen.1997.666900},
  doi       = {10.1109/sequen.1997.666900}
}

@misc{smollmcorpus2024,
  author       = {{Hugging Face}},
  title        = {{SmolLM-Corpus}},
  howpublished = {\url{https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus}},
  note         = {Dataset (FineWeb-Edu-dedup and Cosmopedia v2 subsets used)},
  year         = {2024}
}
"""

ACCENTS = {"́": "'", "̀": "`", "̌": "v", "̈": '"', "̂": "^", "̃": "~",
           "̧": "c", "̊": "r", "̆": "u", "̄": "="}
SPECIAL = {"ß": r"{\ss}", "ø": r"{\o}", "Ø": r"{\O}", "ł": r"{\l}", "Ł": r"{\L}", "æ": r"{\ae}", "đ": r"{\dj}"}


def latex(text):
    out = []
    for ch in unicodedata.normalize("NFC", text):
        if ord(ch) < 128:
            out.append(ch)
            continue
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
            continue
        base, *marks = unicodedata.normalize("NFD", ch)
        if len(marks) != 1 or marks[0] not in ACCENTS:
            raise ValueError(f"no LaTeX mapping for {ch!r} in {text!r}")
        cmd = ACCENTS[marks[0]]
        b = r"\i" if base == "i" else base
        out.append(f"{{\\{cmd}{{{b}}}}}" if cmd.isalpha() else f"{{\\{cmd}{b}}}")
    return "".join(out)


FIXED_NAMES = {
    # BibTeX would split these wrongly: a lowercase part after a hyphen reads as a "von" part, and
    # two-word surnames read as a middle name plus surname.
    "Jy-yong Sohn": "{Jy-yong} Sohn",
    "Carlos Muñoz Ferrandis": "Carlos {Muñoz Ferrandis}",
    "Leandro Von Werra": "Leandro {von Werra}",
}


def name(n):
    n = n.replace("Ben allal", "Ben Allal").replace("Loubna Ben Allal", "Loubna {Ben Allal}")
    n = FIXED_NAMES.get(n, n)
    return latex(n)


def entry(key, aid, venue, year):
    r = META[aid]
    names = [name(a) for a in r["authors"]]
    authors = " and ".join(names) if len(names) <= 6 else names[0] + " and others"
    title = latex(r["title"]).replace("--", "{-}{-}")
    doi = PUBLISHER_DOI.get(key, f"10.48550/arXiv.{aid}")
    if isinstance(venue, dict):
        return (f"@article{{{key},\n  author  = {{{authors}}},\n  title   = {{{{{title}}}}},\n"
                f"  journal = {{{venue['journal']}}},\n  volume  = {{{venue['volume']}}},\n"
                f"  pages   = {{{venue['pages']}}},\n  year    = {{{year}}},\n"
                f"  doi     = {{{doi}}},\n  note    = {{arXiv:{aid}, doi: {doi}}}\n}}\n")
    if venue:
        return (f"@inproceedings{{{key},\n  author    = {{{authors}}},\n  title     = {{{{{title}}}}},\n"
                f"  booktitle = {{{venue}}},\n  year      = {{{year}}},\n  doi       = {{{doi}}},\n"
                f"  note      = {{arXiv:{aid}, doi: {doi}}}\n}}\n")
    return (f"@misc{{{key},\n  author        = {{{authors}}},\n  title         = {{{{{title}}}}},\n"
            f"  howpublished  = {{arXiv preprint arXiv:{aid}}},\n  year          = {{{year}}},\n"
            f"  note          = {{doi: {doi}}},\n  doi           = {{{doi}}}\n}}\n")


def main():
    parts = ["% Generated by paper/make_bib.py from paper/data/arxiv_metadata.json (arXiv API, 2026-11-04).\n"
             "% DOIs checked against Crossref and DataCite on 2026-11-12; see the comments in make_bib.py.\n"]
    for key, (aid, venue, year) in ENTRIES.items():
        parts.append(entry(key, aid, venue, year))
    parts.append(MANUAL.lstrip("\n"))
    out = ROOT / "paper" / "references.bib"
    out.write_text("\n".join(parts), encoding="ascii")
    print(f"{len(ENTRIES) + 3} entries written to {out}")


if __name__ == "__main__":
    main()
