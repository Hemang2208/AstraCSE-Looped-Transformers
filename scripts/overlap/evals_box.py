"""Run both post-training evaluations for a set of runs in one pause: the clean-token re-score
(rescore_clean.py) and the conditional-likelihood MCQ (mcq_cond_box.py).

Accepts the flags pause_rescore.sh passes (--repo --val-dir --masks-dir --work --out) plus --runs.
--out is the re-score output; the MCQ output goes next to it with "mcq_" in front of the file name.
Both evaluations run even if the first fails; the exit status is non-zero if either failed.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def commands(args, python=sys.executable):
    out = Path(args.out)
    mcq_out = out.with_name("mcq_" + out.name)
    common = ["--repo", args.repo, "--work", args.work]
    rescore = [python, "-I", str(HERE / "rescore_clean.py"), *common, "--val-dir", args.val_dir,
               "--masks-dir", args.masks_dir, "--out", str(out), "--runs", *args.runs]
    mcq = [python, "-I", str(HERE / "mcq_cond_box.py"), *common, "--out", str(mcq_out), "--runs", *args.runs]
    return rescore, mcq


def main(argv=None, run=subprocess.run):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args(argv)
    status = 0
    for name, cmd in zip(("rescore", "mcq"), commands(args)):
        print(f"evals_box: starting {name}", flush=True)
        rc = run(cmd).returncode
        print(f"evals_box: {name} exit status {rc}", flush=True)
        status = status or rc
    return status


if __name__ == "__main__":
    sys.exit(main())
