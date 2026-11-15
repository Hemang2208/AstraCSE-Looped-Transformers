"""Tests for local checkpoint pruning in src/train.py (--keep-local).

Pruning must delete only the requested kind of checkpoint (periodic or best),
keep the newest ones by step number (not by name order of other kinds), delete
the matching .sha256 files, and never touch sigterm or final checkpoints.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.train import BEST_CKPT_RE, PERIODIC_CKPT_RE, prune_local_checkpoints  # noqa: E402


def _touch(d: Path, name: str) -> None:
    (d / name).write_bytes(b"x")
    (d / name).with_suffix(".sha256").write_text("h")


def test_keeps_newest_periodic_and_leaves_other_kinds(tmp_path):
    for name in ["step-0000200.pt", "step-0000400.pt", "step-0000600.pt", "step-0001000.pt",
                 "step-0000500-best.pt", "step-0000600-sigterm.pt", "step-0001000-final.pt"]:
        _touch(tmp_path, name)
    removed = prune_local_checkpoints(tmp_path, 2, PERIODIC_CKPT_RE)
    assert sorted(p.name for p in removed) == ["step-0000200.pt", "step-0000400.pt"]
    left = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert left == ["step-0000500-best.pt", "step-0000600-sigterm.pt", "step-0000600.pt",
                    "step-0001000-final.pt", "step-0001000.pt"]
    assert not (tmp_path / "step-0000200.sha256").exists()
    assert (tmp_path / "step-0000600.sha256").exists()


def test_best_pruning_keeps_only_newest_best(tmp_path):
    for name in ["step-0000500-best.pt", "step-0001000-best.pt", "step-0001500-best.pt",
                 "step-0001500.pt"]:
        _touch(tmp_path, name)
    prune_local_checkpoints(tmp_path, 1, BEST_CKPT_RE)
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["step-0001500-best.pt",
                                                              "step-0001500.pt"]


def test_fewer_than_keep_removes_nothing(tmp_path):
    _touch(tmp_path, "step-0000200.pt")
    assert prune_local_checkpoints(tmp_path, 2, PERIODIC_CKPT_RE) == []
    assert (tmp_path / "step-0000200.pt").exists()


def test_order_is_numeric(tmp_path):
    for name in ["step-0000900.pt", "step-0010000.pt", "step-0002000.pt"]:
        _touch(tmp_path, name)
    prune_local_checkpoints(tmp_path, 1, PERIODIC_CKPT_RE)
    assert [p.name for p in tmp_path.glob("*.pt")] == ["step-0010000.pt"]
