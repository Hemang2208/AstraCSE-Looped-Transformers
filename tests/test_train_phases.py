"""Tests for multi-phase training in src/train.py (context extension).

A tiny model is trained with the real train() on CPU: a first phase at context
128, then a second phase at context 256 started with --init-from. Checks that the
data loader follows the config's max_seq_len (it used to stay at 1024 whatever the
config said), that --init-from copies the weights exactly, and that --resume takes
priority over --init-from.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import verify_param_count as vpc  # noqa: E402
from src.data import make_dataloader  # noqa: E402
from src.model import AstraCSE, ModelConfig  # noqa: E402
from src.train import init_weights, train  # noqa: E402


def _config(tmp: Path, name: str, seq: int, lr: float = 1e-3) -> Path:
    full = yaml.safe_load((REPO_ROOT / "configs" / "model.yaml").read_text())
    full["model"].update(d_model=64, n_layer=2, n_head=2, d_head=32, d_ff=128, max_seq_len=seq)
    full["model"]["expected_param_count"] = vpc.count_params(full["model"])["total"]
    full["lr_schedule"]["peak_lr"] = lr
    full["training"]["target_effective_batch_tokens"] = 2 * seq * 2   # grad_accum 2 at micro_bs 2
    p = tmp / f"{name}.yaml"
    p.write_text(yaml.safe_dump(full))
    return p


def _shards(tmp: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    for split, n in (("train", 60_000), ("val", 8_000)):
        d = tmp / split
        d.mkdir()
        rng.integers(0, 15000, size=n, dtype=np.uint16).tofile(d / "shard_0000.bin")
    return tmp / "train", tmp / "val"


def _args(cfg, train_dir, val_dir, ckpt_dir, steps=2, **kw):
    a = dict(config=str(cfg), train_shards=str(train_dir), val_shards=str(val_dir),
             tok_dir=str(REPO_ROOT / "tokenizer"), checkpoint_dir=str(ckpt_dir), resume=None,
             micro_batch_size=2, total_steps=steps, log_every=1, eval_every=1000,
             ckpt_every=1000, s3_prefix=None, keep_local=0, early_stop_patience=2,
             init_from=None, compile=False, seed=None)
    a.update(kw)
    return argparse.Namespace(**a)


def test_loader_follows_the_requested_sequence_length(tmp_path):
    train_dir, _ = _shards(tmp_path)
    loader, _ = make_dataloader(train_dir, 2, seq_len=256, shuffle=False)
    inp, tgt = next(iter(loader))
    assert inp.shape == (2, 256) and tgt.shape == (2, 256)


def test_init_weights_copies_every_tensor(tmp_path):
    short = AstraCSE(ModelConfig.from_yaml(_config(tmp_path, "c128", 128)))
    torch.save({"model_state": short.state_dict(), "global_step": 7}, tmp_path / "src.pt")
    long = AstraCSE(ModelConfig.from_yaml(_config(tmp_path, "c256", 256)))
    init_weights(str(tmp_path / "src.pt"), long)
    for k, v in short.state_dict().items():
        assert torch.equal(v, long.state_dict()[k]), k


class _FakeS3:
    """list_objects_v2 paginator + download_file over a dict of key -> local file."""
    def __init__(self, objects: dict):
        self.objects, self.downloaded = objects, []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        objs = self.objects

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in objs if k.startswith(Prefix)]}
        return P()

    def download_file(self, bucket, key, dest):
        self.downloaded.append(key)
        Path(dest).write_bytes(Path(self.objects[key]).read_bytes())


def test_init_from_s3_prefix_picks_the_newest_final_checkpoint(tmp_path, monkeypatch):
    import boto3
    short = AstraCSE(ModelConfig.from_yaml(_config(tmp_path, "c128", 128)))
    good = tmp_path / "good.pt"
    torch.save({"model_state": short.state_dict(), "global_step": 900}, good)
    other = AstraCSE(ModelConfig.from_yaml(_config(tmp_path, "c128b", 128)))
    stale = tmp_path / "stale.pt"
    torch.save({"model_state": other.state_dict(), "global_step": 80}, stale)
    fake = _FakeS3({
        "checkpoints/run/step-0000080-final.pt": str(stale),
        "checkpoints/run/step-0000900-final.pt": str(good),
        "checkpoints/run/step-0001000.pt": str(stale),          # periodic, not final
        "checkpoints/run_other/step-0009999-final.pt": str(stale),  # another prefix
        "checkpoints/run/sub/step-0005000-final.pt": str(stale),    # not directly under
    })
    monkeypatch.setattr(boto3, "client", lambda name, **kw: fake)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "tmp"))
    target = AstraCSE(ModelConfig.from_yaml(_config(tmp_path, "c256", 256)))
    path = init_weights("s3://bucket/checkpoints/run/", target)
    assert fake.downloaded == ["checkpoints/run/step-0000900-final.pt"]
    assert not (tmp_path / "ckpt").exists()                 # never inside the job's dir
    for k, v in short.state_dict().items():
        assert torch.equal(v, target.state_dict()[k]), k
    assert path.exists()


def test_two_phase_training_end_to_end(tmp_path):
    train_dir, val_dir = _shards(tmp_path)
    c128, c256 = _config(tmp_path, "c128", 128), _config(tmp_path, "c256", 256, lr=1e-4)
    train(_args(c128, train_dir, val_dir, tmp_path / "phase1"))
    final1 = sorted((tmp_path / "phase1").glob("*-final.pt"))
    assert final1, "phase 1 saved no final checkpoint"

    train(_args(c256, train_dir, val_dir, tmp_path / "phase2", init_from=str(final1[-1])))
    final2 = sorted((tmp_path / "phase2").glob("*-final.pt"))
    assert final2
    p2 = torch.load(final2[-1], map_location="cpu", weights_only=False)
    assert p2["global_step"] == 2                              # a fresh schedule, not phase 1's
    # Phase 2 trained from phase 1's weights: they moved, but stayed close.
    p1 = torch.load(final1[-1], map_location="cpu", weights_only=False)
    w1, w2 = p1["model_state"]["embed.weight"], p2["model_state"]["embed.weight"]
    assert not torch.equal(w1, w2)
    assert (w1 - w2).abs().max() < 0.05


def test_seed_is_logged_and_reproduces_the_init(tmp_path):
    train_dir, val_dir = _shards(tmp_path)
    c128 = _config(tmp_path, "c128", 128)
    for run in ("s1", "s2"):
        train(_args(c128, train_dir, val_dir, tmp_path / run, steps=1, seed=1234))
    log1 = (tmp_path / "s1" / "train.log").read_text()
    assert "seed=1234" in log1 and "Training complete" in log1
    w = [torch.load(sorted((tmp_path / r).glob("*-final.pt"))[0], map_location="cpu",
                    weights_only=False)["model_state"]["embed.weight"] for r in ("s1", "s2")]
    assert torch.equal(w[0], w[1])                      # same seed, same data -> same run


def test_resume_takes_priority_over_init_from(tmp_path):
    train_dir, val_dir = _shards(tmp_path)
    c128 = _config(tmp_path, "c128", 128)
    train(_args(c128, train_dir, val_dir, tmp_path / "a", steps=2, ckpt_every=1))
    ckpt = sorted((tmp_path / "a").glob("step-0000001.pt"))[0]
    # A bogus init source would raise if it were used.
    train(_args(c128, train_dir, val_dir, tmp_path / "b", steps=2, resume=str(ckpt),
                init_from=str(tmp_path / "does-not-exist.pt")))
    assert sorted((tmp_path / "b").glob("*-final.pt"))
