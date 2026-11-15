"""Tests for model weight initialisation — Architecture.md §3.3.

Verifies:
1. out_proj and down projections have smaller std (scaled init).
2. Other linear layers have std ≈ 0.02.
3. RMSNorm scale params initialised to 1.0.
4. _init_weights does NOT overwrite the scaled projections.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import AstraCSE, ModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return ModelConfig.from_yaml(REPO_ROOT / "configs" / "model.yaml")


def _empirical_std(t: torch.Tensor) -> float:
    return t.float().std().item()


def test_rmsnorm_scale_init_ones(cfg):
    model = AstraCSE(cfg)
    for name, module in model.named_modules():
        if hasattr(module, "scale") and isinstance(module.scale, torch.nn.Parameter):
            assert torch.allclose(module.scale.data, torch.ones_like(module.scale.data)), (
                f"RMSNorm scale not 1.0 at {name}"
            )


def test_scaled_projections_have_smaller_std(cfg):
    model = AstraCSE(cfg)
    expected_scaled_std = 0.02 / math.sqrt(2 * cfg.n_layer)

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        is_scaled = getattr(module, "_scaled_init_done", False)
        std = _empirical_std(module.weight)
        if is_scaled:
            # Should be ~expected_scaled_std, definitely smaller than 0.02
            assert std < 0.015, (
                f"Scaled projection {name} std={std:.4f} is not smaller than 0.015 "
                f"(expected ~{expected_scaled_std:.4f})"
            )
        else:
            # Normal init: std should be reasonably close to 0.02
            # (exact value varies by layer size; just check it's in a sane range)
            if module.weight.numel() > 100:
                assert 0.005 < std < 0.05, (
                    f"Normal-init layer {name} std={std:.4f} far from 0.02"
                )


def test_no_bias_anywhere(cfg):
    model = AstraCSE(cfg)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None, f"Bias found on Linear layer: {name}"


def test_embed_and_lm_head_share_storage(cfg):
    model = AstraCSE(cfg)
    assert model.embed.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_scaled_init_not_overwritten(cfg):
    """_init_weights must skip modules flagged with _scaled_init_done."""
    model = AstraCSE(cfg)
    expected_scaled_std = 0.02 / math.sqrt(2 * cfg.n_layer)

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not getattr(module, "_scaled_init_done", False):
            continue
        std = _empirical_std(module.weight)
        # If _init_weights overwrote it, std would jump back to ~0.02
        assert std < 0.015, (
            f"Scaled projection {name} std={std:.4f} looks overwritten by _init_weights "
            f"(expected ~{expected_scaled_std:.4f}). _scaled_init_done flag not respected."
        )
