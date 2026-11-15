"""AstraCSE-100M model — Architecture.md §2–9, §31.

Decoder-only transformer with:
  RoPE, RMSNorm (no bias), SwiGLU, causal MHA, weight tying.
All hyperparameters read from configs/model.yaml via the ModelConfig dataclass.

Optional loop-transformer mode (Architecture.md §31): set n_unique_layers < n_layer
to cycle through a smaller set of unique blocks multiple times, trading parameter
count for effective depth at matched compute (Universal-Transformer-style).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 15000
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    d_ff: int = 2048
    max_seq_len: int = 1024
    norm_eps: float = 1e-5
    weight_tying: bool = True
    expected_param_count: int = 96_473_856
    # None = every one of the n_layer depth-slots gets its own weights (baseline).
    # Set < n_layer to loop a smaller set of unique blocks (Architecture.md §31).
    n_unique_layers: Optional[int] = None

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    @property
    def n_unique(self) -> int:
        """Number of distinct weight-sharing groups actually instantiated."""
        return self.n_unique_layers if self.n_unique_layers is not None else self.n_layer

    def __post_init__(self) -> None:
        if self.n_layer % self.n_unique != 0:
            raise ValueError(
                f"n_layer ({self.n_layer}) must be an exact multiple of "
                f"n_unique_layers ({self.n_unique}) so looping divides evenly "
                f"(Architecture.md §31)"
            )

    @classmethod
    def from_yaml(cls, path: Path) -> ModelConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
        return cls(
            vocab_size=raw["vocab_size"],
            d_model=raw["d_model"],
            n_layer=raw["n_layer"],
            n_head=raw["n_head"],
            d_ff=raw["d_ff"],
            max_seq_len=raw["max_seq_len"],
            norm_eps=raw.get("norm_eps", 1e-5),
            weight_tying=raw.get("weight_tying", True),
            # .get() with default so configs that predate this field still load
            expected_param_count=raw.get("expected_param_count", 96_473_856),
            n_unique_layers=raw.get("n_unique_layers", None),
        )


# ---------------------------------------------------------------------------
# RMSNorm — §9
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


# ---------------------------------------------------------------------------
# RoPE — §5
# ---------------------------------------------------------------------------

def _build_rope_cache(d_head: int, max_seq_len: int,
                      device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin tables of shape [max_seq_len, d_head]."""
    theta = 1.0 / (10000 ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, theta)          # [max_seq_len, d_head/2]
    emb = torch.cat([freqs, freqs], dim=-1)        # [max_seq_len, d_head]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, n_head, T, d_head];  cos/sin: [T, d_head]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, d_head]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention — §7
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):  # noqa: ARG002
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model
        self.max_seq_len = cfg.max_seq_len

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # GPT-2-style residual-stream variance scaling (Architecture.md §3.3).
        # Mark with attribute so _init_weights does NOT overwrite this.
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self.out_proj._scaled_init_done = True  # type: ignore[attr-defined]

        # Lazily built full-sequence RoPE cache; rebuilt if device changes.
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None

    def _get_rope(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rope_cos is None or self._rope_cos.device != device:
            self._rope_cos, self._rope_sin = _build_rope_cache(
                self.d_head, self.max_seq_len, device)
        return self._rope_cos, self._rope_sin

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[tuple] = None) -> tuple[torch.Tensor, tuple]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # [B, nh, T, dh]
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Positional encoding: each token is encoded at its ABSOLUTE position.
        # With KV cache, new tokens start at position `past_len`, not 0.
        past_len = kv_cache[0].shape[2] if kv_cache is not None else 0
        all_cos, all_sin = self._get_rope(x.device)
        cos = all_cos[past_len: past_len + T]
        sin = all_sin[past_len: past_len + T]
        q, k = apply_rope(q, k, cos, sin)

        # Append new K/V to the cache (§25)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # §7.4 — fused SDPA; is_causal only for the prefill (no cache) pass
        is_causal = (kv_cache is None)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out), new_cache


# ---------------------------------------------------------------------------
# SwiGLU MLP — §8
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):  # noqa: ARG002
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff,    cfg.d_model, bias=False)

        nn.init.normal_(self.down.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self.down._scaled_init_done = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Transformer Block — §6
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn  = CausalSelfAttention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp   = SwiGLUMLP(cfg, layer_idx)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[tuple] = None) -> tuple[torch.Tensor, tuple]:
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# Full Model — §1–3
# ---------------------------------------------------------------------------

class AstraCSE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # §31: only n_unique distinct blocks are ever instantiated. When
        # n_unique < n_layer, forward() cycles through them, reusing the same
        # weight objects (and therefore the same nn.Parameter tensors) at
        # multiple depth-slots. PyTorch's parameter/gradient machinery
        # de-duplicates by tensor identity automatically — this is the same
        # mechanism already relied on for embed/lm_head weight tying — so a
        # shared block's gradient correctly accumulates the sum of its
        # contributions from every depth-slot it is used at.
        k = cfg.n_unique
        self.is_looped = k < cfg.n_layer
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(k)])

        # Universal-Transformer-style per-depth-slot conditioning: without it,
        # a shared block has no way to tell which of its T loop iterations is
        # currently running. Only allocated when actually looping, so the
        # non-looped baseline is byte-for-byte unchanged.
        self.depth_embed = (
            nn.Parameter(torch.zeros(cfg.n_layer, cfg.d_model)) if self.is_looped else None
        )

        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)

        # LM head — weight tied to embed (§3.2)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Skip layers that already received scaled init in their constructor
                if getattr(module, "_scaled_init_done", False):
                    continue
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        if self.depth_embed is not None:
            nn.init.normal_(self.depth_embed, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                kv_caches: Optional[list] = None,
                return_kv_caches: bool = False,
                ) -> tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            input_ids:        [B, T] token IDs
            kv_caches:        list of (k, v) per depth-slot for incremental decoding
            return_kv_caches: return updated per-depth-slot KV caches when True
        Returns:
            logits [B, T, V], new_kv_caches (or None)
        """
        _, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, (
            f"Input length {T} exceeds trained max_seq_len "
            f"{self.cfg.max_seq_len} (§5.2)"
        )

        x = self.embed(input_ids)

        n_unique = len(self.blocks)
        new_caches: list = []
        for i in range(self.cfg.n_layer):
            block = self.blocks[i % n_unique]
            if self.depth_embed is not None:
                x = x + self.depth_embed[i]
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cache)
            new_caches.append(new_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, (new_caches if return_kv_caches else None)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def verify_param_count(self) -> None:
        actual = self.count_params()
        expected = self.cfg.expected_param_count
        assert actual == expected, (
            f"Parameter count mismatch: got {actual:,}, "
            f"expected {expected:,} (Architecture.md §3)"
        )


def load_model(config_path: Path) -> AstraCSE:
    cfg = ModelConfig.from_yaml(config_path)
    model = AstraCSE(cfg)
    model.verify_param_count()
    return model
