
"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from x_transformers.x_transformers import RotaryEmbedding
from ...utils.mask import add_optional_chunk_mask
from .modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    CausalConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        batch, text_len = text.shape[0], text.shape[1]
        text = text + 1
        text = text[:, :seq_len]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:
            text = torch.zeros_like(text)

        text = self.text_embed(text)

        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed
            text = self.text_blocks(text)

        return text


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, spk_dim=None):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(
            self,
            x: float["b n d"],
            cond: float["b n d"],
            text_embed: float["b n d"],
            spks: float["b d"],
    ):
        to_cat = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks = repeat(spks, "b c -> b t c", t=x.shape[1])
            to_cat.append(spks)

        x = self.proj(torch.cat(to_cat, dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        mu_dim=None,
        long_skip_connection=False,
        spk_dim=None,
        out_channels=None,
        static_chunk_size=50,
        num_decoding_left_chunks=2
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spks = spks.unsqueeze(dim=1)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        t = self.time_embed(t)
        x = self.input_embed(x, cond, mu, spks.squeeze(1))

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        if streaming is True:
            attn_mask = add_optional_chunk_mask(x, mask.bool(), False, False, 0, self.static_chunk_size, -1).unsqueeze(dim=1).bool()
        else:
            # Non-streaming, batch=1 (or CFG-doubled 2) with no padding: the
            # full (B, T, T) attention mask is all-True. Passing None lets
            # SDPA dispatch to FlashAttention instead of the mem-efficient
            # path triggered by a non-None mask.
            attn_mask = None

        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output
