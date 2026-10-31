import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig

from ..utils.mask import make_pad_mask


class CausalMaskedDiffWithDiT(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 pre_lookahead_layer: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio

    @torch.inference_mode()
    def inference(self, token, token_len, prompt_token, prompt_token_len,
                  prompt_feat, prompt_feat_len, embedding,
                  streaming=False, finalize=True,
                  n_timesteps=10, cfg=True,
                  prompt_token_emb=None, embedding_proj=None):
        assert token.shape[0] == 1
        if embedding_proj is None:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
        else:
            embedding = embedding_proj

        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        if prompt_token_emb is None:
            token_emb = self.input_embedding(torch.clamp(token, min=0))
        else:
            gen_emb = self.input_embedding(torch.clamp(token[:, prompt_token.shape[1]:], min=0))
            token_emb = torch.cat([prompt_token_emb, gen_emb], dim=1)
        token = token_emb * mask

        h = self.pre_lookahead_layer(token)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            streaming=False,
            cfg=cfg,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None
