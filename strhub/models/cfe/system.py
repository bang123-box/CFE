# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .modules import CACE,FusionModule, Intra_Inter_ConsistencyLoss
from strhub.models.modules import DecoderLayer, Decoder, TokenEmbedding, TPS_SpatialTransformerNetwork


class CFE(CrossEntropySystem):
    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], embed_dim: int, decoder_dim:int,
                 enc_num_heads: int, enc_mlp_ratio: int, depth: List[int],
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int, 
                 mixer_types: List[Union[int, str]], merge_types:str,num_control_points:int,
                dropout: float, window_size:List[List[int]], iiclexist:bool = True, prenorm:bool = False, tps = False, use_pe = True,
        fpn_layers = [0, 1, 2], cc_weights = 0.2, local_type = "r2", **kwargs) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        self.max_label_length = max_label_length
        mixer_types = [mixer_types[0]] * mixer_types[1] + [mixer_types[2]] * mixer_types[3]
        self.transformation = TPS_SpatialTransformerNetwork(
            F=num_control_points, I_size=tuple(img_size), I_r_size=tuple(img_size),
            I_channel_num=3) if tps else nn.Identity()
        
        self.encoder = CACE(img_size, 3, embed_dims=embed_dim, depth=depth, num_heads=enc_num_heads,
                               mixer_types=mixer_types,window_size=window_size, mlp_ratio=enc_mlp_ratio,\
                               merging_types=merge_types, prenorm=prenorm, local_type=local_type, use_pe=use_pe)
        self.fusion = FusionModule(img_size, embed_dim, decoder_dim, fpn_layers)
        decoder_layer = DecoderLayer(decoder_dim, dec_num_heads, decoder_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(decoder_dim))
        if iiclexist:
            self.iicl = Intra_Inter_ConsistencyLoss(len(charset_train), decoder_dim, 0 * decoder_dim, alpha=cc_weights, \
                                            start=self.tokenizer._stoi[charset_train[0]])
            print(f"cc_weights:  {cc_weights}")
        else:
            self.iicl = None
        
        # We predict <bos> and <pad>
        self.head = nn.Linear(decoder_dim, len(self.tokenizer))
        self.text_embed = TokenEmbedding(len(self.tokenizer), decoder_dim)
        
        # +1 for <eos>
        self.pos_embedding = nn.Parameter(torch.Tensor(1, max_label_length + 1, decoder_dim))
        self.dropout = nn.Dropout(p=dropout)
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_embedding, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_embedding'}
        return param_names

    def encode(self, img: torch.Tensor):
        img = self.transformation(img)
        fpn = self.encoder(img)
        x = self.fusion(fpn)
        return x
        
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape

        tgt_query = self.pos_embedding[:, :L] + self.text_embed(tgt)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        img = self.transformation(images)
        fpn = self.encoder(img)
        memory = self.fusion(fpn)
        
        tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
        tgt_in[:, 0] = self.bos_id
        self_attn_mask = self.get_selfmask(num_steps)
        logits = []
        for i in range(num_steps):
            j = i + 1  # next token index
            tgt_out = self.decode(tgt_in[:, :j], memory, tgt_query_mask=self_attn_mask[:j, :j])
            # the next token probability is in the output's ith token position
            p_i = self.head(tgt_out[0][:, -1, :])
            logits.append(p_i.unsqueeze(1))
            if j < num_steps:
                # greedy decode. add the next token index to the target input
                tgt_in[:, j] = p_i.argmax(-1)
                
                if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                    break
        logits = torch.cat(logits, dim=1)
        return logits, tgt_out

    def get_selfmask(self, T: int):
        return torch.triu(torch.full((T, T), float('-inf'), device=self._device), 1)
        
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.encode(images)
        
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        
        self_attn_mask = self.get_selfmask(tgt_in.shape[1])

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
    
        out = self.decode(tgt_in, memory, tgt_query_mask=self_attn_mask)[0]
        logits = self.head(out).flatten(end_dim=1)
        loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
        loss_numel += n   
        loss /= loss_numel
        self.log('loss', loss)
        if self.ccloss:
            total_steps = self.trainer.estimated_stepping_batches * self.trainer.accumulate_grad_batches
             #and self.trainer.current_epoch <= int(self.trainer.max_epochs-2)
            if self.global_step >= 0.75 * total_steps:
                iicl = self.iicl(out, tgt_out, labels)
                self.log('iicl', iicl)
                loss += iicl
        return loss