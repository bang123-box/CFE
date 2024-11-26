# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from functools import partial
from strhub.models.utils import init_weights
from timm.models.helpers import named_apply

def ConvBNLayer(inchanns, outchanns, kernel, stride, padding=0, activation="relu"):
    return nn.Sequential(
        *[
            nn.Conv2d(inchanns, outchanns, kernel, stride, padding),
            nn.BatchNorm2d(outchanns),
            nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()
        ]
    )

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class RetNetRelPos(nn.Module):
    # The code is modified based on the paper: https://arxiv.org/abs/2307.08621
    def __init__(self, sigma, head_dim, input_shape, local_k, mixer, local_type="r2"):
        super().__init__()
        self.h, self.w = input_shape
        self.hk, self.wk = local_k
        self.mixer = mixer
        self.sigma = sigma
        self.local_type = local_type
        angle = 1.0 / (10000 ** torch.linspace(0, 1, head_dim // 2))  # dims of each head // 2
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()  # 16 * 2 => 32
        d = torch.log(1 - 2 ** (-5 - torch.arange(sigma, dtype=torch.float)))   # num_heads
        #decay = torch.tensor(sigma, dtype=torch.float)   # sigma
        self.register_buffer("angle", angle)
        self.register_buffer("d", d)
        height, width = self.h, self.w
        index = torch.arange(height * width)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
        if self.mixer == 'Local':
            if self.local_type == "r1":
                self.r(height, width)
            elif self.local_type == "r2":
                self.r2(height, width)
            elif self.local_type == "r3":
                self.r1(height, width)
            else:
                print("the local type must be in r1, r2, r3")  
        else: #"global"
            mask = torch.ones(
                [self.sigma, height * width, height ,width],
                dtype=torch.float)
            mask = mask.flatten(2)
            mask = mask.unsqueeze(0)
            decay = torch.zeros((self.sigma, height * width, height * width)).type_as(self.d).unsqueeze(0)
            self.register_buffer("decay_matrix", mask)
            self.register_buffer("mask", decay)

    def r1(self, height, width):
        mask = torch.zeros(
                [self.sigma, height * width, height ,width],
                dtype=torch.float)
        for h in range(0, height):
            for w in range(0, width):
                i = h - torch.arange(0,  height).type_as(mask).unsqueeze(-1)
                j = w - torch.arange(0, width).type_as(mask).unsqueeze(0)
                i_j = torch.abs(i) + torch.abs(j)
                deacy_hw = torch.exp(self.d[:, None, None] * i_j[None, :, :])
                mask[:, h * width + w, :, :] = deacy_hw
        mask = mask.flatten(2)
        mask = mask.unsqueeze(0)
        decay = torch.zeros((self.sigma, height * width, height * width)).type_as(self.d).unsqueeze(0)
        self.register_buffer("decay_matrix", mask)
        self.register_buffer("mask", decay)
    
    def r2(self, height, width):
        mask = torch.zeros(
                [self.sigma, height * width, height ,width],
                dtype=torch.float)
        for h in range(0, height):
            for w in range(0, width):
                i = h - torch.arange(0,  height).type_as(mask).unsqueeze(-1).repeat(1, width)
                j = w - torch.arange(0, width).type_as(mask).unsqueeze(0).repeat(height, 1)
                i, j = torch.abs(i), torch.abs(j)
                i[i < j] = j[i < j]
                deacy_hw = torch.exp(self.d[:, None, None] * i[None, :, :])
                mask[:, h * width + w, :, :] = deacy_hw
        mask = mask.flatten(2)
        mask = mask.unsqueeze(0)
        decay = torch.zeros((self.sigma, height * width, height * width)).type_as(self.d).unsqueeze(0)
        self.register_buffer("decay_matrix", mask)
        self.register_buffer("mask", decay)

    def r3(self, height, width):
        hk, wk =self.hk, self.wk
        mask = torch.zeros(
                [self.sigma, height * width, height + hk - 1, width + wk - 1],
                dtype=torch.float)
        for h in range(0, height):
            for w in range(0, width):
                mask[h * width + w, h:h + hk, w:w + wk] = 1.0
        mask = mask[:, :, hk // 2:height + hk // 2,
                    wk // 2:width + wk // 2].flatten(2)
        mask = mask.unsqueeze(0)
        decay = torch.masked_fill(torch.zeros((self.sigma, height * width, height * width)).type_as(self.d), mask == -1.0, float("-inf"))
        decay = decay.unsqueeze(0)
        self.register_buffer("decay_matrix", mask)
        self.register_buffer("mask", decay)

class OverlapPatchEmbed(nn.Module):
    """Image to the progressive overlapping Patch Embedding.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        embed_dims (int): The dimensions of embedding. Defaults to 768.
        num_layers (int, optional): Number of Conv_BN_Layer. Defaults to 2 and
            limit to [2, 3].
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """
    def __init__(self,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 num_layers: int = 2):
        super().__init__()
        assert num_layers in [2, 3], \
            'The number of layers must belong to [2, 3]'
        self.net = nn.Sequential()
        for num in range(num_layers, 0, -1):
            if (num == num_layers):
                _input = in_channels
            _output = embed_dims // (2**(num - 1))
            self.net.add_module(
                f'ConvBNLayer{str(num_layers - num)}',
                ConvBNLayer(
                    inchanns=_input,
                    outchanns=_output,
                    kernel=3,
                    stride=2,
                    padding=1,
                    activation="gelu"))
            _input = _output
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            x (Tensor): A Tensor of shape :math:`(N, C, H, W)`.
        Returns:
            Tensor: A tensor of shape math:`(N, HW//16, C)`.
        """
        x = self.net(x).flatten(2).permute(0, 2, 1)
        return x


class ConvMixer(nn.Module):
    """The conv Mixer.
    Args:
        embed_dims (int): Number of character components.
        num_heads (int, optional): Number of heads. Defaults to 8.
        input_shape (Tuple[int, int], optional): The shape of input [H, W].
            Defaults to [8, 32].
        local_k (Tuple[int, int], optional): Window size. Defaults to [3, 3].
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """
    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 input_shape: Tuple[int, int] = [8, 32],
                 local_k: Tuple[int, int] = [3, 3],
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.embed_dims = embed_dims
        self.local_mixer = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=local_k,
            stride=1,
            padding=(local_k[0] // 2, local_k[1] // 2),
            groups=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, HW, C)`.

        Returns:
            torch.Tensor: Tensor: A tensor of shape math:`(N, HW, C)`.
        """
        h, w = self.input_shape
        x = x.permute(0, 2, 1).reshape([-1, self.embed_dims, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class AttnMixer(nn.Module):
    """One of mixer of {'Global', 'Local'}. Defaults to Global Mixer.
    Args:
        embed_dims (int): Number of character components.
        num_heads (int, optional): Number of heads. Defaults to 8.
        mixer (str, optional): The mixer type, choices are 'Global' and
            'Local'. Defaults to 'Global'.
        input_shape (Tuple[int, int], optional): The shape of input [H, W].
            Defaults to [8, 32].
        local_k (Tuple[int, int], optional): Window size. Defaults to [7, 11].
        qkv_bias (bool, optional): Whether a additive bias is required.
            Defaults to False.
        qk_scale (float, optional): A scaling factor. Defaults to None.
        attn_drop (float, optional): Attn dropout probability. Defaults to 0.0.
        proj_drop (float, optional): Proj dropout layer. Defaults to 0.0.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """
    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 mixer: str = 'Global',
                 input_shape: Tuple[int, int] = [8, 32],
                 local_k: Tuple[int, int] = [7, 11],
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 local_type: str = "r2",
                 use_pe: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        assert mixer in {'Global', 'Local'}, \
            "The type of mixer must belong to {'Global', 'Local'}"
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_shape = input_shape
        self.use_pe = use_pe
        self.xpos = RetNetRelPos(num_heads, head_dim, input_shape, local_k, mixer, local_type=local_type)
        if input_shape is not None:
            height, width = input_shape
            self.input_size = height * width
            self.embed_dims = embed_dims
        self.mixer = mixer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.
        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H, W, C)`.
        """
        if self.input_shape is not None:
            input_size, embed_dims = self.input_size, self.embed_dims
        else:
            _, input_size, embed_dims = x.shape
        sin, cos, mask, decay = self.xpos.sin, self.xpos.cos, self.xpos.mask, self.xpos.decay_matrix
        qkv = self.qkv(x).reshape((-1, input_size, 3, self.num_heads,
                                   embed_dims // self.num_heads)).permute(
                                       (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        if self.use_pe:
            qr = theta_shift(q, sin, cos)  #bs, num_heads, input_len, dim
            kr = theta_shift(k, sin, cos)  #bs, num_heads, input_len, dim
        else:
            qr, kr = q, k
        attn = qr.matmul(kr.permute(0, 1, 3, 2))
        
        attn = F.softmax(attn, dim=-1)
        if self.mixer == 'Local':
            attn = attn * decay
        attn = self.attn_drop(attn)

        x = attn.matmul(v).permute(0, 2, 1, 3).reshape(-1, input_size,
                                                       embed_dims)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """The MLP block.
    Args:
        in_features (int): The input features.
        hidden_features (int, optional): The hidden features.
            Defaults to None.
        out_features (int, optional): The output features.
            Defaults to None.
        drop (float, optional): cfg of dropout function. Defaults to 0.0.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """
    def __init__(self,
                 in_features: int,
                 hidden_features: int = None,
                 out_features: int = None,
                 drop: float = 0.,
                 **kwargs):
        super().__init__(**kwargs)
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.
        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H, W, C)`.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixingBlock(nn.Module):
    """The Mixing block.
    Args:
        embed_dims (int): Number of character components.
        num_heads (int): Number of heads
        mixer (str, optional): The mixer type. Defaults to 'Global'.
        window_size (Tuple[int ,int], optional): Local window size.
            Defaults to [7, 11].
        input_shape (Tuple[int, int], optional): The shape of input [H, W].
            Defaults to [8, 32].
        mlp_ratio (float, optional): The ratio of hidden features to input.
            Defaults to 4.0.
        qkv_bias (bool, optional): Whether a additive bias is required.
            Defaults to False.
        qk_scale (float, optional): A scaling factor. Defaults to None.
        drop (float, optional): cfg of Dropout. Defaults to 0..
        attn_drop (float, optional): cfg of Dropout. Defaults to 0.0.
        drop_path (float, optional): The probability of drop path.
            Defaults to 0.0.
        pernorm (bool, optional): Whether to place the MxingBlock before norm.
            Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 mixer: str = 'Global',
                 window_size: Tuple[int, int] = [7, 11],
                 input_shape: Tuple[int, int] = [8, 32],
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path=0.,
                 prenorm: bool = True,
                 local_type: str = "r2",
                 use_pe : bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.norm1 = nn.LayerNorm(embed_dims, eps=1e-6)
        if mixer in {'Global', 'Local'}:
            self.mixer = AttnMixer(
                embed_dims,
                num_heads=num_heads,
                mixer=mixer,
                input_shape=input_shape,
                local_k=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop, local_type=local_type, use_pe=use_pe)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                embed_dims,
                num_heads=num_heads,
                input_shape=input_shape,
                local_k=window_size)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = MLP(
            in_features=embed_dims, hidden_features=mlp_hidden_dim, drop=drop)
        self.prenorm = prenorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H*W, C)`.
        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H*W, C)`.
        """
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MerigingBlock(nn.Module):
    """The last block of any stage, except for the last stage.
    Args:
        in_channels (int): The channels of input.
        out_channels (int): The channels of output.
        types (str, optional): Which downsample operation of ['Pool', 'Conv'].
            Defaults to 'Pool'.
        stride (Union[int, Tuple[int, int]], optional): Stride of the Conv.
            Defaults to [2, 1].
        act (bool, optional): activation function. Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 types: str = 'Pool',
                 stride: Union[int, Tuple[int, int]] = [2, 1],
                 act: bool = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1)
        self.norm = nn.LayerNorm(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.
        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H/2, W, 2C)`.
        """
        if self.types == 'Pool':
            x = (self.avgpool(x) + self.maxpool(x)) * 0.5
            out = self.proj(x.flatten(2).permute(0, 2, 1))
        else:
            x = self.conv(x)
            out = x.flatten(2).permute(0, 2, 1)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class CACE(nn.Module):
    # This code is modified from https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/encoders/svtr_encoder.py
    def __init__(self,
                 img_size: Tuple[int, int] = [32, 128],
                 in_channels: int = 3,
                 embed_dims: Tuple[int, int, int] = [128,256,384],
                 depth: Tuple[int, int, int] = [3, 6, 9],
                 num_heads: Tuple[int, int, int] = [4, 8, 12],
                 mixer_types: Tuple[str] = ['Local'] * 8 + ['Global'] * 10,
                 window_size: Tuple[Tuple[int, int]] = [[7, 11], [7, 11],
                                                        [7, 11]],
                 merging_types: str = 'Conv',
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 last_drop: float = 0.1,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 out_channels: int = 192,
                 num_layers: int = 2,
                 prenorm: bool = False,
                 local_type: str = "r2",
                 use_pe : bool = True, 
                 **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.embed_dims = embed_dims
        self.out_channels = out_channels
        self.prenorm = prenorm
        self.patch_embed = OverlapPatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims[0],
            num_layers=num_layers)
        num_patches = (img_size[1] // (2**num_layers)) * (
            img_size[0] // (2**num_layers))
        self.input_shape = [
            img_size[0] // (2**num_layers), img_size[1] // (2**num_layers)
        ]
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = np.linspace(0, drop_path_rate, sum(depth))

        self.blocks1 = nn.ModuleList([
            MixingBlock(
                embed_dims=embed_dims[0],
                num_heads=num_heads[0],
                mixer=mixer_types[0:depth[0]][i],
                window_size=window_size[0],
                input_shape=self.input_shape,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depth[0]][i],
                prenorm=prenorm, local_type=local_type, use_pe=use_pe) for i in range(depth[0])
        ])
        self.downsample1 = MerigingBlock(
            in_channels=embed_dims[0],
            out_channels=embed_dims[1],
            types=merging_types,
            stride=[2, 1])
        input_shape = [self.input_shape[0] // 2, self.input_shape[1]]
        self.merging_types = merging_types

        self.blocks2 = nn.ModuleList([
            MixingBlock(
                embed_dims=embed_dims[1],
                num_heads=num_heads[1],
                mixer=mixer_types[depth[0]:depth[0] + depth[1]][i],
                window_size=window_size[1],
                input_shape=input_shape,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i],
                prenorm=prenorm, local_type=local_type) for i in range(depth[1])
        ])
        self.downsample2 = MerigingBlock(
            in_channels=embed_dims[1],
            out_channels=embed_dims[2],
            types=merging_types,
            stride=[2, 1])
        input_shape = [self.input_shape[0] // 4, self.input_shape[1]]

        self.blocks3 = nn.ModuleList([
            MixingBlock(
                embed_dims=embed_dims[2],
                num_heads=num_heads[2],
                mixer=mixer_types[depth[0] + depth[1]:][i],
                window_size=window_size[2],
                input_shape=input_shape,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1]:][i],
                prenorm=prenorm, local_type=local_type) for i in range(depth[2])
        ])
        named_apply(partial(init_weights, exclude=[]), self)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function except the last combing operation.
        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, 3, H, W)`.
        Returns:
            torch.Tensor: A List Tensor of shape :math:[`(N, H/4, W/4, C_1)`, `(N, H/8, W/4, C_2)`, `(N, H/16, W/4, C_3)`]`.
        """
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        fpn = []
        for blk in self.blocks1:
            x = blk(x)
        fpn.append(x)
        x = self.downsample1(
            x.permute(0, 2, 1).reshape([
                -1, self.embed_dims[0], self.input_shape[0],
                self.input_shape[1]
            ]))

        for blk in self.blocks2:
            x = blk(x)
        fpn.append(x)
        x = self.downsample2(
            x.permute(0, 2, 1).reshape([
                -1, self.embed_dims[1], self.input_shape[0] // 2,
                self.input_shape[1]
            ]))

        for blk in self.blocks3:
            x = blk(x)
        fpn.append(x)
        return fpn

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        """Forward function.
        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, 3, H, W)`.
        Returns:
            torch.Tensor: A List Tensor of shape :math:[`(N, H/4, W/4, C_1)`, `(N, H/8, W/4, C_2)`, `(N, H/16, W/4, C_3)`].
        """
        fpn = self.forward_features(x)
        return fpn


class FusionModule(nn.Module):
    def __init__(self, img_size, embed_dims, out_dim, fpn_layers, **kwargs):
        super().__init__(**kwargs)
        self.h, self.w = img_size[0] // 4, img_size[1]// 4
        self.fpn_layers = fpn_layers
        self.linear = nn.ModuleList()
        for dim in embed_dims:
            self.linear.append(nn.Linear(dim, out_dim))
    
    def forward(self, fpn):
        fusion = []
        assert len(fpn) == len(self.linear), print("the length of output encoder must \
                                                   equal to the length of embed_dims")
        for f, layer in zip(fpn, self.linear):
            fusion.append(layer(f))
        return torch.cat([fusion[i] for i in self.fpn_layers], dim=1)


class Intra_Inter_ConsistencyLoss(nn.Module):
    """Contrastive Center loss.
    Reference:
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=94, in_dim=256, out_dim=2048, eps=1, alpha=0.1, start=0):
        super().__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.eps = eps
        self.alpha = alpha
        self.out_dim = out_dim
        self.start = start
        
        self.linear = nn.Linear(in_dim, out_dim) if out_dim > 0 else nn.Identity()
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.out_dim)) if out_dim > 0 else nn.Parameter(torch.randn(self.num_classes, self.in_dim))
        '''
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, in_dim)
        ) if out_dim > 0 else nn.Identity()
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.in_dim))
        '''
        nn.init.trunc_normal_(self.centers, mean=0, std=0.02)
 
    def forward(self, features: torch.Tensor, targets: torch.Tensor, labels: List[str]): 
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        features = self.linear(features)
        new_x, new_t = [] , []
        for f, l, t  in zip(features, labels, targets):
            new_x.append(f[:len(l)])
            new_t.append(t[:len(l)] - self.start)
        x = torch.cat(new_x, dim=0)
        labels = torch.cat(new_t, dim=0)
        batch_size = x.size(0)
        mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = mat - 2 * x @ self.centers.t()
 
        classes = torch.arange(self.num_classes, device=labels.device).long().unsqueeze(0)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
 
        dist = (distmat * mask.float()).sum(1)
        distmat = distmat * (~mask).float()
        sum_dist = distmat.sum(1) + self.eps
        
        ctc = dist / sum_dist
        loss = self.alpha * ctc.sum() / 2
        return loss