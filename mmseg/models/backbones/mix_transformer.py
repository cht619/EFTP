import copy
import math
import warnings
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import _load_checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers import trunc_normal_
# from ..ops.modules import MSDeformAttn
from .adapter_modules import SpatialPriorModule, InteractionBlockPrompt

from mmseg.models.builder import MODELS
from .mit_adapter import GaussianFilter, SRMFilter, OverlapPatchEmbed, PromptGenerator, PromptGenerator_low_high


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,
                                                           3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W


@MODELS.register_module()
class MixVisionTransformer_DAFormer(BaseModule):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 style=None,
                 pretrained=None,
                 init_cfg=None,
                 freeze_patch_embed=False,
                 **kwargs):
        super().__init__(init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str) or pretrained is None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
        else:
            raise TypeError('pretrained must be a str or None')

        self.num_classes = num_classes
        self.depths = depths
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
        if freeze_patch_embed:
            self.freeze_patch_emb()

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) \
        #     if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        logger = None
        if self.pretrained is None:
            # logger.info('Init mit from scratch.')
            for m in self.modules():
                self._init_weights(m)
        elif isinstance(self.pretrained, str):
            # logger.info('Load mit checkpoint.')
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)

    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


@MODELS.register_module()
class mit_b5_daformer(MixVisionTransformer_DAFormer):

    def __init__(self, **kwargs):
        super(mit_b5_daformer, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


class DAFormer_EVP(MixVisionTransformer_DAFormer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handcrafted_tune = True
        self.tuning_stage = '1234'
        # prompt adapter
        self.prompt_generator = PromptGenerator(
            scale_factor=4, prompt_type='highpass', embed_dims=[64, 128, 320, 512], tuning_stage='1234',
            depths=[3, 6, 40, 3], input_type='fft', freq_nums=0.25, handcrafted_tune=True, embedding_tune=True,
            adaptor='adaptor', img_size=512)

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        if self.handcrafted_tune:
            handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(x)
        else:
            handcrafted1, handcrafted2, handcrafted3, handcrafted4 = None, None, None, None

        # stage 1
        x, H, W = self.patch_embed1(x)
        if '1' in self.tuning_stage:
            prompt1 = self.prompt_generator.init_prompt(x, handcrafted1, 1)
        for i, blk in enumerate(self.block1):
            if '1' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt1, 1, i)
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        if '2' in self.tuning_stage:
            prompt2 = self.prompt_generator.init_prompt(x, handcrafted2, 2)
        for i, blk in enumerate(self.block2):
            if '2' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt2, 2, i)
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        if '3' in self.tuning_stage:
            prompt3 = self.prompt_generator.init_prompt(x, handcrafted3, 3)
        for i, blk in enumerate(self.block3):
            if '3' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x,prompt3, 3, i)
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        if '4' in self.tuning_stage:
            prompt4 = self.prompt_generator.init_prompt(x, handcrafted4, 4)
        for i, blk in enumerate(self.block4):
            if '4' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt4, 4, i)
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x)

        return outs


@MODELS.register_module()
class mit_b5_daformer_EVP(DAFormer_EVP):

    def __init__(self, **kwargs):
        for k in ['patch_size', 'embed_dims', 'num_heads', 'mlp_ratios', 'qkv_bias', 'norm_layer', 'depths', 'sr_ratios']:
            if k in kwargs.keys():
                kwargs.pop(k)
        super(mit_b5_daformer_EVP, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@MODELS.register_module()
class mit_b5_daformer_EVP_low_high(DAFormer_EVP):

    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)

        self.prompt_generator = PromptGenerator_low_high(
            scale_factor=4, prompt_type='highpass', embed_dims=[64, 128, 320, 512], tuning_stage='1234',
            depths=[3, 6, 40, 3], input_type='fft', freq_nums=0.25, handcrafted_tune=True, embedding_tune=True,
            adaptor='adaptor', img_size=512)

    def forward(self, x_ori):
        x = copy.deepcopy(x_ori)
        B = x.shape[0]
        # high component
        outs_high = []
        handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(x)

        x, H, W = self.patch_embed1(x)
        prompt1 = self.prompt_generator.init_prompt(x, handcrafted1, 1)
        for i, blk in enumerate(self.block1):
            x = self.prompt_generator.get_prompt(x, prompt1, 1, i)
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs_high.append(x)
        x, H, W = self.patch_embed2(x)
        prompt2 = self.prompt_generator.init_prompt(x, handcrafted2, 2)
        for i, blk in enumerate(self.block2):
            x = self.prompt_generator.get_prompt(x, prompt2, 2, i)
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs_high.append(x)
        x, H, W = self.patch_embed3(x)
        prompt3 = self.prompt_generator.init_prompt(x, handcrafted3, 3)
        for i, blk in enumerate(self.block3):
            x = self.prompt_generator.get_prompt(x, prompt3, 3, i)
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs_high.append(x)
        x, H, W = self.patch_embed4(x)
        prompt4 = self.prompt_generator.init_prompt(x, handcrafted4, 4)
        for i, blk in enumerate(self.block4):
            x = self.prompt_generator.get_prompt(x, prompt4, 4, i)
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs_high.append(x)

        # low component
        x = copy.deepcopy(x_ori)
        outs_low = []
        handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted_low(x)
        x, H, W = self.patch_embed1(x)
        prompt1 = self.prompt_generator.init_prompt(x, handcrafted1, 1)
        for i, blk in enumerate(self.block1):
            x = self.prompt_generator.get_prompt(x, prompt1, 1, i)
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs_low.append(x)
        x, H, W = self.patch_embed2(x)
        prompt2 = self.prompt_generator.init_prompt(x, handcrafted2, 2)
        for i, blk in enumerate(self.block2):
            x = self.prompt_generator.get_prompt(x, prompt2, 2, i)
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs_low.append(x)
        x, H, W = self.patch_embed3(x)
        prompt3 = self.prompt_generator.init_prompt(x, handcrafted3, 3)
        for i, blk in enumerate(self.block3):
            x = self.prompt_generator.get_prompt(x, prompt3, 3, i)
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs_low.append(x)
        x, H, W = self.patch_embed4(x)
        prompt4 = self.prompt_generator.init_prompt(x, handcrafted4, 4)
        for i, blk in enumerate(self.block4):
            x = self.prompt_generator.get_prompt(x, prompt4, 4, i)
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs_low.append(x)

        outs = [_ for _ in range(4)]
        for i in range(len(outs_high)):
            outs[i] = 0.0 * outs_high[i] + 1.0 * outs_low[i]

        # return outs_low, outs_high
        return outs


@MODELS.register_module()
class daformer_vpt(MixVisionTransformer_DAFormer):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)
        self.num_tokens = 10
        self.prompt_dropout = nn.Dropout(0.1)
        self.prompt_proj = nn.Identity()
        self.embed_dims = [64, 128, 320, 512]
        # vpt config
        self.token_depth = 1
        self.num_tokens_1 = 128
        self.num_tokens_2 = 64
        self.num_tokens_3 = 32
        self.num_tokens_4 = 16
        self.depths = [3, 6, 40, 3]

        self.prompt_dropout = nn.Dropout(0)
        self.prompt_proj = nn.Identity()

        def prompt_init(prompt_embeddings, dim):
            val = math.sqrt(
                6. / float(3 * reduce(mul, [16, 16], 1) + dim))  # noqa
            nn.init.uniform_(prompt_embeddings.data, -val, val)

        self.prompt_embeddings_1 = nn.Parameter(torch.zeros(self.depths[0], self.num_tokens_1, self.embed_dims[0]))
        prompt_init(self.prompt_embeddings_1, self.embed_dims[0])
        self.prompt_embeddings_2 = nn.Parameter(torch.zeros(self.depths[1], self.num_tokens_2, self.embed_dims[1]))
        prompt_init(self.prompt_embeddings_2, self.embed_dims[1])
        self.prompt_embeddings_3 = nn.Parameter(torch.zeros(self.depths[2], self.num_tokens_3, self.embed_dims[2]))
        prompt_init(self.prompt_embeddings_3, self.embed_dims[2])
        self.prompt_embeddings_4 = nn.Parameter(torch.zeros(self.depths[3], self.num_tokens_4, self.embed_dims[3]))
        prompt_init(self.prompt_embeddings_4, self.embed_dims[3])

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)

        for i, blk in enumerate(self.block1):
            x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_1[i])).expand(B, -1, -1), x],
                          dim=1)
            x = blk(x, H + self.token_depth, W)
            x = x[:, self.num_tokens_1:, :]
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_2[i])).expand(B, -1, -1), x],
                          dim=1)
            x = blk(x, H + self.token_depth, W)
            x = x[:, self.num_tokens_2:, :]
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_3[i])).expand(B, -1, -1), x],
                          dim=1)
            x = blk(x, H + self.token_depth, W)
            x = x[:, self.num_tokens_3:, :]
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)

        for i, blk in enumerate(self.block4):
            x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_4[i])).expand(B, -1, -1), x],
                          dim=1)
            x = blk(x, H + self.token_depth, W)
            x = x[:, self.num_tokens_4:, :]
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs
    

@MODELS.register_module()
class mit_b5_daformer_prompt(mit_b5_daformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # prompt generator
        conv_inplane = 64
        with_cp = False
        embed_dims = [64, 128, 320, 512]  # self.embed_dims * num_heads
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dims[0], with_cp=with_cp)
        self.next_dim = [128, 320, 512, None]
        deform_num_heads = 16
        n_points = 4
        init_values = 1e-6
        drop_path_rate = 0.1
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        with_cffn = True
        cffn_ratio = 0.25
        depths = [3, 6, 40, 3]
        deform_ratio = 0.
        use_extra_extractor = False

        self.interactions = nn.Sequential(*[
            InteractionBlockPrompt(dim=embed_dims[i], num_heads=deform_num_heads, n_points=n_points,
                                   init_values=init_values, drop_path=drop_path_rate,
                                   norm_layer=norm_layer, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                                   extra_extractor=use_extra_extractor,
                                   with_cp=with_cp, next_dim=self.next_dim[i])
            for i in range(len(depths))
        ])
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)

        self.patch_embed = [self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4]
        self.block = [self.block1, self.block2, self.block3, self.block4]
        self.norm = [self.norm1, self.norm2, self.norm3, self.norm4]

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _add_level_embed(self, c1, c2, c3):
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        return c1, c2, c3

    def forward(self, x):
        out_prompt = []
        B = x.shape[0]
        outs = []
        # SPM forward
        c1, c2, c3 = self.spm(x)  # b*65536*128 b*16384*128 b*4096*128 b*1024*128
        # Interaction with spm prompt
        for i, (patch_embed, block, norm, layer) in enumerate(
                zip(self.patch_embed, self.block, self.norm, self.interactions)):
            # stage i
            x, H, W = patch_embed(x)

            x_injector, c1, c2, c3, x = layer(x, c1, c2, c3, block, (H, W), backbone='mit')

            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
            out_prompt.append(x_injector)
        outputs = dict()
        outputs["outs"] = outs
        outputs["prompts"] = out_prompt
        return outputs