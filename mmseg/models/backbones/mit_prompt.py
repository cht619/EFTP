import torch
from torch import nn
from torch.nn.init import normal_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
from .adapter_modules import SpatialPriorModule, InteractionBlockPrompt
from mmseg.registry import MODELS
from .mit import MixVisionTransformer
# from ..ops.modules import MSDeformAttn
from mmengine.runner import load_checkpoint
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from .mix_transformer import MixVisionTransformer_DAFormer


@MODELS.register_module()
class mit_b5_prompt_base(MixVisionTransformer_DAFormer):

    def __init__(self,
                 in_channels=3,
                 freeze_backbone=True,
                 deform_num_heads=16,
                 n_points=4,
                 init_values=1e-6,
                 drop_path_rate=0.1,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=0.5,
                 use_extra_extractor=False,
                 with_cp=False,
                 conv_inplane=64,
                 patch_size=4,
                 embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 6, 40, 3],
                 sr_ratios=[8, 4, 2, 1],
                 **kwargs):
        super().__init__(patch_size=patch_size, embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios,
                         qkv_bias=qkv_bias, norm_layer=norm_layer, depths=depths, sr_ratios=sr_ratios, **kwargs)

        # prompt generator
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dims[0], with_cp=with_cp)

        self.next_dim = [128, 320, 512, None]

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

        if freeze_backbone:
            for name, params in self.named_parameters():
                if 'spm' in name or 'interactions' in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False

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

    def forward_features(self, x):
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

        return outs


@MODELS.register_module()
class mit_b5_prompt(mit_b5_prompt_base):

    def __init__(self,
                 freeze_backbone=True,
                 deform_num_heads=16,
                 n_points=4,
                 init_values=1e-6,
                 drop_path_rate=0.1,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=0.5,
                 use_extra_extractor=False,
                 with_cp=False,
                 conv_inplane=64,
                 patch_size=4,
                 embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 6, 40, 3],
                 sr_ratios=[8, 4, 2, 1],
                 **kwargs):
        super().__init__(
                 freeze_backbone=freeze_backbone,
                 deform_num_heads=deform_num_heads,
                 n_points=n_points,
                 init_values=init_values,
                 drop_path_rate=drop_path_rate,
                 with_cffn=with_cffn,
                 cffn_ratio=cffn_ratio,
                 deform_ratio=deform_ratio,
                 use_extra_extractor=use_extra_extractor,
                 with_cp=with_cp,
                 conv_inplane=conv_inplane,
                 patch_size=patch_size,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 qkv_bias=qkv_bias,
                 norm_layer=norm_layer,
                 depths=depths,
                 sr_ratios=sr_ratios,
                 **kwargs)

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dims[0]))
        normal_(self.level_embed)

        if freeze_backbone:
            for name, params in self.named_parameters():
                if 'spm' in name or 'interactions' in name or 'level_embed' in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False

    def _add_level_embed(self, c1, c2, c3):
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        return c1, c2, c3

    def forward_features(self, x):
        out_prompt = []
        B = x.shape[0]
        outs = []
        # SPM forward
        c1, c2, c3 = self.spm(x)
        # 不能处理shape=[540, 960]，得想办法处理一下
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