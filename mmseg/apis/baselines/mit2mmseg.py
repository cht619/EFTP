import argparse
import os.path as osp
from collections import OrderedDict
import os
import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_mit(ckpt):
    new_ckpt = OrderedDict()
    # Process the concat between q linear weights and kv linear weights
    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('model.decode_head.'):  # 针对daformer
            continue
        # patch embedding conversion
        elif k.startswith('patch_embed') or k.startswith('model.backbone.patch_embed'):
            if k.startswith('model.backbone.'):
                k = k.replace('model.backbone.', '') # 针对daformer 把model.backbone置空
            stage_i = int(k.split('.')[0].replace('patch_embed', ''))
            new_k = k.replace(f'patch_embed{stage_i}', f'layers.{stage_i-1}.0')
            new_v = v
            if 'proj.' in new_k:
                new_k = new_k.replace('proj.', 'projection.')
        # transformer encoder layer conversion
        elif k.startswith('block') or k.startswith('model.backbone.block'):
            if k.startswith('model.backbone.'):
                k = k.replace('model.backbone.', '')
                flag = True
            else:
                flag = False
            stage_i = int(k.split('.')[0].replace('block', ''))
            new_k = k.replace(f'block{stage_i}', f'layers.{stage_i-1}.1')
            new_v = v
            if 'attn.q.' in new_k:
                sub_item_k = k.replace('q.', 'kv.')
                new_k = new_k.replace('q.', 'attn.in_proj_')
                if flag:
                    # 这里要注意把原来的加回来不然加不上
                    new_v = torch.cat([v, ckpt['model.backbone.{}'.format(sub_item_k)]], dim=0)
                else:
                    new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
            elif 'attn.kv.' in new_k:
                continue
            elif 'attn.proj.' in new_k:
                new_k = new_k.replace('proj.', 'attn.out_proj.')
            elif 'attn.sr.' in new_k:
                new_k = new_k.replace('sr.', 'sr.')
            elif 'mlp.' in new_k:
                string = f'{new_k}-'
                new_k = new_k.replace('mlp.', 'ffn.layers.')
                if 'fc1.weight' in new_k or 'fc2.weight' in new_k:
                    new_v = v.reshape((*v.shape, 1, 1))
                new_k = new_k.replace('fc1.', '0.')
                new_k = new_k.replace('dwconv.dwconv.', '1.')
                new_k = new_k.replace('fc2.', '4.')
                string += f'{new_k} {v.shape}-{new_v.shape}'
        # norm layer conversion
        elif k.startswith('norm') or k.startswith('model.backbone.norm'):
            # normX.weight/bias ---> layers.X.2.weight/bias
            if k.startswith('model.backbone.norm'):
                k = k.replace('model.backbone.', '')  # norm1.weight
            stage_i = int(k.split('.')[0].replace('norm', ''))  # # 针对daformer 把model.backbone置空
            new_k = k.replace(f'norm{stage_i}', f'layers.{stage_i-1}.2')
            new_v = v
        else:
            new_k = k
            new_v = v
        new_ckpt[new_k] = new_v
    return new_ckpt


def covert_mit_back(new_state_dict):
    old_dict = {}
    for k, v in new_state_dict.items():

        if 'decode_head' in k:
            old_dict[k] = v  # 这里要把head也加上

        elif k.startswith('backbone.layers') or k.startswith('layers'):  # 有些有backbone，有些没有
            k = k.replace('backbone.', '')  # 先把backbone给去掉
            prefix = 'backbone.'
            parts = k.split('.')
            layer_idx = int(parts[1])  # 第几个layer，这里注意backbone是从1开始，layer是从0开始
            module_idx = parts[2]  # 0是代表了哪个模块，attn，norm还是ffn等
            rest = '.'.join(parts[3:])  # 剩下的模块，直接添加
            if module_idx == '0':  # 这里是用来处理path_embed的，这里没问题，直接修改
                # This is the patch embedding layer
                new_k = f'{prefix}patch_embed{layer_idx + 1}.{rest}'  # 记得加1，这里是多加一个
                new_v = v
                if 'projection.' in new_k:  # 这里就不用管norm了
                    new_k = new_k.replace('projection.',
                                          'proj.')  # layers.0.0.projection.weight -> patch_embed1.proj.weight
                old_dict[new_k] = new_v  # 这里别忘了还有norm

            elif module_idx == '1':  # 包含了attn，norm，ffn这几个，是个硬骨头！
                # stage_i = int(k.split('.')[3])  # 每个vit里面有几个block  layers.0.1.x，这里用在第几个block
                new_k = f'{prefix}block{layer_idx + 1}.{rest}'

                if 'attn.in_proj_' in k:  # attn.attn.in_proj_  -> q和kv  attn.q.weight
                    # 开始是q，后面是kv，所以一般分为部分
                    if not v.shape[0] % 3 == 0:
                        print('wrong qkv !!')
                    embed_dim = v.shape[0] // 3  # 这里如果不是整数直接报错，因为必须是整数
                    q_dim = v[:embed_dim]
                    kv_dim = v[embed_dim:]
                    # 这里是同时放k和v，所以要注意一下啊
                    q_name = new_k.replace('attn.in_proj_', 'q.')
                    qkv_name = new_k.replace('attn.in_proj_', 'kv.')
                    old_dict[q_name] = q_dim
                    old_dict[qkv_name] = kv_dim

                elif 'attn.out_proj.' in new_k:  # attn.out_proj.  -> attn.proj
                    new_k = new_k.replace('attn.out_proj', 'proj')
                    old_dict[new_k] = v

                elif 'ffn.layers' in new_k:  # .layers.0.1.1.ffn.layers.0.weight  -> mlp.fc1.weight
                    # 0 <-- fc1.
                    # 1. <-- dwconv.dwconv.
                    # 4. <-- fc2
                    new_k = new_k.replace('ffn.layers.', 'mlp.')
                    if '0.weight' in new_k or '4.weight' in new_k:
                        new_v = v.squeeze()
                    else:
                        new_v = v
                    new_k = new_k.replace('mlp.0.', 'mlp.fc1.')
                    new_k = new_k.replace('mlp.1.', 'mlp.dwconv.dwconv.')  # 这个的v不用变，都是4dim，对应cnn
                    new_k = new_k.replace('mlp.4.', 'mlp.fc2.')
                    old_dict[new_k] = new_v

                else:  # sr和norm这两个不用改
                    old_dict[new_k] = v

            elif module_idx == '2':  # layers.0.2.weight  -> norm{state}.weight
                # k is enough, not new_k
                new_k = k.replace('layers.{}.2'.format(layer_idx), 'norm{}'.format(layer_idx + 1))
                new_v = v
                old_dict['backbone.{}'.format(new_k)] = new_v  # 因为前面把backbone.去掉了，这里要加回来


        else:
            if 'backbone.' not in k:
                k = 'backbone.{}'.format(k)
            old_dict[k] = v

    return old_dict


def main():
    # src = './data/Pth/DAFormer/211108_0934_syn2cs_daformer_s1_e7524/latest.pth'
    src = './data/Pth/DAFormer/211108_1622_gta2cs_daformer_s0_7f24c/latest.pth'
    # src = './data/Pth/Segformer/mit_b5_20220624-658746d9.pth'
    # src = './data/Pth/MixViT/mit_b5.pth'

    checkpoint = CheckpointLoader.load_checkpoint(src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_mit(state_dict)
    save_name = './gta2cs.pth'
    mmengine.mkdir_or_exist(osp.dirname(save_name))
    torch.save(weight, save_name)


def mit_change_fedseg_tta_prompt():
    # cs不用变，cs的已经是旧的版本。所以就改变acdc和nthu的就行
    root = './checkpoints/FedSeg_Prompt'
    # root = './checkpoints/FedSeg'

    for domain in os.listdir(root):
        if domain != 'cs':
            print('Change {} back!!'.format(domain))
            model_path = osp.join(root, domain, 'model.pth')
            new_state_dict = torch.load(model_path)
            old_state_dict = covert_mit_back(new_state_dict)
            torch.save(old_state_dict, osp.join(root, domain, 'model_old.pth'))


if __name__ == '__main__':
    # main()

    # model_pths_dict = {
    #     'mit_b5': './data/Pth/Segformer/mit_b5_20220624-658746d9.pth',
    #     'ade20k': './data/Pth/Segformer/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth',
    #     'cs': './data/Pth/Segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth',
    #     'gta': './data/Pth/Segformer/dataset_pths/gta.pth',
    #     'synthia': './data/Pth/Segformer/dataset_pths/synthia.pth',
    #     'acdc': './data/Pth/Segformer/dataset_pths/acdc.pth',
    # }

    # 那FedTTA训练的checkpoint转回原来mit的形式，才能对linear做upscaling，实现model merging
    mit_change_fedseg_tta_prompt()

    # state = torch.load('./data/Pth/Segformer/mit_b5.pth')
    # convert_mit(state)