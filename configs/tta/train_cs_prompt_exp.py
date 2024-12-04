_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    # '../_base_/models/daformer.py',
    '../_base_/datasets/cs_dataset_brightness.py',
    '../_base_/default_runtime.py'
]

# checkpoint = './data/Pth/DAFormer/gta_dg_daformer_shade_29938/latest.pth'
# checkpoint='./data/Pth/Segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
# checkpoint = './Run_tta/GT_exp/cs_prompt/model_lowpass.pth'
# checkpoint = './fed_avg_gtadg_cs_5000iter.pth'  # pretrain on FedAvg
# checkpoint = './fed_avg_gtadg_cs_5000iter_prompt.pth'
checkpoint='./model_pretrain_cs.pth'

# test_dataloader = dict(batch_size=1, num_workers=4)

test_cfg = dict(type='TestLoop')

model = dict(
    backbone=dict(
        # 是否用prompt，不需要就注释掉
        type='MiT_EVP',  # MiT_EVP, mit_b5_daformer_prompt
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        num_layers=[3, 6, 40, 3],
    ),
    # decode head这里要改一下记得，不然也是对不上的
    # decode_head=dict(
    #     decoder_params=dict(
    #         fusion_cfg=dict(
    #             _delete_=True,
    #             type='aspp',
    #             sep=True,
    #             dilations=(1, 6, 12, 18),
    #             pool=False,
    #             act_cfg=dict(type='ReLU'),
    #             norm_cfg=dict(type='BN', requires_grad=True)))
    # )
)

# 这里为了用上几个dataset，一些params是必须设置的
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

val_dataloader = dict(batch_size=1)
# test_dataloader = dict(batch_size=2,
#                        dataset=dict(
#                            data_prefix=dict(
#                                img_path='leftImg8bit_corrupt/brightness/5/train', seg_map_path='gtFine/train'),
#                        )
#                        )

test_dataloader = dict(batch_size=2,
                       dataset=dict(
                           type='ACDCDataset',
                           data_root='./data/Segmentation/ACDC',
                           data_prefix=dict(
                               img_path='rgb_anon/night/train', seg_map_path='gt/night/train')
                       )
                       )

train_dataloader = dict(batch_size=1)

default_hooks = dict(
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
