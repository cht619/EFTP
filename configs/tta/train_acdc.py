_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/acdc_dataset.py',
    '../_base_/default_runtime.py'
]

# checkpoint = './data/Pth/Segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
checkpoint = './Run/checkpoint/FedSeg_ACDC/rain/train_from_scratch/model.pth'
# checkpoint = './Run/PromptExp/snow/model.pth'


test_cfg = dict(type='TestLoop')


model = dict(
    backbone=dict(
        # 是否用prompt，不需要就注释掉
        # type='MiT_EVP',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        num_layers=[3, 6, 40, 3]),
        decode_head=dict(in_channels=[64, 128, 320, 512]))

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
test_dataloader = dict(batch_size=1)


train_dataloader = dict(batch_size=1)


default_hooks = dict(
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)