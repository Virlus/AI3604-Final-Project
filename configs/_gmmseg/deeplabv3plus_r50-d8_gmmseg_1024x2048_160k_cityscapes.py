_base_ = [
    './models/deeplabv3plus_r50-d8-gmm.py', '../_base_/datasets/cityscapes_1024x2048.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]


model = dict(
    # pretrained='/mnt/petrelfs/yuwenye/GMMSeg/configs/_gmmseg/pretrain/mit_b5_mmseg.pth',
    decode_head=dict(
        decoder_params=dict(
            # * basic gmm setup
            embed_dim=64,
            num_components=5,
            factor_n=2,
            factor_c=1,
            factor_p=1,
        ),
    )
)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=1, workers_per_gpu=1)