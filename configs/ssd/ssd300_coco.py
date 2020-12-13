_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
# optimizer_config = dict(type="CoteachingOptimizerHook",
#                         cooperative_method="naive",
#                         dr_config=dict(max_drop_rate=0.2, num_gradual=20))
# optimizer_config = dict(type="DistillationOptimizerHook",
#                         distillation_method="naive",
#                         distill_config=dict(
#                             alpha=0.2,
#                             beta=1.5,
#                             gamma=2.0,
#                             temperature=1.0,
#                             checkpoint="checkpoints/ssd300_coco_20200307-a92d2092.pth",
#                             train_cfg = dict(
#                                 assigner=dict(
#                                     type='MaxIoUAssigner',
#                                     pos_iou_thr=0.5,
#                                     neg_iou_thr=0.5,
#                                     min_pos_iou=0.,
#                                     ignore_iof_thr=-1,
#                                     gt_max_assign_all=False),
#                                 smoothl1_beta=1.,
#                                 allowed_border=-1,
#                                 pos_weight=-1,
#                                 neg_pos_ratio=3,
#                                 debug=False
#                             ),
#                             test_cfg = dict(
#                                 nms=dict(type='nms', iou_threshold=0.45),
#                                 min_bbox_size=0,
#                                 score_thr=0.02,
#                                 max_per_img=200
#                             ),
#                             model = dict(
#                                 type='SingleStageDetector',
#                                 pretrained='open-mmlab://vgg16_caffe',
#                                 backbone=dict(
#                                     type='SSDVGG',
#                                     input_size=300,
#                                     depth=16,
#                                     with_last_pool=False,
#                                     ceil_mode=True,
#                                     out_indices=(3, 4),
#                                     out_feature_indices=(22, 34),
#                                     l2_norm_scale=20),
#                                 neck=None,
#                                 bbox_head=dict(
#                                     type='SSDHead',
#                                     in_channels=(512, 1024, 512, 256, 256, 256),
#                                     num_classes=80,
#                                     anchor_generator=dict(
#                                         type='SSDAnchorGenerator',
#                                         scale_major=False,
#                                         input_size=300,
#                                         basesize_ratio_range=(0.15, 0.9),
#                                         strides=[8, 16, 32, 64, 100, 300],
#                                         ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
#                                     bbox_coder=dict(
#                                         type='DeltaXYWHBBoxCoder',
#                                         target_means=[.0, .0, .0, .0],
#                                         target_stds=[0.1, 0.1, 0.2, 0.2]
#                                     )
#                                 )
#                             )
#                         )
#                     )
