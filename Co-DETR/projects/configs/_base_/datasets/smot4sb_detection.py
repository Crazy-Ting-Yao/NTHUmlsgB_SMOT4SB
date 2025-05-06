# dataset settings
_base_ = 'coco_detection.py'
dataset_type = 'CocoDataset'
data_root = 'dataset/'
classes = ('bird',)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        classes=classes,
        img_prefix=data_root + 'train/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        classes=classes,
        img_prefix=data_root + 'train/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'pub_test_annotations/0001.json',
        classes=classes,
        img_prefix=data_root + 'pub_test/'))
evaluation = dict(interval=50, metric='bbox')
