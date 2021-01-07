from pathlib import Path
from typing import Optional
from torch import nn
from .backbone import Backbone
from ..util.fetch import Fetcher


class SlowFast(Backbone):
    @property
    def groups(self) -> [[nn.Module]]:
        return [[self.backbone.slow_path.conv1, self.backbone.fast_path.conv1, self.backbone.slow_path.conv1_lateral],
                [self.backbone.slow_path.layer1, self.backbone.fast_path.layer1, self.backbone.slow_path.layer1_lateral],
                [self.backbone.slow_path.layer2, self.backbone.fast_path.layer2, self.backbone.slow_path.layer2_lateral],
                [self.backbone.slow_path.layer3, self.backbone.fast_path.layer3, self.backbone.slow_path.layer3_lateral],
                [self.backbone.slow_path.layer4, self.backbone.fast_path.layer4],
                [self.cls_head]]


class SlowFast4x16_50(SlowFast):

    @staticmethod
    def provide_pretrained_weights() -> Optional[Path]:
        url = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth'
        return Fetcher().load(url, Path('.'))

    def __init__(self, num_classes: int, label_smooth_eps=0.0):
        model = dict(
            type='Recognizer3D',
            backbone=dict(
                type='ResNet3dSlowFast',
                pretrained=None,
                resample_rate=8,  # tau
                speed_ratio=8,  # alpha
                channel_ratio=8,  # beta_inv
                slow_pathway=dict(
                    type='resnet3d',
                    depth=50,
                    pretrained=None,
                    lateral=True,
                    conv1_kernel=(1, 7, 7),
                    dilations=(1, 1, 1, 1),
                    conv1_stride_t=1,
                    pool1_stride_t=1,
                    inflate=(0, 0, 1, 1),
                    norm_eval=False),
                fast_pathway=dict(
                    type='resnet3d',
                    depth=50,
                    pretrained=None,
                    lateral=False,
                    base_channels=8,
                    conv1_kernel=(5, 7, 7),
                    conv1_stride_t=1,
                    pool1_stride_t=1,
                    norm_eval=False)),
            cls_head=dict(
                type='SlowFastHead',
                in_channels=2304,  # 2048+256
                num_classes=num_classes,
                spatial_type='avg',
                dropout_ratio=0.5,
                label_smooth_eps=label_smooth_eps
            ))
        super().__init__(model)


class SlowFast8x8_50(SlowFast):

    @staticmethod
    def provide_pretrained_weights() -> Optional[Path]:
        url = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth'
        return Fetcher().load(url, Path('.'))

    def __init__(self, num_classes: int, label_smooth_eps=0.0):
        model = dict(
            type='Recognizer3D',
            backbone=dict(
                type='ResNet3dSlowFast',
                pretrained=None,
                resample_rate=4,  # tau
                speed_ratio=4,  # alpha
                channel_ratio=8,  # beta_inv
                slow_pathway=dict(
                    type='resnet3d',
                    depth=50,
                    pretrained=None,
                    lateral=True,
                    fusion_kernel=7,
                    conv1_kernel=(1, 7, 7),
                    dilations=(1, 1, 1, 1),
                    conv1_stride_t=1,
                    pool1_stride_t=1,
                    inflate=(0, 0, 1, 1),
                    norm_eval=False),
                fast_pathway=dict(
                    type='resnet3d',
                    depth=50,
                    pretrained=None,
                    lateral=False,
                    base_channels=8,
                    conv1_kernel=(5, 7, 7),
                    conv1_stride_t=1,
                    pool1_stride_t=1,
                    norm_eval=False)),
            cls_head=dict(
                type='SlowFastHead',
                in_channels=2304,  # 2048+256
                num_classes=num_classes,
                spatial_type='avg',
                dropout_ratio=0.5,
                label_smooth_eps=label_smooth_eps
            ))
        super().__init__(model)
