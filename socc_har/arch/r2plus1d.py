from pathlib import Path
from typing import Optional
from torch import nn
from .backbone import Backbone
from ..util.fetch import Fetcher


class R2Plus1D_34(Backbone):

    @staticmethod
    def provide_pretrained_weights() -> Optional[Path]:
        url = 'https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb/r2plus1d_r34_32x2x1_180e_kinetics400_rgb_20200618-63462eb3.pth'
        return Fetcher().load(url, Path('.'))

    @property
    def groups(self) -> [[nn.Module]]:
        return [[self.backbone.conv1], [self.backbone.layer1], [self.backbone.layer2], [self.backbone.layer3], [self.backbone.layer4], [self.cls_head]]

    def __init__(self, num_classes: int):
        model = dict(
            type='Recognizer3D',
            backbone=dict(
                type='ResNet2Plus1d',
                depth=34,
                pretrained=None,
                pretrained2d=False,
                norm_eval=False,
                conv_cfg=dict(type='Conv2plus1d'),
                #norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3),
                norm_cfg=dict(type='BN3d', requires_grad=True, eps=1e-5),
                act_cfg=dict(type='ReLU', inplace=True),
                conv1_kernel=(3, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(1, 1, 1, 1),
                spatial_strides=(1, 2, 2, 2),
                temporal_strides=(1, 2, 2, 2),
                zero_init_residual=False),
            cls_head=dict(
                type='I3DHead',
                num_classes=num_classes,
                in_channels=512,
                spatial_type='avg',
                dropout_ratio=0.5,
                init_std=0.01))
        super().__init__(model)
