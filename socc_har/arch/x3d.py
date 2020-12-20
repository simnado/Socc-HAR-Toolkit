from pathlib import Path
from typing import Optional
from torch import nn
from .backbone import Backbone
from ..util.fetch import Fetcher


class X3D(Backbone):
    @property
    def groups(self) -> [[nn.Module]]:
        return [[self.backbone.conv1_s, self.backbone.conv1_t],
                [self.backbone.layer1],
                [self.backbone.layer2],
                [self.backbone.layer3],
                [self.backbone.layer4, self.backbone.conv5],
                [self.cls_head]]


class X3D_S(X3D):

    @staticmethod
    def provide_pretrained_weights() -> Optional[Path]:
        url = 'https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_s_facebook_13x6x1_kinetics400_rgb_20201027-623825a0.pth'
        return Fetcher().load(url, Path('.'))

    def __init__(self, num_classes: int):
        model = dict(
            type='Recognizer3D',
            backbone=dict(
                type='X3D',
                pretrained=None,
                gamma_w=2,
                gamma_b=2.25,
                gamma_d=2.2,
                norm_cfg=dict(type='BN3d', requires_grad=True),
                norm_eval=False
            ),
            cls_head=dict(
                type='X3DHead',
                in_channels=2048,
                num_classes=num_classes,
                spatial_type='avg',
                dropout_ratio=0.5,
                fc1_bias=False
            )
        )
        super().__init__(model)


class X3D_M(X3D):

    @staticmethod
    def provide_pretrained_weights() -> Optional[Path]:
        url = 'https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
        return Fetcher().load(url, Path('.'))

    def __init__(self, num_classes: int):

        model = dict(
            type='Recognizer3D',
            backbone=dict(
                type='X3D',
                pretrained=None,
                gamma_w=2,
                gamma_b=2.25,
                gamma_d=2.2
            ),
            cls_head=dict(
                type='X3DHead',
                in_channels=2048,
                num_classes=num_classes,
                spatial_type='avg',
                dropout_ratio=0.5,
                fc1_bias=False
            )
        )
        super().__init__(model)

class X3D_L(X3D):

    @staticmethod
    def provide_pretrained_weights() -> Optional[Path]:
        return None

    def __init__(self, num_classes: int):

        model = dict(
            type='Recognizer3D',
            backbone=dict(
                type='X3D',
                pretrained=None,
                gamma_w=2,
                gamma_b=2.25,
                gamma_d=5.0
            ),
            cls_head=dict(
                type='X3DHead',
                in_channels=2048,
                num_classes=num_classes,
                spatial_type='avg',
                dropout_ratio=0.5,
                fc1_bias=False
            )
        )
        super().__init__(model)
