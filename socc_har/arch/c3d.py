from pathlib import Path
from typing import Optional
from .backbone import Backbone
from ..util.fetch import Fetcher


class C3D(Backbone):

    @staticmethod
    def provide_pretrained_weights() -> Optional[Path]:
        url = 'https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth'
        return Fetcher().load(url, Path('.'))

    def __init__(self, num_classes):
        model = dict(
            type='Recognizer3D',
            backbone=dict(
                type='C3D',
                pretrained=None,
                # noqa: E501
                style='pytorch',
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=None,
                act_cfg=dict(type='ReLU'),
                dropout_ratio=0.5,
                init_std=0.005),
            cls_head=dict(
                type='I3DHead',
                num_classes=num_classes,
                in_channels=4096,
                spatial_type=None,
                dropout_ratio=0.5,
                init_std=0.01))
        super().__init__(model)

    @property
    def groups(self):
        return [[self.backbone.conv1a],
                [self.backbone.conv2a],
                [self.backbone.conv3a, self.backbone.conv3b],
                [self.backbone.conv4a, self.backbone.conv4b],
                [self.backbone.conv5a, self.backbone.conv5b],
                [self.backbone.fc6],
                [self.backbone.fc7],
                [self.cls_head]]
