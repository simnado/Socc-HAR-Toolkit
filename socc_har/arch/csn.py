from typing import Optional
from torch import nn
from pathlib import Path
from .backbone import Backbone
from ..util.fetch import Fetcher


class irCSN_152(Backbone):
    @staticmethod
    def provide_pretrained_weights() -> Optional[Path]:
        url = 'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth'
        return Fetcher().load(url, Path('.'))

    @property
    def groups(self) -> [[nn.Module]]:
        return [[self.backbone.conv1],
                [self.backbone.layer1],
                [self.backbone.layer2],
                [self.backbone.layer3],
                [self.backbone.layer4],
                [self.cls_head]]

    def __init__(self, num_classes):
        model = dict(
            type='Recognizer3D',
            backbone=dict(
                type='ResNet3dCSN',
                pretrained2d=False,
                pretrained=None,  # 'modelzoo/irCSN_152_ig65m_from_scratch_lite_new.pth',
                depth=152,
                with_pool2=False,
                bottleneck_mode='ir',
                norm_eval=False,
                # bn_frozen=True,
                zero_init_residual=False),
            cls_head=dict(
                type='I3DHead',
                num_classes=num_classes,
                in_channels=2048,
                spatial_type='avg',
                dropout_ratio=0.5,
                init_std=0.01))
        super().__init__(model)