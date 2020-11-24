from pathlib import Path
from torch import nn
import torch
from src.arch.backbone import Backbone
from src.util.fetch import Fetcher

x3d_s_facebook_13x6x1_kinetics400_rgb = 'https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_s_facebook_13x6x1_kinetics400_rgb_20201027-623825a0.pth'
x3d_m_facebook_16x5x1_kinetics400_rgb = 'https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'


class X3D(Backbone):
    @property
    def groups(self) -> [[nn.Module]]:
        return [[self.backbone.slow_path.conv1, self.backbone.fast_path.conv1, self.backbone.slow_path.conv1_lateral],
                [self.backbone.slow_path.layer1, self.backbone.fast_path.layer1, self.backbone.slow_path.layer1_lateral],
                [self.backbone.slow_path.layer2, self.backbone.fast_path.layer2, self.backbone.slow_path.layer2_lateral],
                [self.backbone.slow_path.layer3, self.backbone.fast_path.layer3, self.backbone.slow_path.layer3_lateral],
                [self.backbone.slow_path.layer4, self.backbone.fast_path.layer4],
                [self.cls_head.fc_cls]]


class X3D_S(X3D):

    def __init__(self, num_classes: int):
        checkpoints = Fetcher().load(x3d_s_facebook_13x6x1_kinetics400_rgb, Path('.'))
        checkpoints_exp = checkpoints.parent.joinpath(f'{checkpoints.name}_exp.pt')
        if not checkpoints_exp.exists():
            input = torch.load(checkpoints)['state_dict']
            out = {k[9:]: v for k, v in input.items()}
            out = dict(state_dict=out)
            torch.save(out, checkpoints.parent.joinpath(checkpoints_exp))

        model = dict(
            type='Recognizer3D',
            backbone=dict(
                type='X3D',
                gamma_w=1,
                gamma_b=2.25,
                gamma_d=2.2
            ),
            cls_head=dict(
                type='X3DHead',
                in_channels=432,
                num_classes=num_classes,
                spatial_type='avg',
                dropout_ratio=0.5,
                fc1_bias=False
            )
        )
        super().__init__(model)


class X3D_M(X3D):

    def __init__(self, num_classes: int):
        checkpoints = Fetcher().load(x3d_m_facebook_16x5x1_kinetics400_rgb, Path('.'))
        checkpoints_exp = checkpoints.parent.joinpath(f'{checkpoints.name}_exp.pt')
        if not checkpoints_exp.exists():
            input = torch.load(checkpoints)['state_dict']
            out = {k[9:]: v for k, v in input.items()}
            out = dict(state_dict=out)
            torch.save(out, checkpoints.parent.joinpath(checkpoints_exp))

        model = dict(
            type='Recognizer3D',
            backbone=dict(
                type='X3D',
                gamma_w=1,
                gamma_b=2.25,
                gamma_d=2.2
            ),
            cls_head=dict(
                type='X3DHead',
                in_channels=432,
                num_classes=num_classes,
                spatial_type='avg',
                dropout_ratio=0.5,
                fc1_bias=False
            )
        )
        super().__init__(model)
