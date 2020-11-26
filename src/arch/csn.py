from torch import nn
import torch
from pathlib import Path
from src.arch.backbone import Backbone
from src.util.fetch import Fetcher

ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb = 'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth'


class irCSN_152(Backbone):
    @property
    def groups(self) -> [[nn.Module]]:
        return [[self.backbone.conv1],
                [self.backbone.layer1],
                [self.backbone.layer2],
                [self.backbone.layer3],
                [self.backbone.layer4],
                [self.cls_head]]

    def __init__(self, num_classes):
        checkpoints = Fetcher().load(ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb, Path('.'))
        checkpoints_exp = checkpoints.parent.joinpath(f'{checkpoints.name}_exp.pt')
        if not checkpoints_exp.exists():
            input = torch.load(checkpoints)['state_dict']
            out = {k[9:]: v for k, v in input.items()}
            out = dict(state_dict=out)
            torch.save(out, checkpoints.parent.joinpath(checkpoints_exp))

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