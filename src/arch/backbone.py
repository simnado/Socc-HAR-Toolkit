from abc import abstractmethod
from torch import nn
from mmaction.models import build_recognizer


class Backbone(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        model = build_recognizer(config, train_cfg=None, test_cfg=None)
        self.backbone = model.backbone
        self.cls_head = model.cls_head
        setattr(self, 'example_input_array', None)

    @property
    @abstractmethod
    def groups(self) -> [[nn.Module]]:
        pass

    #@abstractmethod
    #@staticmethod
    #def provide_pretrained_weights(dest_path: Path) -> Path:
    #    pass

    def forward(self, x):

        x = self.backbone(x)
        x = self.cls_head(x)
        return x
