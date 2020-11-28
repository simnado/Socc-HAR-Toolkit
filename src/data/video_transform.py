import math
import torch
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, CenterCrop, Resize, RandomResizedCrop, ColorJitter, Lambda


class VideoTransformation(object):

    def __init__(self, res=112, do_augmentation=False, pass_through=False):
        self.res = res
        self.do_augmentation = do_augmentation
        self.pass_through = pass_through

    def __call__(self, frames: torch.Tensor):

        assert isinstance(frames, torch.Tensor), 'invalid input'

        num_frames = frames.shape[0]  # t x c x h x w

        # todo: use gpu-batch-transformations -> classifier
        width = frames.shape[3]
        height = frames.shape[2]

        if self.do_augmentation and not self.pass_through:
            transforms = Compose([
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                RandomHorizontalFlip(),
                RandomResizedCrop(size=(self.res, self.res), scale=(0.85, 1), ratio=(1.78, 1.78)),
                Lambda(lambda data: data / 255.0)
            ])
        elif not self.pass_through:
            transforms = Compose([
                CenterCrop((math.floor(height * 0.9), math.floor(width * 0.9))),
                Resize((self.res, self.res)),
                Lambda(lambda data: data / 255.0)
            ])
        else:
            transforms = Compose([
                Resize((self.res, self.res)),
                Lambda(lambda data: data / 255.0)
            ])
        return transforms(frames)
