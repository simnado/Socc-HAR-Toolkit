import math
import random
import torch
from torchvision.transforms import _functional_video as T


class VideoTransformation(object):

    def __init__(self, res=112, do_augmentation=False):
        self.res = res
        self.do_augmentation = do_augmentation

    def __call__(self, frames: torch.Tensor):

        assert isinstance(frames, torch.Tensor), 'invalid input'

        num_frames = frames.shape[0]  # t x c x h x w

        # todo: use gpu-batch-transformations -> classifier
        width = frames.shape[2]
        height = frames.shape[1]
        left = random.randint(0, math.floor(0.1 * width))
        top = random.randint(0, math.floor(0.1 * height))

        flip = random.random() > 0.5

        frames = T.to_tensor(frames)
        if self.do_augmentation:
            if flip:
                frames = T.hflip(frames)
            # one of nearest, linear, bilinear, trilinear, area
            frames = T.resized_crop(frames, top, left, math.floor(height * 0.9), math.floor(width * 0.9),
                                    (self.res, self.res), 'area')
        else:
            frames = T.center_crop(frames, (math.floor(height * 0.9), math.floor(width * 0.9)))
            frames = T.resize(frames, (self.res, self.res), interpolation_mode='bilinear')

        return frames
