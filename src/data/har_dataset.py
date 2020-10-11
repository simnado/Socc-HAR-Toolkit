import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.transforms import _transforms_video as v_transforms
from tqdm.auto import tqdm
from src.data import DatabaseHandle, VideoTransformation


class HarDataset(Dataset):
    def __init__(self, database: DatabaseHandle, res: int, classes: [str], video_metadata: dict,
                 mean=None, std=None, normalized=True, do_augmentation=False,
                 num_frames=32, fps=15, limit_per_class=1000, seed=94,
                 background_min_distance=3, period_max_distance=10, min_action_overlap=0.99, allow_critical=False, num_workers=4):
        """
        Initializes a dataset
        @param classes: array of classes used. default will use all classes specified in csv
        @param res: spatial base resolution
        @param mean:
        @param std:
        @param normalized:
        @param num_frames: number of frame to be sampled
        """
        super(object, self)
        np.random.seed(seed)

        self.background_min_distance = background_min_distance
        self.period_max_distance = period_max_distance  # time border for samples outside a period
        self.min_action_overlap = min_action_overlap
        self.allow_critical = allow_critical

        self.seed = seed
        self.classes = classes
        self.normalized = normalized
        self.do_augmentation = do_augmentation
        self.num_frames = num_frames
        self.res = res
        self._limit_per_class = limit_per_class
        self.fps = fps
        self.duration = self.num_frames / self.fps
        self.database = database
        self.video_metadata = video_metadata
        self.mean = mean
        self.std = std
        self.num_workers = num_workers
        self._id_2_index = dict()

        # indices of clips
        self.x = []
        # labels
        self.y = []
        self.info = []
        self.total_length = 0

        # load or precompute video metadata
        self.clips_per_video = dict()
        self.video_clips = self.get_video_clips()

        # set samples and calc stats
        self.get_samples()

        # precompute means and stds
        if self.normalized and (self.mean is None or self.std is None):
            self.normalized = False
            self.mean, self.std = self.precompute_mean_and_std()
            print(f'mean={self.mean}, std={self.std}')
            self.normalized = True

    def get_samples(self):
        y = []

        print('collecting samples')
        for idx in tqdm(range(len(self.video_clips))):
            video_idx, clip_idx = self.video_clips.get_clip_location(idx)
            keys = self.video_metadata['sac_keys'][video_idx]
            path = self.video_metadata['video_paths'][video_idx]
            records = [self.database.database[key] for key in keys]

            curr_record = records[0]
            key = keys[0]
            if len(keys) == 2 and clip_idx > curr_record['segment'][1] + self.period_max_distance:
                curr_record = records[1]
                key = keys[1]

            annotations = curr_record['annotations']
            period_start = curr_record['segment'][0]
            period_end = curr_record['segment'][1]

            if clip_idx < period_start - self.period_max_distance or clip_idx > period_end + self.period_max_distance:
                # sample is too far away from period boundaries
                continue

            vec, json, critical = self._get_annotations(annotations, clip_idx)

            if vec is None or (critical and not self.allow_critical):
                continue
            else:
                self.x.append(idx)
                y.append(vec)
                video_id = curr_record['url'].split('v=')[1] if 'youtube' in curr_record['url'] else curr_record['url'].split('id=')[1]
                sample_id = f"{key}@{clip_idx}"
                self.info.append(dict(key=key, start=clip_idx, end=clip_idx+self.duration, path=path, video=video_id, critical=critical, annotations=json, id=sample_id))
                self._id_2_index[sample_id] = len(self.info) - 1

        self.y = torch.stack(y)

    def _get_annotations(self, annotations: [], clip_idx: int):
        critical = False
        vec = torch.zeros((len(self.classes)))
        prev_border = 20
        next_border = 20
        json = []

        for idx, anno in enumerate(annotations):
            if anno['segment'][1] < clip_idx:
                prev_border = min(prev_border, clip_idx - anno['segment'][1])
                # next annotation is in the past
                continue

            next_border = min(next_border, anno['segment'][0] - clip_idx)
            overlap = self.overlap(anno['segment'], [clip_idx, clip_idx + self.duration])

            if overlap == 0:
                # assert anno['segment'][0] > clip_idx, 'next annotation should be scheduled after current timestamp'
                # next annotation is in the future
                continue

            elif overlap > self.min_action_overlap:
                # attach overlapping annotation
                label = anno['label']
                index = self.classes.index(label)
                vec[index] = 1
                json.append(anno)
            else:
                # annotation overlap is too small
                continue

        # additional check if background sample is to near to real actions
        if vec is not None and torch.sum(vec).item() == 0:
            if prev_border < self.background_min_distance or next_border < self.background_min_distance:
                critical = True

        return vec, json, critical

    @staticmethod
    def overlap(a: [int], b: [int]):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    def __getitem__(self, index):
        x = self.get_tensor(index)

        # todo: move to data loader??
        if self.normalized:
            x = v_transforms.NormalizeVideo(mean=self.mean, std=self.std)(x)

        y = self.y[index]
        info = self.info[index]

        return x, y, {**info, 'index': index}

    def __len__(self):
        return len(self.x)

    def get_tensor(self, index, resize=True):
        print(index)
        clip_idx = self.x[index]
        try:
            frames = self.video_clips.get_clip(clip_idx)[0]
        except IndexError as err:
            print(f'cannot access video {self.info[index]["video"]} at {self.info[index]["start"]}-{self.info[index]["end"]}. Limit is {self.clips_per_video[self.info[index]["path"]]}')
            raise err
        frames = VideoTransformation(res=self.res if resize else 360, do_augmentation=self.do_augmentation)(frames)

        return frames

    def precompute_mean_and_std(self):
        mean_channel_1 = []
        std_channel_1 = []
        for i in tqdm(range(100)):  # len(self) - 1
            x = self[i][0]
            mean_channel_1.append(x.mean(1).mean(1).mean(1))
            std_channel_1.append(x.std(1).std(1).std(1))
        mean = torch.stack(mean_channel_1).mean(0)
        std = torch.stack(std_channel_1).mean(0)
        return mean.tolist(), std.tolist()

    def get_video_clips(self):

        # segmentation: distance between clips is one second. clips will probably overlap
        clips = VideoClips(self.video_paths,
                           clip_length_in_frames=self.num_frames,
                           frames_between_clips=self.fps,
                           frame_rate=self.fps,
                           num_workers=self.num_workers,
                           _precomputed_metadata=self.video_metadata
                           )

        # generate backward-mapping and cut off actions out of range
        for idx, video_path in enumerate(self.video_paths):
            clips_before = clips.cumulative_sizes[idx - 1] if idx > 0 else 0
            clips_after = clips.cumulative_sizes[idx]
            num_clips = clips_after - clips_before
            self.clips_per_video[video_path] = num_clips

        return clips

    @property
    def video_paths(self):
        return self.video_metadata['video_paths']
