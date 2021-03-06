import torch
from torch.utils.data import Dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision import io
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from .database import DatabaseHandle
from .video_transform import VideoTransformation


class HarDataset(Dataset):
    def __init__(self, database: DatabaseHandle, res: int, classes: [str], video_metadata: dict,
                 do_augmentation=False,
                 num_frames=32, num_frames_per_sample=None, num_chunks=1, fps=15,
                 clip_offset=None,
                 background_min_distance=3, period_max_distance=10, min_action_overlap=0.9, allow_critical=False,
                 num_workers=4, backend='av'):
        """
        Initializes a dataset
        @param classes: array of classes used. default will use all classes specified in csv
        @param res: spatial base resolution
        @param num_frames: number of frame to be sampled
        """
        super()

        self.backend = backend

        self.background_min_distance = background_min_distance
        self.period_max_distance = period_max_distance  # time border for samples outside a period
        self.min_action_overlap = min_action_overlap
        self.allow_critical = allow_critical

        self.classes = classes
        self.do_augmentation = do_augmentation
        self.num_frames = num_frames
        self.num_frames_per_sample = num_frames_per_sample
        if num_frames_per_sample is None:
            self.num_frames_per_sample = num_frames
        self.num_chunks = num_chunks
        self.res = res
        self.fps = fps
        self.duration = self.num_frames_per_sample / self.fps
        self.clip_offset = clip_offset
        if clip_offset is None:
            self.clip_offset = fps  # samples a clip with any second
        self.database = database
        self.video_metadata = video_metadata

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

    def get_samples(self):
        y = []

        for keys in self.video_metadata['sac_keys']:
            keys.sort()

        print('collecting samples')
        for idx in tqdm(range(len(self.video_clips))):
            video_idx, clip_idx = self.video_clips.get_clip_location(idx)
            start = clip_idx * self.clip_offset / self.fps
            end = start + self.duration
            keys = self.video_metadata['sac_keys'][video_idx]
            path = self.video_metadata['video_paths'][video_idx]
            records = [self.database.database[key] for key in keys]

            curr_record = records[0]
            key = keys[0]
            if len(keys) == 2 and start > curr_record['segment'][1] + self.period_max_distance:
                curr_record = records[1]
                key = keys[1]

            annotations = [anno for anno in curr_record['annotations'] if 'deleted' not in anno or anno['deleted'] == False]
            period_start = curr_record['segment'][0]
            period_end = curr_record['segment'][1]

            if end < period_start - self.period_max_distance or start > period_end + self.period_max_distance:
                # sample is too far away from period boundaries
                continue

            vec, json, critical = self._get_annotations(annotations, [start, end])
            video_id = curr_record['url'].split('v=')[1] if 'youtube' in curr_record['url'] else \
                curr_record['url'].split('id=')[1]
            if vec is None or (critical and not self.allow_critical):
                continue
            else:
                self.x.append(idx)
                y.append(vec)
                sample_id = f"{key}@{start:.2f}"
                self.info.append(
                    dict(key=key, start=start, end=end, path=path, video=video_id, critical=critical, annotations=json,
                         id=sample_id))
                self._id_2_index[sample_id] = len(self.info) - 1

        self.y = torch.stack(y)

    def _get_annotations(self, annotations: [], clip_segment: [int]):
        critical = False
        vec = torch.zeros((len(self.classes)))
        prev_border = 20
        next_border = 20
        json = []

        for idx, anno in enumerate(annotations):
            if anno['segment'][1] < clip_segment[0]:
                prev_border = min(prev_border, clip_segment[0] - anno['segment'][1])
                # next annotation is in the past
                continue

            next_border = min(next_border, anno['segment'][0] - clip_segment[0])
            overlap = self.overlap(anno['segment'], clip_segment)

            if overlap == 0:
                # assert anno['segment'][0] > clip_idx, 'next annotation should be scheduled after current timestamp'
                # next annotation is in the future
                continue

            elif overlap > self.min_action_overlap:
                # attach overlapping annotation
                label = anno['label']
                if label not in self.classes:
                    # if class is excluded, ignore it
                    continue
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

    def get_row(self, key, start):
        sample_id = f"{key}@{start:.2f}"
        return self._id_2_index[sample_id]

    def __getitem__(self, index):
        x = self.get_tensor(index)

        y = self.y[index]
        info = self.info[index]

        return x, y, {**info, 'index': index}

    def __len__(self):
        return len(self.x)

    def get_tensor(self, index, resize=True, vr=False, chunked=True):
        clip_index = self.x[index]

        video_idx, clip_idx = self.video_clips.get_clip_location(clip_index)
        video_path = self.video_clips.video_paths[video_idx]
        meta = self.info[index]

        if vr:
            # todo: wait for pre-compiled version including VideoReader
            #frames = []
            #reader = io.VideoReader(video_path, "video")
            #for frame in itertools.takewhile(lambda x: x['pts'] <= meta['end'], reader.seek(meta['start'])):
            #    frames.append(frame['data'])
            #frames = torch.cat(frames)
            frames, _, _ = io.read_video(video_path, meta['start'], meta['end'], pts_unit='sec')
        else:
            clip_pts = self.video_clips.clips[video_idx][clip_idx]
            start_pts = clip_pts[0].item()
            end_pts = clip_pts[-1].item()
            frames, _, _ = io.read_video(video_path, start_pts, end_pts, pts_unit='pts')
            # todo: maybe faster?
            # frames, _, _ = io._video_opt._read_video_from_file(video_path, video_width=224, video_height=224, video_pts_range=(start_pts, end_pts), read_audio_stream=False)

        clip_frames = len(frames)
        if clip_frames < self.num_frames:
            print(f'WARNING: tensor on index={index} has only {clip_frames} frames')

        resample_idx = torch.linspace(0, clip_frames - 1, self.num_frames_per_sample, dtype=torch.int16).tolist()
        frames = frames[resample_idx]

        # todo: run on gpu
        # T, H, W, C -> T, C, H, W
        frames = frames.permute((0, 3, 1, 2))
        frames = VideoTransformation(res=self.res, do_augmentation=self.do_augmentation, pass_through=not resize)(frames)
        # T, H, W, C -> C, T, H, W
        frames = frames.permute((1, 0, 2, 3))

        # reshape if test loop from 4D to 5D (Chunks x C x T x H x W)
        if chunked:
            frames = self._get_chunks(frames)
        assert frames.shape[2] >= self.num_frames, f'chunked tensor on index={index} has only {frames.shape[2]} frames'

        return frames

    def _get_chunks(self, x):
        """see mmaction2 (SampleFrames)
        """
        assert x.shape[0] == 3
        x_len = x.shape[1]  # C x T x S^2
        if self.num_chunks == 1:
            return torch.unsqueeze(x, 0)
        avg_interval = (x_len - self.num_frames) / float(self.num_chunks - 1)
        if self.num_frames < x_len - 1:
            clip_offsets = (np.arange(self.num_chunks) * avg_interval).astype(np.int)
        else:
            print('cannot sample segments')
            clip_offsets = np.zeros((self.num_chunks,), dtype=np.int)
        x = torch.stack([x[:, start:start + self.num_frames] for start in clip_offsets])
        assert x.shape[1] == 3
        return x  # Chunks x C x T x S^2

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
                           clip_length_in_frames=self.num_frames_per_sample,
                           frames_between_clips=self.clip_offset,
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

    @property
    def df(self):
        return pd.from_dict(self.meta)