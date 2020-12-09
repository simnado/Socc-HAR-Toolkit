from pathlib import Path
from typing import Optional
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from src.data import DatabaseHandle, HarDataset, PreProcessing, DataStats
from src.data.util import MediaDir
from src.util.fetch import DatabaseFetcher


class DataModule(LightningDataModule):

    def __init__(self, database: str, data_dir: str, num_frames: int, res: int, fps: int,
                 metadata_path: Optional[str],
                 batch_size=32,
                 classes=None, max_train_samples_per_class=500,
                 num_data_workers=None, seed=2147483647):
        super().__init__()
        self.seed = seed
        self.media_dir = MediaDir(data_dir)

        data_path = self.media_dir.datasets()
        database_path = DatabaseFetcher.load(database, data_path)

        assert database_path.exists(), 'Database does not exist'

        self.database = DatabaseHandle(database_path)

        self.max_train_samples_per_class = max_train_samples_per_class
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.res = res
        self.fps = fps

        self.classes = classes

        if self.classes is None:
            self.classes = self.database.classes

        self.datasets = dict()
        self.stats = dict()
        self.limit_per_class = dict(train=self.max_train_samples_per_class, val=50, test=100)

        self.num_train_samples = None

        self.test_loc_dataset = None

        self.num_data_workers = num_data_workers
        if num_data_workers is None:
            self.num_data_workers = max(4, torch.get_num_interop_threads() * 2)

        self.metadata_out_path = None
        self.precomputed_metadata_file = metadata_path

        self.video_metadata = None
        self.pre_processor = None

    def prepare_data(self, verbose=False):
        self.pre_processor = PreProcessing(self.database, self.media_dir.root, Path(self.precomputed_metadata_file),
                                           res=360)
        self.video_metadata = self.pre_processor.prepare_data(verbose)

        fps = self.video_metadata['train']['video_fps'] + self.video_metadata['val']['video_fps'] + self.video_metadata['test']['video_fps']
        invalid_frame_rates = [idx for idx, fps in enumerate(fps) if fps < self.fps]
        if len(invalid_frame_rates):
            print(f'WARNING: found {len(invalid_frame_rates)} clips with lower frame rate than {self.fps}')

    def setup(self, stage: Optional[str] = None):

        assert self.video_metadata, "No video metadata found, run prepare_data() first"

        assert stage in ['fit', 'test']

        # split dataset
        if stage == 'fit':
            # allowing spatial and temporal augmentation (one sample each second), no critical
            self.datasets['train'] = HarDataset(database=self.database,
                                                video_metadata=self.video_metadata['train'],
                                                res=self.res, classes=self.classes,
                                                do_augmentation=True,
                                                num_frames=self.num_frames, fps=self.fps, clip_offset=self.fps,
                                                num_chunks=1,
                                                num_workers=0)
            self.stats['train'] = DataStats('train', self.datasets['train'], self.limit_per_class['train'], seed=self.seed)
            limit = self.limit_per_class['train']
            self.num_train_samples = sum([limit] + [min(limit, self.stats['train'].samples[cls_idx]) for cls_idx, _ in enumerate(self.classes)])

            # allowing no augmentation, no overlap, no critical
            self.datasets['val'] = HarDataset(database=self.database,
                                              video_metadata=self.video_metadata['val'],
                                              res=self.res, classes=self.classes,
                                              do_augmentation=False,
                                              num_frames=self.num_frames, fps=self.fps, clip_offset=self.num_frames,
                                              num_chunks=1,
                                              num_workers=0)
            self.stats['val'] = DataStats('val', self.datasets['val'], self.limit_per_class['val'], seed=self.seed)

        if stage == 'test':
            # allowing no augmentation, no overlap, but critical and fixed sampling scheme
            self.datasets['test'] = HarDataset(database=self.database,
                                               video_metadata=self.video_metadata['test'],
                                               res=self.res, classes=self.classes,
                                               do_augmentation=False,
                                               # 10 sec clips, no overlap
                                               num_frames=self.num_frames, fps=self.fps, clip_offset=self.fps * 10,
                                               num_chunks=5, num_frames_per_sample=self.fps * 10,
                                               allow_critical=True,
                                               num_workers=0)
            self.stats['test'] = DataStats('test', self.datasets['test'], self.limit_per_class['test'], seed=self.seed)

    def train_dataloader(self):
        assert "train" in self.datasets, "No TrainingSet build, run setup('fit')"

        print(f'sample {self.num_train_samples}/{len(self.datasets["train"])} random clips')
        # should be different each iteration
        sampler = WeightedRandomSampler(weights=self.stats['train'].weights, num_samples=int(self.num_train_samples), replacement=False)
        dl = DataLoader(self.datasets["train"], batch_size=self.batch_size, sampler=sampler,
                        num_workers=self.num_data_workers, collate_fn=self.collate)
        return dl

    def val_dataloader(self):
        assert "val" in self.datasets, "No ValidationSet build, run setup('fit')"

        sampler = SubsetRandomSampler(self.stats["val"].indices)
        dl = DataLoader(self.datasets["val"], batch_size=self.batch_size, num_workers=self.num_data_workers,
                        sampler=sampler, collate_fn=self.collate)
        return dl

    def test_dataloader(self):
        assert "test" in self.datasets, "No TestSet build, run setup('test')"

        # sampler = SubsetRandomSampler(self.indices["test"])
        dl = DataLoader(self.datasets["test"], batch_size=self.batch_size, num_workers=self.num_data_workers,
                        collate_fn=self.collate, shuffle=False)

        return dl

    @property
    def num_classes(self):
        return len(self.classes)

    @staticmethod
    def collate(batch):
        transposed_data = list(zip(*batch))
        x = torch.stack(transposed_data[0], 0)
        y = torch.stack(transposed_data[1], 0)
        info = list(transposed_data[2])
        return x, y, info
