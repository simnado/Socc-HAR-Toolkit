from pathlib import Path
from typing import Optional
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
import numpy as np
from src.data import DatabaseHandle, HarDataset, PreProcessing, DataStats
from src.data.util import DatabaseFetcher, MediaDir


class DataModule(LightningDataModule):

    def __init__(self, database: str, data_dir: str, num_frames: int, res: int, fps: int, metadata_path: Optional[str],
                 batch_size=32, mean=None, std=None,
                 classes=None, max_train_samples_per_class=500,
                 num_data_workers=None, seed=94):
        super().__init__()
        np.random.seed(seed)
        self.seed = seed
        self.media_dir = MediaDir(data_dir)

        database_path = self.media_dir.database(database)
        DatabaseFetcher.load(database, database_path)

        assert database_path.exists(), 'Database does not exist'

        self.database = DatabaseHandle(database_path)

        self.max_train_samples_per_class = max_train_samples_per_class
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.res = res
        self.fps = fps
        self.mean = mean
        self.std = std
        self.classes = classes

        if self.classes is None:
            self.classes = self.database.classes

        self.datasets = dict()
        self.stats = dict()
        self.indices = dict()
        self.limit_per_class = dict(train=self.max_train_samples_per_class, val=50, test=100)

        self.test_loc_dataset = None

        self.num_data_workers = num_data_workers
        if num_data_workers is None:
            self.num_data_workers = max(4, torch.get_num_interop_threads() * 2)

        self.metadata_out_path = None
        self.precomputed_metadata_file = metadata_path

        self.video_metadata = None
        self.pre_processor = None

    def prepare_data(self):
        self.pre_processor = PreProcessing(self.database, self.media_dir.root, Path(self.precomputed_metadata_file),
                                           res=360)
        self.video_metadata = self.pre_processor.prepare_data()

    def setup(self, stage: Optional[str] = None):

        assert self.video_metadata, "No video metadata found, run prepare_data() first"

        assert stage in ['fit', 'test']

        # split dataset
        if stage == 'fit':
            self.datasets['train'] = HarDataset(database=self.database,
                                                video_metadata=self.video_metadata['train'],
                                                res=self.res, classes=self.classes,
                                                normalized=True, num_frames=self.num_frames, fps=self.fps,
                                                mean=self.mean, std=self.std, do_augmentation=True, seed=self.seed)
            self.stats['train'] = DataStats('train', self.datasets['train'], self.limit_per_class['train'])

            self.mean = self.datasets['train'].mean
            self.std = self.datasets['train'].std

            self.datasets['val'] = HarDataset(database=self.database,
                                              video_metadata=self.video_metadata['val'],
                                              res=self.res, classes=self.classes,
                                              normalized=True, num_frames=self.num_frames, fps=self.fps,
                                              mean=self.mean, std=self.std, seed=self.seed)
            self.stats['val'] = DataStats('val', self.datasets['val'], self.limit_per_class['val'])

        if stage == 'test':
            self.datasets['test'] = HarDataset(database=self.database,
                                               video_metadata=self.video_metadata['test'],
                                               res=self.res, classes=self.classes,
                                               normalized=True, num_frames=self.num_frames, fps=self.fps,
                                               mean=self.mean, std=self.std,
                                               seed=self.seed)
            self.stats['test'] = DataStats('test', self.datasets['test'], self.limit_per_class['test'])


            self.test_loc_dataset = HarDataset(database=self.database,
                                               video_metadata=self.video_metadata['test'],
                                               res=self.res, classes=self.classes,
                                               normalized=True, num_frames=self.num_frames, fps=self.fps,
                                               mean=self.mean, std=self.std,
                                               allow_critical=True, seed=self.seed)

    @property
    def train_dataloader(self):
        assert "train" in self.datasets, "No TrainingSet build, run setup('fit')"

        limit = self.limit_per_class['train']
        dataset = self.datasets["train"]
        stats = self.stats['train']
        num_samples = sum([limit] + [min(limit, stats.samples) for _ in self.classes])
        print(f'sample {num_samples}/{len(dataset)} clips')
        sampler = WeightedRandomSampler(stats.weights, num_samples) # should be different each iteration
        dl = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                        num_workers=self.num_data_workers)
        return dl

    @property
    def val_dataloader(self):
        assert "val" in self.datasets, "No ValidationSet build, run setup('fit')"
        sampler = SubsetRandomSampler(self.indices["val"])
        dl = DataLoader(self.datasets["val"], batch_size=self.batch_size, num_workers=self.num_data_workers,
                        sampler=sampler)
        return dl

    @property
    def test_dataloader(self):
        assert "test" in self.datasets, "No TestSet build, run setup('test')"
        sampler = SubsetRandomSampler(self.indices["test"])
        dl = DataLoader(self.datasets["test"], batch_size=self.batch_size, num_workers=self.num_data_workers,
                        sampler=sampler)
        return dl

    @property
    def test_localization_dataloader(self):
        mnist_test = DataLoader(self.test_loc_dataset, batch_size=self.batch_size, num_workers=self.num_data_workers)
        return mnist_test

    @property
    def num_classes(self):
        return len(self.classes)
