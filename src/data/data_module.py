from pathlib import Path
from typing import Optional
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
import numpy as np
from src.data import DatabaseHandle, HarDataset, PreProcessing
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

        self.train_dataset = None
        self.val_dataset = None
        self.test_cl_dataset = None
        self.test_loc_dataset = None

        self.num_data_workers = num_data_workers
        if num_data_workers is None:
            self.num_data_workers = max(4, torch.get_num_interop_threads() * 2)

        self.metadata_out_path = None
        self.precomputed_metadata_file = metadata_path

        self.video_metadata = None

    def prepare_data(self):
        pre_processor = PreProcessing(self.database, self.media_dir.root, Path(self.precomputed_metadata_file), res=360)
        self.video_metadata = pre_processor.prepare_data()

    def setup(self, stage: Optional[str] = None):

        assert self.video_metadata, "No video metadata found, run prepare_data() first"

        # split dataset
        if stage == 'fit':
            self.train_dataset = HarDataset(database=self.database,
                                                       video_metadata=self.video_metadata['train'],
                                                       res=self.res, classes=self.classes,
                                                       normalized=True, num_frames=self.num_frames, fps=self.fps,
                                                       mean=self.mean, std=self.std, do_augmentation=True, seed=self.seed)
            self.mean = self.train_dataset.mean
            self.std = self.train_dataset.std
            self.val_dataset = HarDataset(database=self.database,
                                                     video_metadata=self.video_metadata['val'],
                                                     res=self.res, classes=self.classes,
                                                     normalized=True, num_frames=self.num_frames, fps=self.fps,
                                                     mean=self.mean, std=self.std, seed=self.seed)
        if stage == 'test':
            self.test_cl_dataset = HarDataset(database=self.database,
                                                         video_metadata=self.video_metadata['test'],
                                                         res=self.res, classes=self.classes,
                                                         normalized=True, num_frames=self.num_frames, fps=self.fps,
                                                         mean=self.mean, std=self.std,
                                                         allow_critical=False, seed=self.seed)

            self.test_loc_dataset = HarDataset(database=self.database,
                                                          video_metadata=self.video_metadata['test'],
                                                          res=self.res, classes=self.classes,
                                                          normalized=True, num_frames=self.num_frames, fps=self.fps,
                                                          mean=self.mean, std=self.std,
                                                          allow_critical=True, seed=self.seed)

    @property
    def train_dataloader(self):
        limit = self.limit_per_class['train']
        num_samples = sum([limit] + [min(limit, self.train_dataset.stats[cls]['samples']) for cls in self.classes])
        print(f'sample {num_samples}/{len(self.train_dataset)} clips')
        #  sampler = WeightedRandomSampler(self.train_dataset.w, num_samples)
        indices = torch.multinomial(torch.tensor(self.train_dataset.w), num_samples)
        sampler = SubsetRandomSampler(indices)
        dl = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler,
                        num_workers=self.num_data_workers)
        return dl

    @property
    def val_dataloader(self):
        limit = self.limit_per_class['val']
        num_samples = sum([limit] + [min(limit, self.val_dataset.stats[cls]['samples']) for cls in self.classes])
        print(f'sample {num_samples}/{len(self.val_dataset)} clips')
        indices = torch.multinomial(torch.tensor(self.val_dataset.w), num_samples)
        sampler = SubsetRandomSampler(indices)

        dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_data_workers,
                        sampler=sampler)
        return dl

    @property
    def test_classification_dataloader(self):
        limit = self.limit_per_class['test']
        num_samples = sum([limit] + [min(limit, self.test_cl_dataset.stats[cls]['samples']) for cls in self.classes])
        print(f'sample {num_samples}/{len(self.test_cl_dataset)} clips')
        indices = torch.multinomial(torch.tensor(self.test_cl_dataset.w), num_samples)
        sampler = SubsetRandomSampler(indices)

        mnist_test = DataLoader(self.test_cl_dataset, batch_size=self.batch_size, num_workers=self.num_data_workers,
                                sampler=sampler)
        return mnist_test

    @property
    def test_localization_dataloader(self):
        mnist_test = DataLoader(self.test_loc_dataset, batch_size=self.batch_size, num_workers=self.num_data_workers)
        return mnist_test

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def limit_per_class(self):
        return dict(train=self.max_train_samples_per_class, val=50, test=100)
