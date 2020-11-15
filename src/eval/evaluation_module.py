import random
from typing import Optional, List
from matplotlib import pyplot as plt
from pathlib import Path
import torch
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay
from src.data import HarDataset, DataModule, MediaDir
from src.eval import OutDir, ClipPlot, PlotIterator


class EvaluationModule:

    def __init__(self, out_dir: str, data_module: DataModule, logger, img_format='eps'):
        self.dm = data_module
        self.logger = logger
        self.out_dir = OutDir(out_dir)

        assert img_format in ['svg', 'eps', 'png']
        self.file_format = img_format

    def plot_background_ratio(self, context='all', save=False, upload=False):
        assert context in ['train', 'val', 'test', 'all']

        fig, ax1 = plt.subplots()

        if context != 'all':
            bg_ratio = self.dm.stats[context].background_ratio
            pw_ratio = self.dm.stats[context].overlap_samples / len(self.dm.datasets[context])
            single_ratio = 1 - bg_ratio - pw_ratio
        else:
            bg_samples = sum([self.dm.stats[context].background_samples for context in ['train', 'val', 'test']])
            pw_samples = sum([self.dm.stats[context].overlap_samples for context in ['train', 'val', 'test']])
            total_samples = sum([len(self.dm.datasets[context]) for context in ['train', 'val', 'test']])
            bg_ratio = bg_samples / total_samples
            pw_ratio = pw_samples / total_samples
            single_ratio = 1 - bg_ratio - pw_ratio

        labels = 'Background', 'Single Action', 'Overlapping Actions'
        sizes = [bg_ratio, single_ratio, pw_ratio]

        ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.close()

        self._handle(fig, context, 'background_ratio', save, upload)

        return fig

    def plot_distribution(self, show='annotations', context='all', save=False, upload=False):
        assert context in ['train', 'val', 'test', 'all']

        assert show in ['annotations', 'used_samples']

        fig, ax = plt.subplots(dpi=120)

        if context != 'all':
            actions = self.dm.stats[context].actions
            samples = self.dm.stats[context].samples
            resamples = self.dm.stats[context].resamples.tolist() + [self.dm.stats[context].background_resamples]
        else:
            actions = torch.sum(torch.tensor([self.dm.stats[context].actions for context in ['train', 'val', 'test']]), dim=0)
            samples = torch.sum(torch.tensor([self.dm.stats[context].samples for context in ['train', 'val', 'test']]), dim=0)
            resamples = torch.sum(torch.stack([self.dm.stats[context].resamples + [self.dm.stats[context].background_resamples] for context in ['train', 'val', 'test']]), dim=0)

        plt.xticks(rotation=45, ha="right")
        ax.bar(self.dm.classes, samples, label='samples')
        if context in ['train', 'val', 'test'] and show == 'used_samples':
            ax.bar(self.dm.classes + ['background'], resamples, label='used samples')
        if show == 'annotations':
            ax.bar(self.dm.classes, actions, label='annotations')
        if context != 'all':
            ax.hlines(self.dm.limit_per_class[context], -0.5, 30.5)
        ax.set_title(f'{context} set' if context != 'all' else 'all data sets')
        ax.legend()
        ax.set_yscale('log')
        plt.close()

        self._handle(fig, context, f'class_distribution_{show}', save, upload)

        return fig

    def plot_pairwise_tuples(self, context='all', absolute=False, save=False, upload=False):
        assert context in ['train', 'val', 'test', 'all']

        fig, ax = plt.subplots(figsize=(20, 20))

        if context != 'all':
            occs = self.dm.stats[context].pairwise_occs
        else:
            occs = [self.dm.stats[context].pairwise_occs for context in ['train', 'val', 'test']]
            occs = torch.stack(occs)
            occs = torch.sum(occs, dim=0)

        if not absolute:
            # divide occurrences by total occurrences per class
            occs = torch.transpose(occs / torch.diag(occs) * 100, 0, 1)

        display = ConfusionMatrixDisplay(confusion_matrix=occs, display_labels=self.dm.classes)
        ax.title.set_text(f'pairwise overlaps in {context} set')

        display.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical', values_format='.2g')
        ax.set(ylabel="Samples",
               xlabel="Overlaps")

        self._handle(fig, context, 'pairwise_occurrences', save, upload)

        plt.close()
        return fig

    def get_sample_plot(self, row: Optional[int] = None, video: Optional[Path] = None, offset: Optional[int] = None,
                        pred: Optional[torch.Tensor] = None, context='train'):

        assert context in ['train', 'val', 'test', 'all']
        if context == 'all':
            context = ['train', 'val', 'test'][random.randint(0, 2)]
            print(f'context set to {context}')

        dataset = self.dm.datasets[context]
        indices = self.dm.stats[context].indices

        if row is None and video is not None and offset is not None:
            print('searching..')
            for idx, info in enumerate(dataset.info):
                if video.samefile(info['path']) and info['start'] == offset:
                    row = idx
                    print(f'row set to {row} by video={video} and offset={offset}')
                    break

        if row is None:
            row = indices[random.randint(0, len(indices) - 1)]
            print(f'row set to {row} by random choice')

        if row is not None:
            return ClipPlot(self.logger, dataset=dataset, context=context, row=row, pred=pred, save_dir=self.out_dir.sample())

    def get_sample_plots(self, label: Optional[str], indices: Optional[List[int]] = None,
                         pred: Optional[torch.Tensor] = None, context='train'):

        assert context in ['train', 'val', 'test', 'all']
        if context == 'all':
            context = ['train', 'val', 'test'][random.randint(0, 2)]
            print(f'context set to {context}')

        dataset = self.dm.datasets[context]
        stats = self.dm.stats[context]
        num_samples = len(dataset)

        if pred:
            assert indices and len(pred) == len(indices), 'length of `pred` has to match length of `indices`'

        if not indices:
            indices = stats.indices

        if label:
            label_idx = dataset.classes.index(label)
            if label != 'none':
                num_samples = int(stats.samples[label_idx])
                indices = torch.argsort(dataset.y[:, label_idx], descending=True)[0:num_samples]
            else:
                num_samples = int(stats.background_samples)
                indices = torch.argsort(torch.sum(dataset.y, dim=1))[0:num_samples]
            indices = indices[torch.randperm(num_samples)].tolist()

        return PlotIterator(dataset, self.logger, indices, pred, context, self.out_dir.sample())

    def _handle(self, fig, context: str, type: str, save: bool, upload: bool):
        now = datetime.now()
        filename = f'{self.out_dir.stats()}/{type}_{context}_{now.strftime("%Y%m-%d%H-%M%S")}.{self.file_format}'
        if save:
            fig.savefig(filename, format=self.file_format)
        if upload:
            with self.logger.experiment.context_manager(context):
                self.logger.experiment.log_asset(filename)
