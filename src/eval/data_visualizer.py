import glob
import os
from typing import Optional, List
from matplotlib import pyplot as plt
import torch
from pytorch_lightning.loggers import LightningLoggerBase
from sklearn.metrics import ConfusionMatrixDisplay
from src.data import DataModule
from src.eval.plots import ClipPlot, PlotIterator


class DataVisualizer:

    def __init__(self, dm: SoccarDataModule, logger: LightningLoggerBase, out_dir='Data', file_format='eps'):
        self.dm = dm
        self.dataset = dict(train=dm.train_dataset, val=dm.val_dataset, test=dm.test_cl_dataset)
        self.logger = logger
        self.out_dir = out_dir

        assert file_format in ['svg', 'eps', 'png']
        self.file_format = file_format

    def get_sample_plot(self, row: Optional[int], video: Optional[str], offset: Optional[int],
                        pred: Optional[torch.Tensor], context='train'):
        dataset = self.dataset[context]

        if not row:
            assert video is not None and offset is not None, 'either `row` or `video` and `offset` has to be specified'
            for idx, info in enumerate(dataset.info):
                if info['path'] == video and info['offset'] == offset:
                    row = idx
                    break

        if row:
            x, y, info = dataset[row]
            return ClipPlot(self.logger, x, y, {**info, 'context': context}, pred, dataset.labels, self.out_dir)

    def get_sample_plots(self, label: Optional[str], indices: Optional[List[int]] = None,
                         pred: Optional[torch.Tensor] = None, context='train'):
        dataset = self.dataset[context]
        num_samples = len(dataset)

        if pred:
            assert indices and len(pred) == len(indices), 'length of `pred` has to match length of `indices`'

        if not indices:
            indices = [i for i in range(num_samples)]

        if label:
            num_samples = dataset.stats[label]['samples']
            if label != 'none':
                label_idx = dataset.classes.index(label)
                indices = torch.argsort(dataset.y[:, label_idx], descending=True)[0:num_samples].tolist()
            else:
                indices = torch.argsort(torch.sum(dataset.y, dim=1))[0:num_samples].tolist()

        return PlotIterator(dataset, self.logger, indices, pred, context, self.out_dir)

    def plot_match(self, match_id=None):
        pass

    def plot_pairwise_tuples(self, save=False, context='train'):
        dataset = self.dataset[context]
        num_classes = len(dataset.classes)
        res = torch.zeros([num_classes, num_classes])
        filename = f'{self.out_dir}/{context}-pairwise-overlaps.{self.file_format}'

        for row in range(num_classes):
            for col in range(num_classes):
                filtered = dataset.y[:, [row, col]]
                summed = torch.sum(filtered, dim=1)
                relevant = summed > 1
                res[row, col] = torch.sum(relevant)

        # divide occurrences by total occurrences per class
        res = torch.transpose(res / torch.diag(res) * 100, 0, 1)

        fig = plt.figure(figsize=(20, 20))
        display = ConfusionMatrixDisplay(confusion_matrix=res, display_labels=dataset.classes)
        ax = fig.add_subplot()
        ax.title.set_text(f'pairwise overlaps in {context} set')
        display.plot(ax=ax, cmap=plt.cm.Blues)

        if save:
            fig.savefig(filename, format=self.file_format)
            with self.logger.experiment.context_manager(context):
                self.logger.experiment.log_asset(filename)

        plt.close()
        return res, fig, filename

    def plot_distribution(self, save=False, context='train'):
        dataset = self.dataset[context]

        actions = []
        samples = []
        for cls in dataset.classes:
            actions.append(dataset.stats[cls]['actions'])
            samples.append(dataset.stats[cls]['samples'])

        fig, ax = plt.subplots(figsize=(15, 10))
        plt.xticks(rotation=45, ha="right")
        ax.bar(dataset.classes, samples, label='samples')
        ax.bar(dataset.classes, actions, label='annotations')
        ax.hlines(self.dm.limit_per_class[context], -0.5, 30.5)
        ax.set_title(f'{context} set')
        ax.legend()
        ax.set_yscale('log')
        plt.close()

        if save:
            filename = f'{self.out_dir}/{context}-distribution.{self.file_format}'
            fig.savefig(filename, format=self.file_format)
            with self.logger.experiment.context_manager(context):
                self.logger.experiment.log_asset(filename)

        return fig

    @staticmethod
    def clean_up(self):
        for f in glob.glob(f"{self.out_dir}/*.mp4"):
            os.remove(f)
        for f in glob.glob(f"{self.out_dir}/*.svg"):
            os.remove(f)
        for f in glob.glob(f"{self.out_dir}/*.gif"):
            os.remove(f)
