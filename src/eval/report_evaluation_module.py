from typing import Optional, List
from matplotlib import pyplot as plt
from pathlib import Path
import torch
import pandas as pd
from src.data import HarDataset, DataModule, MediaDir
from src.eval import OutDir, ClipPlot, PlotIterator, EvaluationModule


class ReportEvaluationModule(EvaluationModule):

    def __init__(self, out_dir: str, data_module: DataModule, report: pd.DataFrame, logger, img_format='eps'):
        super().__init__(out_dir, data_module, logger, img_format)
        self.report = report

    def get_sample_plot_by_report(self, context='train', epoch=None):
        if epoch is None:
            epoch = self.num_epochs - 1

        df = self.report
        sample = df[(df.subset == context) & (df.epoch == epoch)].sample()

        # todo: context=all
        paths = [path for path in self.dm.datasets[context].video_metadata['video_paths'] if sample.video in path]

        preds = np.fromstring(sample.scores.tolist()[0][1:-1], dtype=float, sep=' ')
        preds = torch.from_numpy(preds)
        return self.get_sample_plot(video=Path(paths[0]), offset=sample.start, pred=preds)

    def get_top_loss_plots(self, context='train', epoch=None, limit=50):
        # todo:
        return self.get_sample_plots(indices=[], pred=None, context=context)

    def integrity_check(self) -> bool:
        df = self.report

        # val is deterministic
        for index, row in df[df.subset == 'val'].sample(n=10).iterrows():
            assert len(df[(df.key == row.key) & (df.start == row.start)]) == self.num_epochs

        # test is deterministic
        if self.num_test_runs > 0:
            for index, row in df[df.subset == 'test'].sample(n=10).iterrows():
                assert len(df[(df.key == row.key) & (df.start == row.start)]) == self.num_test_runs

        # train is not deterministic
        occs = [len(df[(df.key == row.key) & (df.start == row.start)]) for index, row in df[df.subset == 'test'].sample(n=10).iterrows()]
        occs = [occ == occs[0] for occ in occs]
        assert False in occs

        return True

    def _label_occurances(self, split: str, epoch: int, label: str):
        df = self.report
        return len(df[(df.subset == split) & (df.epoch == epoch) & (df.labels.str.contains(label, na=False))])

    def _background_occurances(self, split: str, epoch: int):
        df = self.report
        return len(df[(df.subset == split) & (df.epoch == epoch) & (df.labels.str.contains('nan', na=True))])

    def train_samples_boxplot(self, save=True, upload=True):
        fig, ax = plt.subplots(dpi=120)

        plt.xticks(rotation=90, ha="right")

        ax.set_title(f'train samples per epoch')

        occs = [[self._label_occurances('train', epoch, label) for epoch in range(self.num_epochs)] for label in self.dm.classes]
        occs = occs + [self._background_occurances('train', epoch) for epoch in range(self.num_epochs)]

        ax.boxplot(occs, vert=True, patch_artist=True, labels=self.dm.classes)

        plt.close()

        self._handle(fig, 'train', f'samples', save, upload)

        return fig

    @property
    def num_epochs(self):
        return self.report.epoch.nunique()

    @property
    def num_test_runs(self):
        return self.report[self.report.subset == 'test'].epoch.nunique()
