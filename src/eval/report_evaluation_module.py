from typing import Optional
from matplotlib import pyplot as plt
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from src.data import HarDataset, DataModule, MediaDir
from src.eval import OutDir, ClipPlot, PlotIterator, EvaluationModule, MultiLabelStatScores, MultiLabelStatCurves


class ReportEvaluationModule(EvaluationModule):

    def __init__(self, out_dir: str, data_module: DataModule, report: pd.DataFrame, logger, img_format='eps'):
        super().__init__(out_dir, data_module, logger, img_format)
        self.report = report
        self.train_scalars: [MultiLabelStatScores] = [None for _ in range(self.num_epochs)]
        self.val_scalars: [MultiLabelStatScores] = [None for _ in range(self.num_epochs)]
        self.test_scalars: [MultiLabelStatScores] = [None for _ in range(100)]

        self.train_curve: Optional[MultiLabelStatCurves] = None
        self.val_curve: Optional[MultiLabelStatCurves] = None
        self.test_curve: Optional[MultiLabelStatCurves] = None

    def get_sample_plot_by_report(self, context='train', epoch=None):
        if epoch is None:
            epoch = self.num_epochs - 1

        df = self.report
        sample = df[(df.subset == context) & (df.epoch == epoch)].sample()

        # todo: context=all
        paths = [path for path in self.dm.datasets[context].video_metadata['video_paths'] if sample.video.item() in path]

        preds = np.fromstring(sample.scores.item()[1:-1], dtype=float, sep=', ')
        preds = torch.from_numpy(preds)
        return self.get_sample_plot(video=Path(paths[0]), offset=sample.start.item(), pred=preds)

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
        occs = []
        for index, row in df[df.subset == 'train'].sample(n=10).iterrows():
            occ = len(df[(df.key == row.key) & (df.start == row.start)])
            assert occ <= self.num_epochs
            occs.append(occ)

        # occurances should be different
        occs = [occ == occs[0] for occ in occs]
        assert False in occs

        return True

    def _label_occurances(self, split: str, epoch: int, label: str):
        df = self.report
        return len(df[(df.subset == split) & (df.epoch == epoch) & (df.labels.str.contains(label, na=False))])

    def _background_occurances(self, split: str, epoch: int):
        df = self.report
        return len(df[(df.subset == split) & (df.epoch == epoch) & (df.labels.str.contains('nan', na=True))])

    def _get_y(self, split: str, epoch: int):
        df = self.report
        df = df[(df.subset == split) & (df.epoch == epoch)]
        y = df.y.tolist()
        y = [np.fromstring(score[1:-1], sep=', ') for score in y]
        y = torch.Tensor(y)
        return y

    def _get_scores(self, split: str, epoch: int):
        df = self.report
        df = df[(df.subset == split) & (df.epoch == epoch)]
        out = df.scores.tolist()
        out = [np.fromstring(score[1:-1], sep=', ') for score in out]
        out = torch.Tensor(out)
        return out

    def train_samples_boxplot(self, save=True, upload=True):
        fig, ax = plt.subplots(dpi=120)

        plt.xticks(rotation=90)

        ax.set_title(f'train samples per epoch')

        occs = [[self._label_occurances('train', epoch, label) for epoch in range(self.num_epochs)] for label in self.dm.classes]
        occs = occs + [[self._background_occurances('train', epoch) for epoch in range(self.num_epochs)]]

        ax.hlines(self.dm.limit_per_class['train'], -0.5, 33.5, color='grey')
        ax.boxplot(occs, vert=True, patch_artist=True, labels=self.dm.classes + ['background'])

        plt.tight_layout()
        plt.close()

        self._handle(fig, 'train', f'samples', save, upload)

        return fig

    def _init_train_scalars(self, epoch: int):
        if self.train_scalars[epoch] is None:
            scalars = MultiLabelStatScores(self.dm.num_classes, threshold=0.5)
            scalars(self._get_scores('train', epoch), self._get_y('train', epoch))
            self.train_scalars[epoch] = scalars

            scalars = MultiLabelStatScores(self.dm.num_classes, threshold=0.5)
            scalars(self._get_scores('val', epoch), self._get_y('val', epoch))
            self.val_scalars[epoch] = scalars

    def _init_test_scalars(self, threshold: int):
        assert -1 < threshold < 101
        if self.test_scalars[threshold] is None:
            scalars = MultiLabelStatScores(self.dm.num_classes, threshold=threshold / 100.0)
            scalars(self._get_scores('test', self.num_epochs - 1), self._get_y('test', self.num_epochs - 1))
            self.test_scalars[threshold] = scalars

    def _init_train_curve(self):
        if self.train_curve is None:
            curve = MultiLabelStatCurves(self.dm.num_classes)
            curve(self._get_scores('train', self.num_epochs - 1), self._get_y('train', self.num_epochs - 1))
            self.train_curve = curve

    def _init_val_curve(self):
        if self.val_curve is None:
            curve = MultiLabelStatCurves(self.dm.num_classes)
            curve(self._get_scores('val', self.num_epochs - 1), self._get_y('val', self.num_epochs - 1))
            self.val_curve = curve

    def _init_test_curve(self):
        if self.test_curve is None:
            curve = MultiLabelStatCurves(self.dm.num_classes)
            curve(self._get_scores('test', self.num_epochs - 1), self._get_y('test', self.num_epochs - 1))
            self.test_curve = curve

    def get_metric_by_epoch(self, metric: str, reduction: str, save=True, upload=False):
        fig, ax = plt.subplots(dpi=120)
        ax.set_title(f'{reduction} {metric}')

        train = []
        val = []
        for epoch in range(self.num_epochs):
            self._init_train_scalars(epoch)
            train.append(getattr(self.train_scalars[epoch], metric)(class_reduction=reduction))
            val.append(getattr(self.val_scalars[epoch], metric)(class_reduction=reduction))

        ax.plot(train, label='train data')
        ax.plot(val, label='val data')
        ax.legend()

        plt.tight_layout()
        plt.close()
        self._handle(fig, 'train', f'{reduction} {metric} while training', save, upload)
        return fig

    def get_curve(self, split: str, metric: str, reductions: [str], classes: [str], save=True, upload=False):
        curve: Optional[MultiLabelStatCurves] = None

        if split == 'train':
            self._init_train_curve()
            curve = self.train_curve
        elif split == 'val':
            self._init_val_curve()
            curve = self.val_curve
        elif split == 'test':
            self._init_test_curve()
            curve = self.test_curve

        fpr, tpr, thresholds, peak_idx = getattr(curve, metric)(reductions, [self.dm.classes.index(cls) for cls in classes])

        fig, ax = plt.subplots(dpi=120)
        ax.set_title(f'{metric}')
        ax.set_xlabel('false-positive rate (fpr)')
        ax.set_ylabel('true-positive rate (tpr)')

        ax.plot([0,1], [0,1], linestyle='--')

        labels = reductions + classes

        for line in range(len(fpr)):
            color = next(ax._get_lines.prop_cycler)['color']
            peak = peak_idx[line]
            ax.plot(fpr[line], tpr[line], color=color, label=f'{labels[line]}: th = %0.2f' % thresholds[line][peak])
            ax.plot(fpr[line][peak], tpr[line][peak], color=color, marker='o')

        ax.legend()

        plt.tight_layout()
        plt.close()
        self._handle(fig, 'train', f'samples', save, upload)
        return fig

    def get_scalars(self, save=True, upload=False):

        splits = ['train', 'val', 'test']
        if self.num_test_runs == 0:
            splits.pop()

        scalars: Optional[MultiLabelStatScores] = None
        curve: Optional[MultiLabelStatCurves] = None
        fig, ax = plt.subplots(dpi=120)
        ax.set_title(f'scalar metrics by subset')

        labels = []
        width = 0.35  # the width of the bars

        x = np.arange(18)  # the label locations

        for idx, split in enumerate(splits):
            if split == 'train':
                self._init_train_curve()
                self._init_train_scalars(self.num_epochs - 1)
                curve = self.train_curve
                scalars = self.train_scalars[self.num_epochs - 1]
            elif split == 'val':
                self._init_val_curve()
                self._init_train_scalars(self.num_epochs - 1)
                curve = self.val_curve
                scalars = self.val_scalars[self.num_epochs - 1]
            elif split == 'test':
                self._init_test_curve()
                self._init_test_scalars(50)
                curve = self.test_curve
                scalars = self.test_scalars[50]
            metrics = {
                f'{split}_balanced_accuracy_micro': scalars.balanced_accuracy('micro'),
                f'{split}_balanced_accuracy_macro': scalars.balanced_accuracy('macro'),
                f'{split}_balanced_accuracy_weighted': scalars.balanced_accuracy('weighted'),
                f'{split}_accuracy_micro': scalars.balanced_accuracy('micro'),
                f'{split}_accuracy_macro': scalars.balanced_accuracy('macro'),
                f'{split}_accuracy_weighted': scalars.balanced_accuracy('weighted'),
                f'{split}_f1_micro': scalars.f1('micro'),
                f'{split}_f1_macro': scalars.f1('macro'),
                f'{split}_f1_weighted': scalars.f1('weighted'),
                f'{split}_precision_micro': scalars.precision('micro'),
                f'{split}_precision_macro': scalars.precision('macro'),
                f'{split}_precision_weighted': scalars.precision('weighted'),
                f'{split}_recall_micro': scalars.recall('micro'),
                f'{split}_recall_macro': scalars.recall('macro'),
                f'{split}_recall_weighted': scalars.recall('weighted'),
                f'{split}_auroc_micro': curve.auroc('micro'),
                f'{split}_auroc_macro': curve.auroc('macro'),
                f'{split}_auroc_weighted': curve.auroc('macro'),
            }

            labels = [key[len(split)+1:] for key, value in metrics.items()]
            values = [value for key, value in metrics.items()]
            ax.bar(x + (idx - 1.5) * width / 3, values, width / 3, label=split)

            if upload:
                self.logger.log_metrics(metrics)

        plt.xticks(x, rotation=90)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.tight_layout()
        plt.close()
        self._handle(fig, 'train', f'samples', save, upload)
        return fig

    def get_scalar_by_class(self, split: str, metric: str, save=True, upload=False):
        scalars: Optional[MultiLabelStatScores] = None
        curve: Optional[MultiLabelStatCurves] = None
        fig, ax = plt.subplots(dpi=120)
        title = f'{metric} by class'
        ax.set_title(title)

        if split == 'train':
            self._init_train_curve()
            self._init_train_scalars(self.num_epochs - 1)
            curve = self.train_curve
            scalars = self.train_scalars[self.num_epochs - 1]
        elif split == 'val':
            self._init_val_curve()
            self._init_train_scalars(self.num_epochs - 1)
            curve = self.val_curve
            scalars = self.val_scalars[self.num_epochs - 1]
        elif split == 'test':
            self._init_test_curve()
            self._init_test_scalars(50)
            curve = self.test_curve
            scalars = self.test_scalars[50]

        metric = getattr(scalars, metric) if metric != 'auroc' else curve.auroc
        scalars = metric('none')
        order = torch.argsort(torch.Tensor(scalars)).tolist()
        metrics = dict()

        for idx in order:
            cls = self.dm.classes[idx]
            metrics[f'{split}_{metric}_{cls}'] = scalars[idx]

        values = [value for key, value in metrics.items()]
        classes = [self.dm.classes[idx] for idx in order]
        ax.bar(classes, values)

        if upload:
            self.logger.log_metrics(metrics)

        plt.xticks(classes, rotation=90)
        ax.set_xticklabels(classes)

        plt.tight_layout()
        plt.close()
        self._handle(fig, 'train', title, save, upload)
        return fig

    def cluster_classes_by_metrics(self, split: str, save=True, upload=False):
        pass

    def get_threshold_by_metric(self, split: str, metric: str, reduction: str, save=True, upload=False):
        fig, ax = plt.subplots(dpi=120)
        ax.set_title(f'threshold by {reduction} {metric}')
        ax.set_xlabel('threshold')
        ax.set_ylabel(f'score')

        for i in range(100):
            self._init_test_scalars(i)
        # todo: support classes like in get_curve
        x = torch.linspace(0.01, 1.0, 100)
        metrics = [getattr(scalar, metric)(reduction) for scalar in self.test_scalars]
        color = next(ax._get_lines.prop_cycler)['color']
        peak = torch.argmax(torch.Tensor(metrics)).item()
        ax.plot(x, metrics, color=color)
        ax.plot(x[peak], metrics[peak], color=color, marker='o')

        if upload:
            # todo: move to _handle()
            self.logger.log_metrics({f'threshold_by_{metric}_{reduction}': x[peak]})

        plt.tight_layout()
        plt.close()
        self._handle(fig, split, f'threshold by {reduction} {metric}', save, upload)
        return fig

    @property
    def num_epochs(self):
        return self.report.epoch.nunique()

    @property
    def num_test_runs(self):
        return self.report[self.report.subset == 'test'].epoch.nunique()
