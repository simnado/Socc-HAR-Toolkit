from typing import Optional
from matplotlib import pyplot as plt
import torch
import pandas as pd
import numpy as np

from ..data.data_module import DataModule
from .evaluation_module import EvaluationModule
from .metrics.stat_scores_multiple_labels import MultiLabelStatScores
from .metrics.stat_curves_multiple_labels import MultiLabelStatCurves


class ReportEvaluationModule(EvaluationModule):

    def __init__(self, out_dir: str, data_module: DataModule, report: pd.DataFrame, logger, img_format=['png'],
                 consensus='max'):
        super().__init__(out_dir, data_module, logger, img_format)

        assert consensus in ['max', 'avg']

        self.report = report
        self.consensus = consensus
        self.train_scalars: [MultiLabelStatScores] = [None for _ in range(self.num_epochs)]
        self.val_scalars: [MultiLabelStatScores] = [None for _ in range(self.num_epochs)]
        self.test_scalars: [MultiLabelStatScores] = [None for _ in range(100)]

        self.train_curve: Optional[MultiLabelStatCurves] = None
        self.val_curve: Optional[MultiLabelStatCurves] = None
        self.test_curve: Optional[MultiLabelStatCurves] = None

    def get_top_loss_samples(self, context='train', epoch=None, label=None, limit=500, desc=True):
        if epoch is None:
            epoch = self.num_epochs - 1

        if context == 'test':
            df = self.test_df
            epoch = self.last_test_epoch
        else:
            df = self.report

        df = df[(df.subset == context) & (df.epoch == epoch)]

        if label and label != 'background':
            df = df[((df.labels.str.contains(f'{label},', na=False)) | (df.labels.str.contains(f'{label}$', na=False)))]
        elif label and label == 'background':
            df = df[df.labels.str.contains('nan', na=True)]

        df = df.sort_values(by=['loss'], ascending=not desc).head(limit)

        rows = []
        pred = []

        for i in range(len(df.index)):
            sample = df.iloc[i]
            try:
                dataset_idx = self.dm.datasets[context].get_row(sample.key, sample.start)
                rows.append(dataset_idx)
            except KeyError as e:
                print(f'Error: {str(e)} not present in {context} dataset')
                continue

            preds = np.fromstring(sample.scores[1:-1], dtype=float, sep=', ')
            preds = torch.from_numpy(preds)
            pred.append(preds)

        rows = torch.IntTensor(rows)
        pred = torch.stack(pred, dim=0)
        return (rows, pred)

    def integrity_check(self) -> bool:
        df = self.report

        # val is deterministic
        for index, row in df[df.subset == 'val'].sample(n=10).iterrows():
            num_samples = len(df[(df.key == row.key) & (df.start == row.start)])
            assert num_samples == self.num_epochs, f'only got {num_samples} val samples for row {row.key}@{row.start}'

        # test is deterministic
        if self.num_test_runs > 0:
            for index, row in self.test_df.sample(n=10).iterrows():
                assert len(self.test_df[(self.test_df.key == row.key) & (self.test_df.start == row.start)]) == 1
                # no test duplicates
                assert len(self.test_df[(self.test_df.key == row.key) & (self.test_df.start == row.start) & (
                            self.test_df.epoch == row.epoch)]) == 1

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
        df = df[(df.subset == split) & (df.epoch == epoch)]
        df = df[((df.labels.str.contains(f'{label},', na=False)) | (df.labels.str.contains(f'{label}$', na=False)))]
        return len(df.index)

    def _background_occurances(self, split: str, epoch: int):
        df = self.report
        return len(df[(df.subset == split) & (df.epoch == epoch) & (df.labels.str.contains('nan', na=True))])

    def _get_y(self, split: str, epoch: int):
        if split == 'test':
            df = self.test_df
        else:
            df = self.report
            df = df[(df.subset == split) & (df.epoch == epoch)]
        y = df.y.tolist()
        y = [np.fromstring(score[1:-1], sep=', ') for score in y]
        y = torch.Tensor(y)
        return y

    def _get_scores(self, split: str, epoch: int):
        if split == 'test':
            df = self.test_df
        else:
            df = self.report
            df = df[(df.subset == split) & (df.epoch == epoch)]
        out = df.scores.tolist()
        out = [np.fromstring(score[1:-1], sep=', ') for score in out]
        out = torch.Tensor(out)
        return out

    def train_samples_boxplot(self, save=True, upload=True):
        fig, ax = plt.subplots(figsize=(5, 10))

        ax.set_title(f'train samples per epoch')

        occs = [[self._label_occurances('train', epoch, label) for epoch in range(self.num_epochs)] for label in
                self.dm.classes]
        occs = occs + [[self._background_occurances('train', epoch) for epoch in range(self.num_epochs)]]

        ax.vlines(self.dm.limit_per_class['train'], 0.5, 33.5, color='grey')
        ax.boxplot(occs, vert=False, labels=self.dm.classes + ['background'])

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
            scalars(self._get_scores('test', self.last_test_epoch), self._get_y('test', self.last_test_epoch))
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
            curve(self._get_scores('test', self.last_test_epoch), self._get_y('test', self.last_test_epoch))
            self.test_curve = curve

    def get_metric_by_epoch(self, metric: str, reduction: str, save=True, upload=False):
        assert reduction in ['micro', 'macro', 'weighted']

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
        if upload:
            self.logger.log_metrics({
                f'train_{metric}_{reduction}': train[-1],
                f'val_{metric}_{reduction}': val[-1]
            })

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

        curves = getattr(curve, metric)(['micro', 'macro'], [i for i in range(len(self.dm.classes))])
        fig, ax = plt.subplots(dpi=120)
        ax.set_title(f'{metric}')
        ax.set_xlabel('false-positive rate (fpr)')
        ax.set_ylabel('true-positive rate (tpr)')

        ax.plot([0, 1], [0, 1], linestyle='--')

        labels = reductions + self.dm.classes
        keys = reductions + [i for i in range(self.dm.num_classes)]

        metrics = {}
        for idx, key in enumerate(keys):
            fpr, tpr, thresholds, peak = curves[key]
            label = labels[idx]
            threshold = thresholds[peak]
            metrics[f'{split}_{metric}_threshold_{label}'] = threshold
            if label in reductions or label in classes:
                color = next(ax._get_lines.prop_cycler)['color']
                linestyle = '-' if label in classes else ':'
                label = (f'{label}: θ=%0.2f' % threshold) if label in classes else label
                ax.plot(fpr, tpr, color=color, label=label, linestyle=linestyle)
                ax.plot(fpr[peak], tpr[peak], color=color, marker='o')

        ax.legend()

        plt.tight_layout()
        plt.close()

        print(metrics)

        if upload == True:
            self.logger.log_metrics(metrics)
        self._handle(fig, 'train', f'samples', save, upload)
        return fig

    def get_scalars(self, save=True, upload=False):

        splits = ['train', 'val', 'test']
        if self.num_test_runs == 0:
            splits.pop()

        scalars: Optional[MultiLabelStatScores] = None
        curve: Optional[MultiLabelStatCurves] = None
        fig, ax = plt.subplots(dpi=120)

        labels = []
        values = dict()

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
                f'{split}_accuracy_micro': scalars.accuracy('micro'),
                f'{split}_accuracy_macro': scalars.accuracy('macro'),
                f'{split}_accuracy_weighted': scalars.accuracy('weighted'),
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

            labels = [key[len(split) + 1:] for key, value in metrics.items()]
            values[split] = [float(value) for key, value in metrics.items()]

            if upload:
                self.logger.log_metrics(metrics)

        df = pd.DataFrame({'train': values['train'],
                           'val': values['val'],
                           'test': values['test']
                           }, index=labels)
        df.plot.barh(stacked=False, title='scalar metrics by subset', ax=ax)

        plt.tight_layout()
        plt.close()
        self._handle(fig, 'train', f'scalars per subset', save, upload)
        return fig

    def get_scalar_by_class(self, split: str, metric: str, save=True, upload=False):
        scalars: Optional[MultiLabelStatScores] = None
        curve: Optional[MultiLabelStatCurves] = None
        fig, ax = plt.subplots(dpi=120)
        title = f'{metric} by class'

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

        metric_fn = getattr(scalars, metric) if metric != 'auroc' else curve.auroc
        scalars = metric_fn('none')
        order = torch.argsort(torch.Tensor(scalars), descending=False).tolist()
        metrics = dict()

        for idx in order:
            cls = self.dm.classes[idx]
            metrics[f'{split}_{metric}_{cls}'] = scalars[idx]

        values = [float(value) for key, value in metrics.items()]
        classes = [self.dm.classes[idx] for idx in order]
        df = pd.DataFrame({split: values}, index=classes)
        df.plot.barh(stacked=False, title=title, ax=ax, legend=False)

        if upload:
            self.logger.log_metrics(metrics)

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
            self.logger.log_metrics({f'threshold_by_{metric}_{reduction}': x[peak]})

        plt.tight_layout()
        plt.close()
        self._handle(fig, split, f'threshold by {reduction} {metric}', save, upload)
        return fig

    def get_metrics_by_consensus(self, save=True, upload=False):
        fig, ax = plt.subplots(dpi=120)

        stored_consensus = self.consensus
        metrics = dict()

        self.consensus = 'max'
        curve = MultiLabelStatCurves(self.dm.num_classes)
        curve(self._get_scores('test', self.last_test_epoch), self._get_y('test', self.last_test_epoch))
        scalars = MultiLabelStatScores(self.dm.num_classes, threshold=50 / 100.0)
        scalars(self._get_scores('test', self.last_test_epoch), self._get_y('test', self.last_test_epoch))
        metrics['max'] = {
            f'test_balanced_accuracy_macro_{self.consensus}_consensus': float(scalars.balanced_accuracy('macro')),
            f'test_accuracy_macro_{self.consensus}_consensus': float(scalars.accuracy('macro')),
            f'test_f1_macro_{self.consensus}_consensus': float(scalars.f1('macro')),
            f'test_precision_macro_{self.consensus}_consensus': float(scalars.precision('macro')),
            f'test_recall_macro_{self.consensus}_consensus': float(scalars.recall('macro')),
            f'test_auroc_macro_{self.consensus}_consensus': curve.auroc('macro'),
        }

        self.consensus = 'avg'
        curve = MultiLabelStatCurves(self.dm.num_classes)
        curve(self._get_scores('test', self.last_test_epoch), self._get_y('test', self.last_test_epoch))
        scalars = MultiLabelStatScores(self.dm.num_classes, threshold=50 / 100.0)
        scalars(self._get_scores('test', self.last_test_epoch), self._get_y('test', self.last_test_epoch))
        metrics['avg'] = {
            f'test_balanced_accuracy_macro_{self.consensus}_consensus': float(scalars.balanced_accuracy('macro')),
            f'test_accuracy_macro_{self.consensus}_consensus': float(scalars.accuracy('macro')),
            f'test_f1_macro_{self.consensus}_consensus': float(scalars.f1('macro')),
            f'test_precision_macro_{self.consensus}_consensus': float(scalars.precision('macro')),
            f'test_recall_macro_{self.consensus}_consensus': float(scalars.recall('macro')),
            f'test_auroc_macro_{self.consensus}_consensus': curve.auroc('macro'),
        }

        self.consensus = stored_consensus

        max = [v for k, v in metrics['max'].items()]
        avg = [v for k, v in metrics['avg'].items()]

        index = ['balanced accuracy', 'accuracy', 'f1',
                 'precision', 'recall', 'auroc']
        df = pd.DataFrame({'max': max,
                           'avg': avg}, index=index)
        df.plot.barh(stacked=False, title='consensus functions', ax=ax, ylabel='macro')

        if upload:
            self.logger.log_metrics(metrics['max'])
            self.logger.log_metrics(metrics['avg'])

        self._handle(fig, 'test', f'metrics by consensus', save, upload)

    @property
    def num_epochs(self):
        df = self.report
        return df[df.subset == 'train'].epoch.nunique()

    @property
    def num_test_runs(self):
        df = self.report
        df = df[df.subset == 'test']
        duplicates = df.duplicated(subset=['subset', 'key', 'start']).sum() > 0
        if duplicates == True:
            return 2
        return self.report[self.report.subset == 'test'].epoch.nunique()

    @property
    def last_test_epoch(self):
        return self.report[self.report.subset == 'test'].epoch.max()

    @property
    def test_df(self):
        df = self.report
        df = df[df.subset == 'test']
        if self.consensus == 'max':
            return df.drop_duplicates(subset=['subset', 'key', 'start'], keep='first')
        elif self.consensus == 'avg':
            return df.drop_duplicates(subset=['subset', 'key', 'start'], keep='last')
