from pathlib import Path
from pytorch_lightning import Callback
import torch
import pandas as pd

from src.eval import MultiLabelStatCurves


class Reporter(Callback):

    def __init__(self, out_dir: Path, classes: [str]):
        """
        Args:
            out_dir: Path
        """
        super().__init__()
        self.out_dir = out_dir
        self.out_dir.mkdir(exist_ok=True)

        self.classes = classes

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.df = pd.DataFrame(
            columns=['subset', 'key', 'video', 'start', 'end', 'labels', 'critical', 'epoch', 'index', 'y', 'scores',
                     'loss'])

        self.report_file = self.out_dir.joinpath('report.csv')

    def on_fit_start(self, trainer, pl_module):
        self.df = pd.DataFrame(
            columns=['subset', 'key', 'video', 'start', 'end', 'labels', 'critical', 'epoch', 'index', 'y', 'scores',
                     'loss'])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        outputs = outputs[0][0]['extra']
        if batch_idx == 0:
            self.train_data = outputs
        else:
            self.push(self.train_data, outputs)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        self._report('train', self.train_data, pl_module.train_stat_curves.scores, pl_module.train_stat_curves.target, trainer.current_epoch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            self.val_data = outputs
        else:
            self.push(self.val_data, outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._report('val', self.val_data, pl_module.val_stat_curves.scores, pl_module.val_stat_curves.target, trainer.current_epoch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            self.test_data = outputs
        else:
            self.push(self.test_data, outputs)

    def on_test_epoch_end(self, trainer, pl_module):

        self._report('test', self.test_data, pl_module.test_stat_curves.scores, pl_module.test_stat_curves.target, trainer.current_epoch)

    def _report(self, context: str, data: dict, scores: torch.Tensor, y: torch.Tensor, epoch: int):

        worst_idx = torch.argsort(data['losses'], descending=True)

        df = pd.DataFrame(
            columns=['subset', 'key', 'video', 'start', 'end', 'labels', 'critical', 'epoch', 'index', 'y', 'scores',
                     'loss'])

        for row in worst_idx:
            info = data['meta'][row.item()]
            df = df.append({'key': info['key'], 'video': info['video'], 'start': info['start'], 'end': info['end'],
                            'critical': info['critical'],
                            'index': int(info['index']),
                            'labels': ', '.join(
                                [self.classes[idx] for idx in torch.arange(0, 32)[y[row] > 0] ])},
                           ignore_index=True)

        df['loss'] = data['losses'][worst_idx].tolist()
        df['scores'] = scores[worst_idx].tolist()
        df['y'] = y[worst_idx].tolist()
        df['epoch'] = epoch
        df['subset'] = context

        self.df = self.df.append(df, ignore_index=True)
        self.df.to_csv(self.report_file)

    @staticmethod
    def push(data: dict, out):
        data['losses'] = torch.cat([data['losses'], out['losses']], dim=0)
        data['meta'] += out['meta']
