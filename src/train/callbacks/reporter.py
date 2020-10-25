from itertools import chain
from pathlib import Path
from pytorch_lightning import Callback
import torch
import pandas as pd


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
        self._save_worst_samples('train', self.train_data, trainer.current_epoch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            self.val_data = outputs
        else:
            self.push(self.val_data, outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._save_worst_samples('val', self.val_data, trainer.current_epoch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            self.test_data = outputs
        else:
            self.push(self.test_data, outputs)

    def on_test_epoch_end(self, trainer, pl_module):
        self._save_worst_samples('test', self.test_data, trainer.current_epoch)

    def _save_worst_samples(self, context: str, data: dict, epoch: int):

        worst_idx = torch.argsort(data['losses'], descending=True)

        df = pd.DataFrame(
            columns=['subset', 'key', 'video', 'start', 'end', 'labels', 'critical', 'epoch', 'index', 'y', 'scores',
                     'loss'])

        for idx in worst_idx:
            info = data['meta'][idx.item()]
            df = df.append({'key': info['key'], 'video': info['video'], 'start': info['start'], 'end': info['end'],
                            'critical': info['critical'],
                            'index': int(info['index']),
                            'labels': ', '.join(
                                [self.classes[idx] for idx in torch.arange(0, 32)[data['y'][idx] > 0]])},
                           ignore_index=True)

        df['loss'] = data['losses'][worst_idx].tolist()
        df['scores'] = data['scores'][worst_idx].tolist()
        df['y'] = data['y'][worst_idx].tolist()
        df['epoch'] = epoch
        df['subset'] = context

        self.df = self.df.append(df, ignore_index=True)
        self.df.to_csv(self.report_file)

    @staticmethod
    def push(data: dict, out):
        data['y'] = torch.cat([data['y'], out['y']], dim=0)
        data['scores'] = torch.cat([data['scores'], out['scores']], dim=0)
        data['losses'] = torch.cat([data['losses'], out['losses']], dim=0)
        data['meta'].append(*out['meta'])
