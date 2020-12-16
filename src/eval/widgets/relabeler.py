from pathlib import Path
from IPython.display import display
import ipywidgets as widgets
from src.data import DataModule
from src.eval import ClipPlot, Plotter, Transactions, ReportEvaluationModule
from pytorch_lightning.loggers import LightningLoggerBase
import torch
import pandas as pd


class Relabeler(Plotter):
    def __init__(self, dm: DataModule, logger: LightningLoggerBase, report: pd.DataFrame, transactions: Transactions,
                 save_dir: Path):
        self.eval = ReportEvaluationModule(str(save_dir), dm, report, logger)

        self.epoch = widgets.Dropdown(
            options=[(f'{idx + 1} {"(test)" if idx == self.eval.last_test_epoch else ""}', idx) for idx in
                     range(self.eval.num_epochs)],
            value=None
        )

        self.sort_descending = widgets.Checkbox(
            value=True,
            description='Absteigend',
            disabled=False,
            indent=False
        )

        super().__init__(dm, logger, save_dir)

        self.scores = None

        self.container = widgets.VBox([
            widgets.Box([self.set_select, self.epoch, self.class_select, self.sort_descending, self.select_btn]),
            widgets.Box([self.status, self.period, self.start, self.end, self.labels]),
            self.canvas,
            self.controls,
            widgets.HBox([self.prev_btn, self.save_btn, self.next_btn])
        ])

        self.on_select(None)

    def on_select(self, b):
        self.indices, self.scores = self.eval.get_top_loss_samples(
            self.set_select.value, self.epoch.value, self.label, desc=self.sort_descending.value)

        self.indices = self.indices.tolist()
        self.index = 0
        self.update()

    def get_plot(self):
        return ClipPlot(None, dataset=self.dataset, context=self.set_select.value, row=self.indices[self.index],
                        pred=self.scores[self.index],
                        save_dir=self.save_dir)

    def _ipython_display_(self):
        return display(self.container)

    @property
    def total(self):
        return len(self.indices)

    @property
    def is_first(self):
        return self.index == 0

    @property
    def is_last(self):
        return self.index + 1 == self.total or self.total == 0

    @property
    def dataset(self):
        return self.dm.datasets[self.set_select.value]

    @property
    def stats(self):
        return self.dm.stats[self.set_select.value]

    @property
    def label(self):
        return self.class_select.value

    @property
    def mode(self):
        return self.controls.value