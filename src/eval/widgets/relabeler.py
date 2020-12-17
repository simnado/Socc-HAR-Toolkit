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

        self.epoch_select = widgets.Dropdown(
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
        self.action_buttons = widgets.ToggleButtons(
            options=['pass', 'add', 'edit', 'delete'],
            value='pass'
        )
        self.action_buttons.observe(self.on_action_mode_changed, 'value')

        self.rl_label = widgets.Dropdown()
        self.rl_label.observe(self.on_relabel_entity_changed, 'value')

        self.rl_segment = widgets.FloatRangeSlider(
            value=[5, 7.5],
            min=0,
            max=10.0,
            step=0.1,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )

        self.rl_submit = widgets.Button(description='OK')
        self.rl_submit.on_click(self.on_relabel)

        self.container = widgets.VBox([
            widgets.Box([self.set_select, self.epoch_select, self.class_select, self.sort_descending, self.select_btn]),
            widgets.Box([self.status, self.period, self.start, self.end, self.labels]),
            self.canvas,
            self.controls,
            widgets.HBox([self.prev_btn, self.save_btn, self.next_btn]),
            self.action_buttons,
            widgets.HBox([self.rl_label, self.rl_segment, self.rl_submit]),
        ])

        self.on_select(None)

    def on_relabel_entity_changed(self, b):
        if self.relabel_action == 'delete' or self.relabel_action == 'edit':
            row = self.indices[self.index]
            start = self.dataset.info[row]['start']
            end = self.dataset.info[row]['end']
            print(self.nearby_actions)
            annos = [anno for (url, label) in self.nearby_actions if url == self.rl_label.value]

            self.rl_segment.disabled = False
            self.rl_segment.min = 0
            self.rl_segment.max = end + 1
            self.rl_segment.min = start - 1

            # todo: get annos
            self.rl_segment.value = annos[0]['segment']


    def on_action_mode_changed(self, b):
        self.rl_label.value = None
        self.rl_label.disabled = False
        self.rl_segment.disabled = True
        self.rl_segment.value = [0,0]
        if self.relabel_action == 'pass':
            self.rl_label.disabled = True
            self.rl_segment.disabled = True
        elif self.relabel_action == 'add':
            self.rl_label.options = self.dm.classes
        elif self.relabel_action == 'edit':
            self.rl_label.options = self.nearby_actions
        elif self.relabel_action == 'delete':
            self.rl_label.options = self.nearby_actions

    def on_relabel(self, b):
        print(f'{self.relabel_action} {self.rl_label.value} at {self.rl_segment.value}')
        self.action_buttons.value = 'pass'

    def on_select(self, b):
        self.indices, self.scores = self.eval.get_top_loss_samples(
            self.set_select.value, self.epoch, self.label, desc=self.sort_descending.value)

        self.indices = self.indices.tolist()
        self.index = 0
        self.update()

    def update(self):
        super().update()
        self.action_buttons.value = 'pass'

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

    @property
    def epoch(self):
        return self.epoch_select.value

    @property
    def relabel_action(self):
        return self.action_buttons.value

    @property
    def nearby_actions(self):
        print('nearby')
        row = self.indices[self.index]
        annos = self.dataset.info[row]['annotations']
        print([(anno['label'], anno['url']) for anno in annos])
        return [(anno['label'], anno['url']) for anno in annos]