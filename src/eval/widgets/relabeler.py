from pathlib import Path
from IPython.display import display
import ipywidgets as widgets
from src.data import DataModule
from src.eval import ClipPlot, Plotter, Transactions, ReportEvaluationModule
from pytorch_lightning.loggers import LightningLoggerBase
import pandas as pd


class Relabeler(Plotter):
    def __init__(self, dm: DataModule, logger: LightningLoggerBase, report: pd.DataFrame, transactions: Transactions,
                 save_dir: Path):
        self.eval = ReportEvaluationModule(str(save_dir), dm, report, logger)
        self.transactions = transactions
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
            layout=widgets.Layout(width='100%')
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

        self.changelog = widgets.Textarea(
            value='',
            description='Changes:',
            disabled=True,
            layout=widgets.Layout(width='100%', height='100px')
        )
        self.container = widgets.GridBox([
            widgets.Box([self.set_select, self.epoch_select, self.class_select, self.sort_descending, self.select_btn]),
            widgets.Box([self.status, self.period, self.start, self.end, self.labels]),
            self.canvas,
            self.controls,
            widgets.HBox([self.prev_btn, self.save_btn, self.next_btn]),
            self.action_buttons,
            widgets.HBox([self.rl_label, self.rl_segment, self.rl_submit], layout=widgets.Layout(width='100%')),
            self.changelog
        ], layout=widgets.Layout(grid_template_columns="repeat(1, 700px)"))

        self.on_select(None)

    def on_relabel_entity_changed(self, b):
        if self.rl_label.value and (self.relabel_action == 'add' or self.relabel_action == 'edit' or self.relabel_action == 'pass'):
            self.rl_segment.disabled = self.relabel_action == 'pass'

            print(f'index {self.rl_label.value} selected')
            if self.relabel_action != 'add':
                self.rl_segment.value = self.nearby_actions[self.rl_label.value]['segment']

    def on_action_mode_changed(self, b):

        self.rl_label.value = None
        self.rl_label.disabled = False
        self.rl_segment.disabled = False
        self.rl_segment.value = [0,0]
        if self.relabel_action == 'pass':
            self.rl_segment.disabled = True
            labels = [(anno['label'], idx) for idx, anno in enumerate(self.nearby_actions) if 'verified' not in anno]
            self.rl_label.options = labels
        elif self.relabel_action == 'add':
            self.rl_label.options = self.dm.classes
        elif self.relabel_action == 'edit':
            labels = [(anno['label'], idx) for idx, anno in enumerate(self.nearby_actions) if 'verified' not in anno]
            self.rl_label.options = labels
        elif self.relabel_action == 'delete':
            labels = [(anno['label'], idx) for idx, anno in enumerate(self.nearby_actions) if 'verified' not in anno]
            self.rl_label.options = labels
        self.rl_label.value = None

    def on_relabel(self, b):
        if self.relabel_action == 'pass':
            annotation_label = self.meta['annotations'][self.rl_label.value]
            self.transactions.verify(self.period.value, annotation_label['url'], annotation_label['label'])
            annotation_label = self.meta['annotations'][self.rl_label.value]
            self.changelog.value += f'\nVERIFIED {annotation_label["label"]} at {self.rl_segment.value}'
        elif self.relabel_action == 'add':
            self.transactions.add(self.period.value, self.rl_label.value, self.rl_segment.value)
            self.changelog.value += f'\nADD {self.rl_label.value} at {self.rl_segment.value}'
        elif self.relabel_action == 'edit':
            annotation_label = self.meta['annotations'][self.rl_label.value]
            self.transactions.adjust(self.period.value, annotation_label['url'], annotation_label['label'], self.rl_segment.value)
            self.changelog.value += f'\nEDIT {annotation_label["label"]} at {self.rl_segment.value}'
        elif self.relabel_action == 'delete':
            annotation_label = self.meta['annotations'][self.rl_label.value]
            self.transactions.remove(self.period.value, annotation_label['url'], annotation_label['label'])
            self.changelog.value += f'\nDELETE {annotation_label["label"]} at {self.rl_segment.value}'

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
        self.changelog.value = ''
        start = self.meta['start']
        end = self.meta['end']
        self.rl_segment.min = 0
        self.rl_segment.max = end + 1
        self.rl_segment.min = start - 1

    def get_plot(self):
        return ClipPlot(self.logger, dataset=self.dataset, context=self.set_select.value, row=self.indices[self.index],
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
