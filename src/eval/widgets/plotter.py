from pathlib import Path
from IPython.display import display
import ipywidgets as widgets
from src.data import DataModule
from src.eval import ClipPlot
from pytorch_lightning.loggers import LightningLoggerBase
import torch


class Plotter:
    def __init__(self, dm: DataModule, logger: LightningLoggerBase, save_dir: Path):
        self.indices = []
        self.index = 0
        self.plot = None
        self.dm = dm
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True)
        self.logger = logger

        self.set_select = widgets.Dropdown(
            options=['train', 'val', 'test'],
            value='train',
            description='Dataset:',
        )

        self.class_select = widgets.Dropdown(
            options=['background'] + dm.classes,
            value=None,
            description='Class:',
        )

        self.select_btn = widgets.Button(
            description='OK',
            disabled=self.set_select.value is None,
        )

        self.status = widgets.Label(f'')
        self.period = widgets.HTML(
            value='',
            description='Period:',
            disabled=True
        )
        self.start = widgets.HTML(
            value='',
            description='Start:',
            disabled=True
        )
        self.end = widgets.HTML(
            value='',
            description='End:',
            disabled=True
        )
        self.labels = widgets.HTML(
            value='',
            description='Labels:',
            disabled=True
        )

        self.canvas = widgets.HTML()
        self.controls = widgets.ToggleButtons(
            options=[('grid (image)', 'grid'), ('tensor (video)', 'sample'), ('clip (video)', 'clip')],
            value='sample'
        )

        self.prev_btn = widgets.Button(
            description='<',
            disabled=self.is_first,
        )

        self.next_btn = widgets.Button(
            description='>',
            disabled=self.is_last,
        )

        self.save_btn = widgets.Button(
            description='save',
            disabled=self.plot is None,
        )


        self.select_btn.on_click(self.on_select)
        self.prev_btn.on_click(self.on_prev)
        self.next_btn.on_click(self.on_next)
        self.save_btn.on_click(self.on_save)
        def on_mode_change(b):
            self.update()
        self.controls.observe(on_mode_change, 'value')

        self.container = widgets.VBox([
            widgets.Box([self.set_select, self.class_select, self.select_btn]),
            widgets.Box([self.status, self.period, self.start, self.end, self.labels]),
            self.canvas,
            self.controls,
            widgets.HBox([self.prev_btn, self.save_btn, self.next_btn])
        ])

        #self.on_select(None)

    def on_select(self, b):
        self.indices = self.stats.indices

        if self.label:
            label_idx = self.dm.classes.index(self.label)
            if self.label != 'background':
                num_samples = int(self.stats.samples[label_idx])
                self.indices = torch.argsort(self.dataset.y[:, label_idx], descending=True)[0:num_samples]
            else:
                num_samples = int(self.stats.background_samples)
                self.indices = torch.argsort(torch.sum(self.dataset.y, dim=1))[0:num_samples]

        self.indices = self.indices[torch.randperm(len(self.indices))].tolist()
        self.index = 0
        self.update()

    def on_prev(self, b):
        self.index = max(0, self.index - 1)
        self.update()

    def on_next(self, b):
        self.index = min(self.total - 1, self.index + 1)
        self.update()

    def on_save(self, b):
        if self.mode == 'grid':
            filename = self.plot.save('svg')
        else:
            filename = self.plot.save('gif')
            print(f'saved to {filename}')
            filename = self.plot.save('mp4')
        print(f'saved to {filename}')

    def update(self):
        self.prev_btn.disabled = self.is_first
        self.next_btn.disabled = self.is_last

        self.labels.value = ', '.join([anno['label'] for anno in self.nearby_actions])

        self.canvas.value = ''
        self.plot = self.get_plot()

        self.save_btn.disabled = self.plot is None
        self.status.value = f'{self.index + 1}/{self.total}'
        self.period.value = self.plot.info['key']
        self.start.value = str(self.plot.info['start'])
        self.end.value = str(self.plot.info['end'])

        self.canvas.value = self.plot.show(mode=self.mode)

    def get_plot(self):
        return ClipPlot(self.logger, dataset=self.dataset, context=self.set_select.value, row=self.indices[self.index],
                        pred=None,
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
    def meta(self):
        row = self.indices[self.index]
        return self.dataset.info[row]

    @property
    def nearby_actions(self):
        return self.meta['annotations']