from typing import Optional
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
import numpy as np
from celluloid import Camera
from pathlib import Path
from src.data import HarDataset
from torchvision.utils import make_grid
from pytorch_lightning.loggers import LightningLoggerBase
from IPython.core.display import HTML, Video


class ClipPlot:

    def __init__(self, logger: LightningLoggerBase, dataset: HarDataset, context: str, row: int,
                 pred: Optional[torch.Tensor], save_dir: Path):
        self.logger = logger
        self.row = row
        self.context = context
        self.dataset = dataset
        self.y = dataset.y[row]
        self.info = {**dataset.info[row], 'context': context}

        self.pred = pred
        self.classes = dataset.classes
        self.y_labels = np.array(self.classes)[np.array(self.y) > 0.5].tolist()
        self.save_dir = save_dir
        self._grid_fig = None
        self._sample_fig = None
        self._clip_fig = None
        self.filename = f'{self.info["key"]}_{self.info["start"]}-{self.info["end"]}'
        self.title = f'Class={", ".join(self.y_labels)} {self.filename}'

    def show(self, mode='clip'):
        if mode == 'grid':
            return self.grid_plot
        elif mode == 'clip':
            return HTML(self.clip_plot.to_html5_video())
        elif mode == 'sample':
            return HTML(self.sample_plot.to_html5_video())

    def save(self, format: str):
        filename = self.save_dir.joinpath(f'{self.filename}.{format}')
        if format == 'svg':
            self.grid_plot.savefig(filename, format='svg')
        elif format == 'mp4':
            self.clip_plot.save(filename,
                                writer=FFMpegWriter(fps=12, metadata=dict(artist='SoccHAR-32'), bitrate=1800))
        elif format == 'gif':
            self.clip_plot.save(filename, writer=PillowWriter(fps=12))

        if logger:
            with self.logger.experiment.context_manager("val"):
                self.logger.experiment.log_asset(filename,
                                                 metadata={'split': self.info['context'], 'id': self.filename,
                                                           'pred': self.pred})

        return filename

    def _score_plot(self, axes):
        pred = self.pred
        score_title = 'predictions scores'

        if self.pred is None:
            pred = self.y
            score_title = 'ground truth'

        sort = pred.argsort(descending=True)
        axes.set_xlim(0, 100)
        axes.set_title(score_title)
        axes.xaxis.set_visible(False)
        axes.yaxis.set_visible(False)
        axes.grid(axis='x')

        top_k = [pred[sort[i]] * 100 for i in range(5)]
        top_k_label = [self.classes[sort[i]] for i in range(5)]
        axes.barh(y=[-1, -2, -3, -4, -5], width=top_k, label=top_k_label, color='powderblue')
        for i in range(5):
            axes.annotate(top_k_label[i], xy=(5, -1 * (i + 1)))

        return axes

    @property
    def grid_plot(self):
        if self._grid_fig:
            return self._grid_fig

        x = self._get_x()
        num_frames = x.shape[0]

        # frame grid
        img_list = x.permute((0, 3, 1, 2))
        grid = make_grid(img_list, padding=10)

        # create figure
        self._grid_fig = plt.figure(figsize=(28, 3 * (num_frames // 8)), dpi=200)
        gs = self._grid_fig.add_gridspec(1, 5, wspace=0.1, hspace=0.1)

        # plot grid
        axes = plt.Subplot(self._grid_fig, gs[:, 0:4])
        axes.axis('off')
        axes.set_title(self.title)
        axes.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
        self._grid_fig.add_subplot(axes)

        axes = plt.Subplot(self._grid_fig, gs[:, 4])
        self._score_plot(axes)

        # plot ground truth
        self._grid_fig.add_subplot(axes)
        plt.close()

        return self._grid_fig

    @property
    def clip_plot(self):
        if self._clip_fig:
            return self._clip_fig

        self._clip_fig = self._build_animated_plot(resized=False)
        plt.close()

        return self._clip_fig

    @property
    def sample_plot(self):
        if self._sample_fig:
            return self._sample_fig

        self._sample_fig = self._build_animated_plot(resized=True)
        plt.close()

        return self._sample_fig

    def _build_animated_plot(self, resized: bool):
        matplotlib.use("Agg")

        # plot pred
        animation_fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(12, 8))
        x = self._get_x(resize=resized)

        camera = Camera(animation_fig)
        for idx, img in enumerate(x):
            ax0.set_title(self.title)
            ax0.imshow(img)
            ax0.text(x=len(img) + 5, y=len(img) + 12, s=f'{idx}/{len(x)}')

            self._score_plot(ax1)

            camera.snap()

        return camera.animate(interval=100, blit=True, repeat_delay=1000)

    def _get_x(self, resize=True):
        x = self.dataset.get_tensor(self.row, resize)
        return x.permute((1, 2, 3, 0))  # (T, H, W, C)

