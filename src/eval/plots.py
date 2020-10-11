import os
from typing import Optional
import base64
from PIL import ImageDraw, ImageFont
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from IPython.core.display import HTML, Video
from celluloid import Camera
from matplotlib.animation import FFMpegWriter, PillowWriter
from pytorch_lightning.loggers import LightningLoggerBase
from torchvision import io
from torchvision.utils import make_grid
from torchvision.transforms import functional as T
from src.data import HarDataset


class VideoPlot:

    def __init__(self, logger: LightningLoggerBase, video_path: str, segment: [int], classes: [str], match_key: str,
                 ground_truth_annotations: dict, prediction_annotations: dict, save_dir='.'):
        self.classes = classes
        self.segment = segment
        self.logger = logger
        self.video_path = video_path
        self.ground_truth_annotations = ground_truth_annotations
        self.prediction_annotations = prediction_annotations
        self.save_dir = save_dir
        self.match_key = match_key
        self.title = f'{self.match_key}@{self.segment[0]}-{self.segment[1]}'

    def decode(self):

        path = f'{self.save_dir}/demo-{self.title}.mp4'
        if os.path.exists(path):
            return

        src_video, _, _ = io.read_video(self.video_path, self.segment[0], self.segment[1], pts_unit='sec')
        duration = self.segment[1] - self.segment[0]
        num_frames = src_video.shape[0]
        fps = num_frames / duration

        demo_vid = torch.zeros((num_frames, 360, 640, 3))

        for idx, img in tqdm(enumerate(src_video), total=src_video.shape[0]):
            # img: [H, W, C]

            img = img.permute((2, 0, 1))
            img = T.to_pil_image(img)

            img = T.resize(img, (360, 640))

            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            w, h = font.getsize(self.title)
            draw.rectangle((0, 10, w, 10 + h), fill='white')
            draw.text((0, 10), self.title, (0, 0, 0))

            offset = self.segment[0] + round(idx/fps)

            pred_labels = [anno['label'] for anno in self.prediction_annotations if ClassificationDataset.overlap(anno['segment'], [offset, offset+1]) > 0]
            pred_labels.sort()
            gt_labels = [anno['label'] for anno in self.ground_truth_annotations if ClassificationDataset.overlap(anno['segment'], [offset, offset+1]) > 0]
            gt_labels.sort()

            predicted_line = f'predicted: {", ".join(pred_labels)}'
            w, h = font.getsize(predicted_line)
            draw.rectangle((0, 20, w, 20 + h), fill='white')
            draw.text((0, 20), predicted_line, (0, 0, 0))

            correct = ''.join(pred_labels) == ''.join(gt_labels)
            if correct:
                gt_line = f'correct!'
            else:
                gt_line = f'should be: {", ".join(gt_labels)}'
            w, h = font.getsize(gt_line)
            draw.rectangle((0, 30, w, 30 + h), fill='white')
            draw.text((0, 30), gt_line, (0, 0, 0))

            img = T.to_tensor(img)
            img = img.permute((1,2,0))


            demo_vid[idx] = img

        res = io.write_video(path, (demo_vid * 255).type(torch.uint8), 25)
        return demo_vid

    def show(self):
        self.decode()
        location = f'{self.save_dir}/demo-{self.title}.mp4'
        if os.path.isfile(location):
            ext = 'mp4'
        else:
            print("Error: Please check the path.")
            return
        video_encoded = open(location, "rb")
        binary_file_data = video_encoded.read()
        base64_encoded_data = base64.b64encode(binary_file_data)
        video_tag = '<video width="320" height="240" controls alt="test" src="data:video/{0};base64,{1}">'.format(ext, base64_encoded_data)
        return HTML(data=video_tag)
        #print(location)
        #return Video('/content/develop/Classifier/notebooks/demo-266236@2@4576-4606.mp4')

    def save(self):
        self.decode()
        filename = f'{self.save_dir}/demo-{self.title}.mp4'
        with self.logger.experiment.context_manager("test"):
            self.logger.experiment.log_asset(filename)


class ClipPlot:

    def __init__(self, logger: LightningLoggerBase, clip: torch.Tensor, y: torch.Tensor, info: dict, classes: [str],
                 pred: Optional[torch.Tensor], save_dir='.'):
        self.logger = logger
        self.x = clip.permute((1, 2, 3, 0))  # (T, H, W, C)
        self.y = y
        self.info = info
        self.pred = pred
        self.classes = classes
        self.y_labels = np.array(self.classes)[np.array(self.y) > 0.5].tolist()
        self.save_dir = save_dir
        self._grid_fig = None
        self._clip_fig = None
        self.filename = f'{self.info["video"].split("/")[-2]}@{self.info["offset"]}'
        self.title = f'Class={", ".join(self.y_labels)} {self.filename}'

    def show(self, mode='clip'):
        if mode == 'grid':
            return self.grid_plot
        elif mode == 'clip':
            return HTML(self.clip_plot.to_html5_video())

    def save(self, format: str):
        filename = self.filename
        if format == 'svg':
            filename = f'{self.save_dir}/{self.filename}.svg'
            self.grid_plot.savefig(filename, format='svg')
        elif format == 'mp4':
            filename = f'{self.save_dir}/{self.filename}.mp4'
            self.clip_plot.save(filename,
                                writer=FFMpegWriter(fps=12, metadata=dict(artist='Soccar-32'), bitrate=1800))
        elif format == 'gif':
            filename = f'{self.save_dir}/{self.filename}.gif'
            self.clip_plot.save(filename, writer=PillowWriter(fps=12))

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

        num_frames = self.x.shape[0]

        # frame grid
        img_list = self.x.permute((0, 3, 1, 2))
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

        matplotlib.use("Agg")

        # plot pred
        animation_fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(12, 8))

        camera = Camera(animation_fig)
        for idx, img in enumerate(self.x):
            ax0.set_title(self.title)
            ax0.imshow(img)
            ax0.text(x=len(img) + 5, y=len(img) + 12, s=f'{idx}/{len(self.x)}')

            self._score_plot(ax1)

            camera.snap()

        self._clip_fig = camera.animate(interval=100, blit=True, repeat_delay=1000)
        plt.close()

        return self._clip_fig


class PlotIterator(object):
    def __init__(self, dataset: HarDataset, logger: LightningLoggerBase,
                 indices: [int], pred: Optional[torch.Tensor], context=None, save_dir='Data'):
        self.dataset = dataset
        self.indices = indices
        self.pred = pred
        # self.order = indices
        self.pointer = 0
        self.save_dir = save_dir
        self.logger = logger
        self.context = context

    def __iter__(self):
        # todo: permute indices and pred: self.order = np.random.permutation(self.indices)
        self.pointer = 0
        return self

    def __next__(self):
        if self.pointer > len(self.indices):
            raise StopIteration
        else:
            # row = self.order[self.pointer]
            row = self.indices[self.pointer]
            _, y, info = self.dataset[row]
            x = self.dataset.get_tensor(row)
            plot = ClipPlot(self.logger, x, y, {**info, 'context': self.context}, self.dataset.classes,
                            self.pred[self.pointer] if self.pred is not None else None, self.save_dir)
            self.pointer += 1
            return plot
