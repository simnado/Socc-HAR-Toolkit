import itertools
import math
from pathlib import Path
import numpy as np
import time
import torch
from pytorch_lightning.core.memory import ModelSummary
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from pytorch_lightning.core.lightning import LightningModule
from src.arch.backbone import Backbone
from src.eval import MultiLabelStatCurves, MultiLabelStatScores
from torchvision.transforms import Normalize


class Classifier(LightningModule):

    def __init__(self, backbone: Backbone, lr: float, weight_decay: float, epochs: int,
                 scheduler: str, optim: str, batch_size: int, num_classes: int,
                 mean: [float], std: [float],
                 train_samples: int, pretrained_path: Path,
                 num_frames: int, res: int, fps: int, consensus='max',
                 trainable_groups=None,
                 **kwargs):

        assert consensus in ['avg', 'max']

        super().__init__()
        self.save_hyperparameters('lr', 'weight_decay', 'epochs', 'scheduler', 'optim', 'batch_size',
                                  'num_frames', 'res', 'fps', 'consensus',
                                  'trainable_groups',
                                  'pretrained_path', 'mean', 'std')
        self.backbone = backbone
        self.hparams.name = self.backbone.__class__.__name__

        if self.hparams.pretrained_path is None:
            self.hparams.pretrained_path = self.backbone.provide_pretrained_weights()
        if self.hparams.pretrained_path is not None:
            self.load_weights()

        self.id = f'{self.hparams.name}_{trainable_groups}g_{num_frames}x{res}x{res}_{fps}fps_{optim}_{scheduler}_{epochs}ep_wd={weight_decay}'
        if pretrained_path:
            self.id = f'{self.id}_pretrainedOn{pretrained_path.name}'

        self.num_classes = num_classes

        self.mean = mean
        self.std = std
        self.normalize = Normalize(self.mean, self.std)

        self.reportLoss = BCEWithLogitsLoss(reduction='none')
        self.loss = BCEWithLogitsLoss(reduction='mean')

        self.train_samples = train_samples
        self.epoch_start = None

        # metrics
        self.train_stat_scores = MultiLabelStatScores(self.num_classes, threshold=0.5)
        self.val_stat_scores = MultiLabelStatScores(self.num_classes, threshold=0.5)
        self.test_stat_scores = MultiLabelStatScores(self.num_classes, threshold=0.5)

        self.train_stat_curves = MultiLabelStatCurves(self.num_classes)
        self.val_stat_curves = MultiLabelStatCurves(self.num_classes)
        self.test_stat_curves = MultiLabelStatCurves(self.num_classes)

    def forward(self, x):
        batch_size, num_chunks, channels, frames, width, height = x.size()

        assert batch_size <= self.hparams.batch_size  # smaller if test loop
        assert num_chunks <= 5
        assert channels == 3
        assert frames == self.hparams.num_frames
        assert width == height

        # todo: data layer

        # transforms
        for batch in range(batch_size):
            for chunk in range(num_chunks):
                # transform (C x T x S^2) to (T x C x S^2)
                x[batch][chunk] = self.normalize(x[batch][chunk].permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        # reshape chunks to extra samples (6D -> 5D)
        x = x.reshape((-1,) + x.shape[2:])

        assert len(x.shape) == 5
        assert x.shape[1] == 3

        out = self.backbone(x)

        # reshape back to one score per sample
        out = out.view(batch_size, num_chunks, -1)
        if self.hparams.consensus == 'avg':
            out = out.mean(dim=1)
        elif self.hparams.consensus == 'max':
            out = out.max(dim=1).values

        assert len(out) == batch_size

        return out

    def configure_optimizers(self):

        self.unfreeze_layers(groups=self.hparams.trainable_groups)

        self.hparams.accumulate_grad_batches = self.accumulate_grad_batches

        opt_params = []
        max_lrs = []
        for idx, layers in enumerate(self.trainable_layers):
            for layer in layers:
                params = layer.parameters()
                lr = self.lrs[idx]
                max_lrs.append(lr)
                opt_params.append({'params': params, 'lr': lr})

        if self.hparams.optim == 'adam':
            # set beta2=0.99 (better choice with one-cyle acc to fast.ai)
            optimizer = torch.optim.Adam(opt_params, lr=self.hparams.lr[0], weight_decay=self.hparams.weight_decay,
                                         betas=(0.9, 0.99), eps=1e-5)
        elif self.hparams.optim == 'sgd':
            optimizer = torch.optim.SGD(opt_params, lr=self.hparams.lr[0], weight_decay=self.hparams.weight_decay,
                                        momentum=0.9)
        else:
            raise Exception('no optimizer set')

        if self.hparams.scheduler == 'cosine':
            print(f'start cosine annealing with {self.train_iterations} iterations')
            # lr_scheduler = CosineAnnealingLR(optimizer, iterations)
            lr_scheduler = OneCycleLR(optimizer, max_lr=max_lrs, div_factor=10,
                                      steps_per_epoch=self.train_iterations,
                                      epochs=self.hparams.epochs,
                                      pct_start=0.05)
            lr_scheduler = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
                'reduce_on_plateau': False,
                'monitor': 'val_loss'
            }
        elif self.hparams.scheduler == 'plateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams.patience)
            lr_scheduler = {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'reduce_on_plateau': True,
                'monitor': 'val_loss'
            }
        elif self.hparams.scheduler == 'cycle':
            total_steps = math.ceil(self.hparams.epochs * self.train_iterations / self.accumulate_grad_batches / 64) * 64
            lr_scheduler = OneCycleLR(optimizer, max_lr=max_lrs, div_factor=10, total_steps=total_steps,
                                      pct_start=0.25, verbose=False)
            lr_scheduler = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
                'reduce_on_plateau': False,
            }
        else:
            raise Exception('no scheduler set')

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.epoch_start = time.time()
            self.train_stat_scores.reset()
            self.train_stat_curves.reset()

        x, y, info = batch
        out = self(x)
        losses = self.reportLoss(out, y)
        losses = torch.mean(losses, dim=1)  # reduce per sample
        # todo: is this a problem? batch_loss = losses.mean()
        batch_loss = self.loss(out, y)
        scores = torch.sigmoid(out)

        self.train_stat_scores(scores, y)
        self.train_stat_curves(scores, y)

        self.log('train_max_score', scores.max())
        self.log('train_batch_loss', batch_loss, prog_bar=True)
        self.log('train_batch_f1_macro', self.train_stat_scores.f1('macro'), prog_bar=True)
        self.log('train_batch_balanced_acc_macro', self.train_stat_scores.balanced_accuracy('macro'), prog_bar=True)

        return {'losses': losses, 'meta': info, 'loss': batch_loss}

    def training_epoch_end(self, outputs):
        outputs = {k: [dic[k] for dic in outputs] for k in outputs[0]}

        train_time = time.time() - self.epoch_start

        self.log('train_precision_macro', self.train_stat_scores.precision('macro'))
        self.log('train_recall_macro', self.train_stat_scores.recall('macro'))
        self.log('train_f1_macro', self.train_stat_scores.f1('macro'))
        self.log('train_balanced_acc_macro', self.train_stat_scores.balanced_accuracy('macro'), prog_bar=True)
        self.log('train_acc_macro', self.train_stat_scores.accuracy('macro'))
        self.log('train_hamming_loss', self.train_stat_scores.hamming_loss())
        self.log('train_time', train_time, on_epoch=True)

        self.log('train_loss', torch.cat(outputs['losses'], dim=0).mean())

        self.log('train_auroc_micro', self.train_stat_curves.auroc('micro'))
        self.log('train_auroc_macro', self.train_stat_curves.auroc('macro'))

    # Validation stuff from here

    def validation_step(self, batch, batch_idx):

        if batch_idx == 0:
            self.val_stat_scores.reset()
            self.val_stat_curves.reset()

        x, y, info = batch
        out = self(x)
        losses = self.reportLoss(out, y)
        losses = torch.mean(losses, dim=1)  # reduce per sample

        scores = torch.sigmoid(out)

        self.val_stat_scores(scores, y)
        self.val_stat_curves(scores, y)

        self.log('val_max_score', scores.max())

        return {'losses': losses, 'meta': info}

    def validation_epoch_end(self, outputs):

        outputs = {k: [dic[k] for dic in outputs] for k in outputs[0]}

        self.log('val_precision_macro', self.val_stat_scores.precision('macro'))
        self.log('val_recall_macro', self.val_stat_scores.recall('macro'))
        self.log('val_f1_macro', self.val_stat_scores.f1('macro'))
        self.log('val_balanced_acc_macro', self.val_stat_scores.balanced_accuracy('macro'), prog_bar=True)
        self.log('val_acc_macro', self.val_stat_scores.accuracy('macro'))
        self.log('val_hamming_loss', self.val_stat_scores.hamming_loss())

        self.log('val_loss', torch.cat(outputs['losses'], dim=0).mean())

        self.log('val_auroc_micro', self.val_stat_curves.auroc('micro'))
        self.log('val_auroc_macro', self.val_stat_curves.auroc('macro'))

    # Test stuff from here

    def test_step(self, batch, batch_idx):

        if batch_idx == 0:
            self.test_stat_scores.reset()
            self.test_stat_curves.reset()

        x, y, info = batch
        out = self(x)
        losses = self.reportLoss(out, y)
        losses = torch.mean(losses, dim=1)  # reduce per sample

        scores = torch.sigmoid(out)

        self.test_stat_scores(scores, y)
        self.test_stat_curves(scores, y)

        self.log('test_max_score', scores.max())

        return {'losses': losses, 'meta': info}

    def test_epoch_end(self, outputs):

        outputs = {k: [dic[k] for dic in outputs] for k in outputs[0]}
        self.log('test_loss', torch.cat(outputs['losses'], dim=0).mean())

    def count_parameters(self):
        trainable_parameters = 0
        total_parameters = 0

        for p in self.backbone.parameters():
            if p.requires_grad:
                trainable_parameters += p.numel()
            total_parameters += p.numel()

        return trainable_parameters, total_parameters

    def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
        model_summary = ModelSummary(self.backbone, mode=mode)
        # todo: for backbone and head
        return model_summary

    def unfreeze_layers(self, only_head=False, groups=None):
        if groups is None:
            groups = len(self.backbone.groups)
        if only_head:
            groups = 1

        self.hparams.trainable_groups = groups

        self.freeze()

        print('unfreezed layers: ')
        for layer in list(itertools.chain(*self.trainable_layers)):
            for name, param in layer.named_parameters():
                param.requires_grad = True

    @property
    def trainable_layers(self):
        return self.backbone.groups[::-1][0:self.hparams.trainable_groups]

    @property
    def lrs(self):
        if type(self.hparams.lr) == float:
            self.hparams.lr = [self.hparams.lr]
        return np.geomspace(self.hparams.lr[-1], self.hparams.lr[0], self.hparams.trainable_groups)

    @property
    def accumulate_grad_batches(self):
        return max(1, 64 // self.hparams.batch_size)

    @property
    def train_iterations(self):
        return math.ceil(self.train_samples / self.hparams.batch_size)

    def load_weights(self):
        if self.hparams.pretrained_path and self.hparams.pretrained_path.exists():
            state = torch.load(str(self.hparams.pretrained_path))

            try:
                self.load_state_dict(state['state_dict'], strict=True)
                print('all weights loaded')
                return True
            except RuntimeError as e:
                pass
            except KeyError as e:
                pass

            fc_layers = [f'cls_head.{name}' for name, param in self.backbone.groups[-1][0].named_parameters()]
            if 'state_dict' in state:
                state = state['state_dict']
            state = {k: v for k, v in state.items() if k not in fc_layers}
            try:
                self.backbone.load_state_dict(state, strict=False)
                print(f'backbone weights loaded. head layers ignored: {", ".join(fc_layers)}')
                return True
            except RuntimeError:
                pass

            print('cannot load weights.')
        else:
            print('no pretrained weight provided')

        return False

