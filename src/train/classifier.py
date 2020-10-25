import itertools
from pathlib import Path
import numpy as np
import time
import torch
from pytorch_lightning.core.memory import ModelSummary
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics import Accuracy

from src.arch.backbone import Backbone
from src.eval import hamming, roc_auc


class Classifier(LightningModule):

    def __init__(self, backbone: Backbone, lr: float, weight_decay: float, epochs: int,
                 scheduler: str, optim: str, batch_size: int, num_classes: int,
                 train_iterations: int, pretrained_path: Path,
                 num_frames: int, res: int, fps: int,
                 trainable_groups=None, accumulate_grad_batches=1, patience=5,
                 **kwargs):

        super().__init__()
        self.save_hyperparameters('lr', 'weight_decay', 'epochs', 'scheduler', 'optim', 'batch_size',
                                  'num_frames', 'res', 'fps',
                                  'patience', 'accumulate_grad_batches', 'trainable_groups',
                                  'pretrained_path')
        self.backbone = backbone
        self.hparams.name = self.backbone.__class__.__name__

        self.id = f'{self.hparams.name}_{trainable_groups}g_{num_frames}x{res}x{res}_{fps}fps_{optim}_{scheduler}_{epochs}ep_wd={weight_decay}'
        if pretrained_path:
            self.id = f'{self.id}_pretrainedOn{pretrained_path.name}'

        self.num_classes = num_classes

        self.loss = BCEWithLogitsLoss(reduction='none')

        self.train_iterations = train_iterations
        self.epoch_start = None

        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        # batch_size, channels, frames, width, height = x.size()
        # todo: data layer
        return self.backbone(x)

    def configure_optimizers(self):

        self.unfreeze_layers(groups=self.hparams.trainable_groups)

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

        total_steps = int(self.hparams.epochs * self.train_iterations / self.hparams.accumulate_grad_batches)
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
            lr_scheduler = OneCycleLR(optimizer, max_lr=max_lrs, div_factor=10, total_steps=total_steps,
                                      pct_start=0.25)
            lr_scheduler = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
                'reduce_on_plateau': False,
                'monitor': 'val_loss'
            }
        else:
            raise Exception('no scheduler set')

        return [optimizer], [lr_scheduler]

    def on_train_start(self) -> None:
        super().on_train_start()
        train_params, total_params = self.count_parameters()
        self.logger.experiment.log_other('total_parameters', total_params)
        self.logger.experiment.log_other('trainable_parameters', train_params)
        self.logger.experiment.log_text(text=str(self.summarize()), metadata={'type': 'summary'})
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.epoch_start = time.time()

        x, y, info = batch
        out = self(x)
        losses = self.loss(out, y)
        losses = torch.mean(losses, dim=1)  # reduce per sample
        batch_loss = losses.mean()
        scores = torch.sigmoid(out)

        self.train_acc(scores, y)

        self.log('batch_loss', batch_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('train_loss', batch_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return {'y': y, 'scores': scores, 'losses': losses, 'meta': info, 'loss': batch_loss}

    def training_epoch_end(self, outputs):

        train_time = time.time() - self.epoch_start

        #self.log('train_micro_auc', micro_auc, on_epoch=True)
        #self.log('train_macro_auc', macro_auc, on_epoch=True)
        self.log('train_time', train_time, on_epoch=True)

    # Validation stuff from here

    def validation_step(self, batch, batch_idx):

        x, y, info = batch
        out = self(x)
        losses = self.loss(out, y)
        losses = torch.mean(losses, dim=1)  # reduce per sample

        batch_loss = losses.mean()
        scores = torch.sigmoid(out)

        self.val_acc(scores, y)

        self.log('val_loss', batch_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return {'y': y, 'scores': scores, 'losses': losses, 'meta': info}

    def validation_epoch_end(self, outputs):

        # prevent half precision metric bugs
        # y = torch.FloatTensor(y.to(torch.float32).tolist())
        # pred = torch.FloatTensor(pred.to(torch.float32).tolist())

        #acc = 1 - hamming(y, scores)

        #micro_auc, macro_auc = 0, 0

        #try:
        #    micro_auc, macro_auc, _ = roc_auc(y, scores)

        #except ValueError as err:
        #    print('cannot calculate roc: ')
        #    print(err)

        #self.log('val_macro_auc', macro_auc)
        #self.log('val_micro_auc', micro_auc)
        print('val epoch ends')

    # Test stuff from here

    def test_step(self, batch, batch_idx):

        x, y, info = batch
        out = self(x)
        losses = self.loss(out, y)
        losses = torch.mean(losses, dim=1)  # reduce per sample

        batch_loss = losses.mean()
        scores = torch.sigmoid(out)

        # y = y.cpu().detach()
        # pred = pred.cpu().detach()

        self.log('test_loss', batch_loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)

        return {'y': y, 'scores': scores, 'losses': losses, 'ids': np.array(info['id'])}

    def test_epoch_end(self, outputs):

        # prevent half precision metric bugs
        # y = torch.FloatTensor(y.to(torch.float32).tolist())
        # pred = torch.FloatTensor(pred.to(torch.float32).tolist())

        #acc = 1 - hamming(y, pred)
        #micro_auc, macro_auc = 0, 0

        #try:
        #    micro_auc, macro_auc, _ = roc_auc(y, pred)

        #except ValueError as err:
        #    print('cannot calculate roc: ')
        #    print(err)

        #self.log('val_acc', acc, prog_bar=True)
        #self.log('test_macro_auc', macro_auc)
        #self.log('test_micro_auc', micro_auc)
        pass

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
            groups = len(self.layers)
        if only_head:
            groups = 1

        self.hparams.trainable_groups = groups

        self.freeze()

        for layer in list(itertools.chain(*self.trainable_layers)):
            for name, param in layer.named_parameters():
                param.requires_grad = True
                print(f'{name} is trainable')

    @property
    def trainable_layers(self):
        return self.backbone.groups[::-1][0:self.hparams.trainable_groups]

    @property
    def lrs(self):
        if type(self.hparams.lr) == float:
            self.hparams.lr = [self.hparams.lr]
        return np.linspace(self.hparams.lr[-1], self.hparams.lr[0], self.hparams.trainable_groups)

    def load_weights(self):
        if self.hparams.pretrained_path and self.hparams.pretrained_path.exists():
            state = torch.load(str(self.hparams.pretrained_path))
            if 'state_dict' in state:
                self.load_state_dict(state['state_dict'], strict=True)
                print('weights loaded.')
                return True

        return False

