from typing import Optional
import torch
from pytorch_lightning.loggers import LightningLoggerBase
from src.data import HarDataset
from src.eval.plot import ClipPlot


class PlotIterator(object):
    def __init__(self, dataset: HarDataset, logger: LightningLoggerBase,
                 indices: [int], pred: Optional[torch.Tensor], context=None, save_dir='Data'):
        self.dataset = dataset
        self.indices = indices
        self.pred = pred
        self.pointer = 0
        self.save_dir = save_dir
        self.logger = logger
        self.context = context

    def __iter__(self):
        self.pointer = 0
        return self

    def __next__(self):
        if self.pointer > len(self.indices):
            raise StopIteration
        else:
            # row = self.order[self.pointer]
            row = self.indices[self.pointer]
            plot = ClipPlot(self.logger, dataset=self.dataset, context=self.context, row=row, pred=self.pred[self.pointer], save_dir=self.save_dir)
            self.pointer += 1
            return plot
