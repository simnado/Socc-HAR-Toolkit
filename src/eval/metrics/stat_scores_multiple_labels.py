from typing import Optional, Any
import torch
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.functional.reduction import class_reduce


class MultiLabelStatScores(Metric):
    def __init__(
            self,
            num_classes: int,
            threshold: float = 0.5,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.num_classes = num_classes
        # tp, fp, tn, fn, sup
        self.add_state("tp", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")
        self.add_state("sup", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")

        self.threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        assert preds.shape[1] == self.num_classes

        preds = (preds > self.threshold).float()

        self.tp += (preds * target).to(torch.long).sum(dim=0)
        self.fp += (preds * (target == 0)).to(torch.long).sum(dim=0)
        self.tn += ((preds == 0) * (target == 0)).to(torch.long).sum(dim=0)
        self.fn += ((preds == 0) * target).to(torch.long).sum(dim=0)
        self.sup += (target == 1).to(torch.long).sum(dim=0)

    def compute(self):
        """
        Computes accuracy over state.
        """
        # todo: return default metric
        return self.sup.sum()

    def balanced_accuracy(self, class_reduction: str):
        # based on https://statisticaloddsandends.wordpress.com/2020/01/23/what-is-balanced-accuracy/
        sensitivity = class_reduce(self.tp, self.tp + self.fn, self.sup, class_reduction)
        specificity = class_reduce(self.tn, self.fp + self.tn, self.sup, class_reduction)
        return (sensitivity + specificity) / 2

    def accuracy(self, class_reduction: str):
        return class_reduce(self.tp, self.sup, self.sup, class_reduction=class_reduction)

    def precision(self, class_reduction: str):
        return class_reduce(self.tp, self.tp + self.fp, self.sup, class_reduction=class_reduction)

    def recall(self, class_reduction: str):
        return class_reduce(self.tp, self.tp + self.fn, self.sup, class_reduction=class_reduction)

    def f1(self, class_reduction: str):
        prec = self.precision(class_reduction)
        rec = self.recall(class_reduction)
        beta = 1
        num = (1 + beta ** 2) * prec * rec
        denom = ((beta ** 2) * prec + rec)
        return class_reduce(num, denom, self.sup, class_reduction=class_reduction)

    def hamming_loss(self):
        return 1 - self.accuracy('micro')
