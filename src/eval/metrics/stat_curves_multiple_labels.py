from typing import Optional, Any
import torch
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.functional.reduction import class_reduce
from pytorch_lightning.metrics import Accuracy
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, auc
import numpy as np


class MultiLabelStatCurves(Metric):
    def __init__(
            self,
            num_classes: int,
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
        self.add_state("scores", default=torch.zeros(0, self.num_classes), dist_reduce_fx="cat")
        self.add_state("target", default=torch.zeros(0, self.num_classes), dist_reduce_fx="cat")
        self.add_state("sup", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        assert preds.shape[1] == self.num_classes

        self.scores = torch.cat([self.scores, preds], dim=0)
        self.target = torch.cat([self.target, target], dim=0)
        self.sup += (target == 1).to(torch.long).sum(dim=0)

    def compute(self):
        """
        Computes accuracy over state.
        """
        # todo: return default metric
        return self.sup.sum()

    def threshold_finder_scalar(self, class_reduction='macro'):
        xs = torch.linspace(0.05, 0.95, 29)

        metric = Accuracy

        if class_reduction == 'micro':
            return [metric(torch.reshape(self.target, (-1,)), torch.reshape(self.scores, -1), threshold=i) for i in xs]

        class_curves = torch.zeros((32, len(xs)))
        for cls in range(self.num_classes):
            class_curves[cls] = torch.stack([metric(threshold=i)(self.target[:, cls], self.scores[:, cls]) for i in xs])

        if class_reduction == 'macro':
            return xs, class_curves.mean(dim=1)
        elif class_reduction is None:
            return xs, class_curves

    def roc(self, class_reduction=None):
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        peaks = dict()

        if class_reduction == 'micro':
            targets = torch.reshape(self.target, -1)
            scores = torch.reshape(self.scores, -1)
            fpr, tpr, thresholds = roc_curve(targets, scores)
            peaks = thresholds[np.argmax(tpr - fpr)]
        else:
            for i in range(self.num_classes):
                fpr[i], tpr[i], thresholds[i] = roc_curve(self.target[:, i], self.scores[:, i])
                peaks[i] = thresholds[i][np.argmax(tpr[i] - fpr[i])]
                # todo: auc_score = auc(fpr[i], tpr[i])
            if class_reduction == 'macro':
                # todo:
                pass

        return fpr, tpr, thresholds, peaks

    def auroc(self, class_reduction: str):

        try:
            if class_reduction is None:
                return roc_auc_score(self.target.cpu(), self.scores.cpu(), average=None)
            elif class_reduction in ['micro', 'macro']:
                return roc_auc_score(self.target.cpu(), self.scores.cpu(), average=class_reduction)
        except ValueError as e:
            print('cannot compute roc: not all classes present')
            print(e)
            return 0

    def precision(self, class_reduction: str):
        return class_reduce(self.tp, self.tp + self.fp, self.sup, class_reduction=class_reduction)
