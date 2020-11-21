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

    def roc(self, class_reduction: [str], class_idxs: [int]):
        fpr = []
        tpr = []
        thresholds = []
        peak_idxs = []

        if 'micro' in class_reduction:
            targets = torch.flatten(self.target)
            scores = torch.flatten(self.scores)
            fpr_m, tpr_m, thresholds_m = roc_curve(targets, scores)

            fpr.append(fpr_m)
            tpr.append(tpr_m)
            thresholds.append(thresholds_m)
            peak_idxs.append(np.argmax(tpr_m - fpr_m))
        for i in class_idxs:
            fpr_c, tpr_c, thresholds_c = roc_curve(self.target[:, i], self.scores[:, i])
            fpr.append(fpr_c)
            tpr.append(tpr_c)
            thresholds.append(thresholds_c)

            peak_idxs.append(np.argmax(tpr_c - fpr_c))
        if 'macro' in class_reduction:
            # todo:
            pass

        return fpr, tpr, thresholds, peak_idxs

    def auroc(self, class_reduction: str):

        try:
            if class_reduction is 'none':
                return roc_auc_score(self.target.cpu(), self.scores.cpu(), average=None)
            elif class_reduction in ['micro', 'macro']:
                return roc_auc_score(self.target.cpu(), self.scores.cpu(), average=class_reduction)
        except ValueError as e:
            print('cannot compute roc: not all classes present')
            print(e)
            return 0

    def precision(self, class_reduction: str):
        return class_reduce(self.tp, self.tp + self.fp, self.sup, class_reduction=class_reduction)
