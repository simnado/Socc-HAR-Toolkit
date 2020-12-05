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
        curves = dict()
        fpr = []
        tpr = []

        if 'micro' in class_reduction:
            targets = torch.flatten(self.target)
            scores = torch.flatten(self.scores)
            fpr_m, tpr_m, thresholds_m = roc_curve(targets, scores)
            curves['micro'] = (fpr_m, tpr_m, thresholds_m, np.argmax(tpr_m - fpr_m))
        for i in class_idxs:
            fpr_c, tpr_c, thresholds_c = roc_curve(self.target[:, i], self.scores[:, i])
            curves[i] = (fpr_c, tpr_c, thresholds_c, np.argmax(tpr_c - fpr_c))
            fpr.append(fpr_c)
            tpr.append(tpr_c)
        if 'macro' in class_reduction:
            all_fpr = np.unique(np.concatenate(fpr))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.num_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= self.num_classes
            fpr_macro = all_fpr
            tpr_macro = mean_tpr
            curves['macro'] = (fpr_macro, tpr_macro, [0 for _ in fpr_macro], np.argmax(tpr_macro - fpr_macro))

        return curves

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
