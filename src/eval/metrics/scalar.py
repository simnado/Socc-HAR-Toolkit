import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, \
    multilabel_confusion_matrix, hamming_loss, f1_score, balanced_accuracy_score


def accuracy(y, out, threshold=0.5):
    """
    calculates accuracy: fraction of correct samples. a sample is correct if all of its labels correctly classified
    @param threshold: float or torch.Tensor with either 1 threshold for all classes or single thresholds per class
    @param out: multi label Tensor
    @param y: multi label Tensor
    @return:
    """
    out_labels = (out > threshold) * 1
    y = (y > 0.5) * 1

    num_classes = out.shape[1]
    total_score = accuracy_score(y, out_labels)
    assert total_score == ((out_labels.boo() == y.bool()).float().mean())
    class_score = [accuracy_score(y[:, cls], out_labels[:, cls]) for cls in range(num_classes)]
    avg_score = torch.Tensor(class_score).mean().item()
    return total_score, class_score, avg_score


def balanced_accuracy(y, out, threshold=0.5):
    out_labels = (out > threshold) * 1
    y = (y > 0.5) * 1
    num_classes = out.shape[1]
    class_score = [balanced_accuracy_score(y[:, cls], out_labels[:, cls]) for cls in range(num_classes)]
    avg_score = torch.Tensor(class_score).mean().item()
    return class_score, avg_score


def precision(y, out, threshold=0.5):
    out_labels = (out > threshold) * 1
    y = (y > 0.5) * 1
    micro_score = precision_score(y, out_labels, average='micro')
    macro_score = precision_score(y, out_labels, average='macro')
    class_score = precision_score(y, out_labels, average=None)
    avg_score = torch.Tensor(class_score).mean().item()
    return micro_score, macro_score, class_score, avg_score


def recall(y, out, threshold=0.5):
    out_labels = (out > threshold) * 1
    y = (y > 0.5) * 1
    micro_score = recall_score(y, out_labels, average='micro')
    macro_score = recall_score(y, out_labels, average='macro')
    class_score = recall_score(y, out_labels, average=None)
    avg_score = torch.Tensor(class_score).mean().item()
    return micro_score, macro_score, class_score, avg_score


def f1(y, out, threshold=0.5):
    out_labels = (out > threshold) * 1
    y = (y > 0.5) * 1
    micro_score = f1_score(y, out_labels, average='micro')
    macro_score = f1_score(y, out_labels, average='macro')
    class_score = f1_score(y, out_labels, average=None)
    avg_score = torch.Tensor(class_score).mean().item()
    return micro_score, macro_score, class_score, avg_score

def roc(y, out, classes: [str]):
    fpr = dict()
    tpr = dict()
    thresholds = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y[:, i], out[:, i])

    class_thresholds = [thresholds[i][np.argmax(tpr[i] - fpr[i])] for i in range(len(classes))]
    return fpr, tpr, thresholds, class_thresholds


def roc_auc(y, out):
    out_labels = out
    y = (y > 0.5) * 1
    micro, macro, cls = 0, 0, [0 for _ in y]
    try:
        micro = roc_auc_score(y, out_labels, average='micro')
        macro = roc_auc_score(y, out_labels, average='macro')
        cls = roc_auc_score(y, out, average=None)
    except ValueError:
        print('cannot compute roc: not all classes present')

    return micro, macro, cls

def global_threshold_finder(y, out):
    xs = torch.linspace(0.05, 0.95, 29)
    accuracies = [accuracy(y, out, threshold=i) for i in xs]
    return xs, accuracies

def accuracy_by_threshold(y, out, classes: [str]):
    y = (y > 0.5) * 1
    acc = torch.Tensor(len(classes), 100)
    acc_thresholds = [0.0]
    for i in range(100):
        threshold = 0.01 * (i + 1)

        for cls in range(1, len(classes)):
            acc[cls, i] = accuracy(y[:, cls:cls+1], (out[:, cls:cls+1] > threshold) * 1)[0]

    for cls in range(1, len(classes)):
        threshold_idx = torch.argmax(acc[cls, :]).item()
        threshold = threshold_idx / 100.0
        acc_thresholds.append(threshold)

    return acc, acc_thresholds


def hamming(y, out, threshold=0.5):
    out_labels = (out > threshold) * 1
    y = (y > 0.5) * 1
    return hamming_loss(y, out_labels)


# todo: calc sample_weight for true and false
def confusion_matrix(y, out, threshold=0.5):
    out_labels = (out > threshold) * 1
    y = (y > 0.5) * 1
    cms = multilabel_confusion_matrix(y, out_labels, samplewise=False)
    return cms
