import torch
from src.data import HarDataset


class DataStats:

    def __init__(self, split: str, dataset: HarDataset, resample_class_limit: int, seed=94):
        self.split = split
        self.dataset = dataset
        self.resample_class_limit = resample_class_limit

        self.pairwise_occs = None
        self.actions = [0 for _ in self.dataset.classes]
        self.samples = [0 for _ in self.dataset.classes]
        self.resamples = [0 for _ in self.dataset.classes]
        self.ratios = [0 for _ in self.dataset.classes]
        self.background_samples = 0
        self.overlap_samples = 0
        self.background_ratio = 0
        self.weights = None
        self.indices = None

        self.seed = seed

        self._analyze()

    def _analyze(self):
        torch.manual_seed(self.seed)

        num_classes = len(self.dataset.classes)

        # collect actions from annotations
        actions = {cls: 0 for cls in self.dataset.classes}
        for anno in self.dataset.database.get_annotations(self.split):
            actions[anno['label']] += 1
        self.actions = [actions[cls] for cls in self.dataset.classes]

        # collect samples from samples
        self.samples = torch.sum(self.dataset.y, dim=0).tolist()

        summed = torch.sum(self.dataset.y, dim=1)
        self.background_samples = torch.sum(summed == 0).item()
        self.overlap_samples = torch.sum(summed > 1).item()

        # calc ratio per class
        self.ratios = [self.samples[cls_idx] / len(self.dataset.y) for cls_idx in range(num_classes)]
        self.background_ratio = self.background_samples / len(self.dataset.y)

        # calc weights per class
        # weight is avg of background_ratio / class_ratio
        factors = self.dataset.y * self.background_ratio / torch.tensor(self.ratios)
        row_sum = torch.sum(self.dataset.y, dim=1)
        weights = torch.sum(factors, dim=1) / torch.max(row_sum, torch.ones(row_sum.shape))
        self.weights = torch.max(weights, torch.ones(weights.shape))

        num_samples = sum([self.resample_class_limit] + [min(self.resample_class_limit, self.samples[cls_idx]) for cls_idx, _ in enumerate(self.dataset.classes)])
        self.indices = torch.multinomial(self.weights, int(num_samples))
        print(f'sample {len(self.indices)}/{len(self.dataset)} clips')

        y = self.dataset.y[self.indices]
        self.resamples = torch.sum(y, dim=0)

        # iterate pairwise
        self.pairwise_occs = torch.zeros([num_classes, num_classes])

        for row in range(num_classes):
            for col in range(num_classes):
                filtered = self.dataset.y[:, [row, col]]
                summed = torch.sum(filtered, dim=1)
                relevant = summed > 1
                self.pairwise_occs[row, col] = torch.sum(relevant)
