from pathlib import Path
from pytorch_lightning import Callback
import torch


class Reporter(Callback):

    def __init__(self, out_dir: Path):
        """
        Args:
            out_dir: Path
        """
        super().__init__()
        self.out_dir = out_dir

        # todo: as dataframe
        self.report = dict(experiment=None, epochs=0, train=[], val=[], test=[], loc_test=[])
        self.report_file = f'{self.out_dir}/report.pt'
        #torch.save(self.report, self.report_file)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        self._save_worst_samples('train', outputs.y, outputs.scores, outputs.lossses, outputs.ids)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._save_worst_samples('val', outputs.y, outputs.scores, outputs.lossses, outputs.ids)

    def on_validation_epoch_end(self, trainer, pl_module, outputs):
        print(outputs)  # todo: is outputs set?

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._save_worst_samples('test', outputs.y, outputs.scores, outputs.lossses, outputs.ids)

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        pass

    def _save_worst_samples(self, context: str, y, scores, losses, ids):
        # copy to work with half precision
        print(context)
        print(y.shape)
        print(ids.shape)
        print('---')

        worst_idx = torch.argsort(losses, descending=True)

        # todo:
        #self.report['epochs'] = self.current_epoch + 1
        #self.report[context].append(dict(id=ids[worst_idx], pred=pred[worst_idx], loss=losses[worst_idx], y=y[worst_idx]))
        #torch.save(self.report, self.report_file)