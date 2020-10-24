from pathlib import Path
from typing import Optional
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ProgressBar
from pytorch_lightning import Trainer as LightningTrainer
from src.train.callbacks.reporter import Reporter


class Trainer(LightningTrainer):
    def __init__(self, run_id: str, working_dir: Path, **kwargs):
        self.run_id = run_id
        self.out_dir = working_dir
        self.root_dir = self.out_dir.joinpath(f'run/{run_id}')
        self.root_dir.mkdir(exist_ok=True, parents=True)

        self.early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=True)

        self.lr_logger = LearningRateMonitor()
        self.checkpointing = ModelCheckpoint(monitor='val_loss', filepath=str(self.root_dir), save_top_k=2,
                                             verbose=True)
        self.progress = ProgressBar()
        self.reporter = Reporter(self.out_dir)

        super().__init__(default_root_dir=self.root_dir,
                         checkpoint_callback=self.checkpointing,
                         callbacks=[self.lr_logger, self.progress, self.reporter, self.early_stopping],
                         **kwargs)

    def save(self, save_dir: Optional[str] = None, filename: Optional[str] = None, local=False):
        """
        saves the best checkpoints
        @return: the local path and url to checkpoint file
        """

        if save_dir is None:
            save_dir = self.root_dir

        model_path = Path(self.checkpointing.best_model_path)
        res = self.logger.experiment.log_asset(model_path)

        if filename is None:
            filename = f'{self.run_id}_ep{self.current_epoch}_{res["assetId"]}.pt'

        if local:
            print(f'{save_dir}/{filename}')
            model_path.rename(f'{save_dir}/{filename}')

        # todo: get filename
        #report = f'{self.out_dir}/report.pt'
        #if os.path.exists(report):
        #    res2 = self.logger.experiment.log_asset(report)
        #    if local:
        #        print(f'{save_dir}/report.pt')
        #        os.rename(report, f'{save_dir}/report_{res2["assetId"]}.pt')
        #else:
        #    print('report not found: {report} does not exist')
