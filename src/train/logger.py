import torch
from pytorch_lightning.loggers import CometLogger

# todo: only works with private git
COMET_API_TOKEN = "lL8eJl4CI5wlfif6CuU3WS4FE"


class TrainLogger(CometLogger):

    def __init__(self, run_id: str, project_name: str):
        workspace, project_name = project_name.split('/')
        super().__init__(api_key=COMET_API_TOKEN,
                         workspace=workspace,
                         project_name=project_name,
                         experiment_name=run_id)
        self.run_id = run_id
        if torch.cuda.is_available():
            self.experiment.add_tag(torch.cuda.get_device_name(0))

    def log_metric(self, key, value, step):
        return self.experiment.log_metric(key, value, step=step)

    @staticmethod
    def from_existing_run(experiment_key: str):
        return CometLogger(api_key=COMET_API_TOKEN, experiment_key=experiment_key.split('/').pop())
