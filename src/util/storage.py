from pathlib import Path
from comet_ml.api import API
from ipywidgets import widgets

from src.util.benchmarks import benchmarks

API_KEY = "lL8eJl4CI5wlfif6CuU3WS4FE"


class Storage:

    def __init__(self):
        self.phase = widgets.Dropdown(
            options=[1, 2, 3],
            value=None,
            description='Phase:',
        )

        self.exp = widgets.Dropdown(
            options=[],
            value=None,
            description='Experiment:',
        )

        self.phase.observe(self.on_phase_change, 'value')

    def on_phase_change(self, b):
        self.exp.options = [(key, val) for key, val in benchmarks[self.phase.value - 1].items()]

    def widget(self):
        return widgets.VBox([
            self.phase,
            self.exp,
        ])

    @property
    def experiment_path(self):
        return self.exp.value


class StoredExperiment:

    def __init__(self, experiment_path):
        self.comet_api = API(api_key=API_KEY)
        self.experiment_path = experiment_path

    def get_checkpoints(self) -> Path:

        filename = Path(f'{self.experiment_path.replace("/", "_")}.ckpt')
        if self.experiment_path and not filename.exists():

            experiment = self.comet_api.get(self.experiment_path)
            assets = experiment.get_asset_list()
            ckpt = next(asset for asset in assets if '.ckpt' in asset['fileName'])

            if ckpt:
                print('download ckpt from ' + self.experiment_path)
                bin = experiment.get_asset(ckpt['assetId'], 'binary')
                with filename.open('wb') as file:
                    file.write(bin)
            else:
                raise Exception('no checkpoints found')

        return filename

    def get_report(self) -> Path:

        filename = Path(f'{self.experiment_path.replace("/", "_")}.csv')
        if self.experiment_path and not filename.exists():

            experiment = self.comet_api.get(self.experiment_path)
            assets = experiment.get_asset_list()
            reports = [asset for asset in assets if 'report.csv' in asset['fileName']]
            report = reports[-1]

            if report:
                print('download report from ' + self.experiment_path)
                bin = experiment.get_asset(report['assetId'], 'binary')
                with filename.open('wb') as file:
                    file.write(bin)
            else:
                raise Exception('no report found')

        return filename