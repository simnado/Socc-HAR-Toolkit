from pathlib import Path
from comet_ml.api import API


class Storage:

    def __init__(self, api_key):
        # "lL8eJl4CI5wlfif6CuU3WS4FE"
        self.comet_api = API(api_key=api_key)

    def get_checkpoints(self, experiment_path: str) -> Path:

        filename = Path(f'{experiment_path.replace("/", "_")}.ckpt')
        if experiment_path and not filename.exists():

            experiment = self.comet_api.get(experiment_path)
            assets = experiment.get_asset_list()
            ckpt = next(asset for asset in assets if '.ckpt' in asset['fileName'])

            if ckpt:
                print('download ckpt from ' + experiment_path)
                bin = experiment.get_asset(ckpt['assetId'], 'binary')
                with filename.open('wb') as file:
                    file.write(bin)
            else:
                raise Exception('no checkpoints found')

        return filename

    def get_report(self, experiment_path: str) -> Path:

        filename = Path(f'{experiment_path.replace("/", "_")}.csv')
        if experiment_path and not filename.exists():

            experiment = self.comet_api.get(experiment_path)
            assets = experiment.get_asset_list()
            reports = [asset for asset in assets if 'report.csv' in asset['fileName']]
            report = reports[-1]

            if report:
                print('download report from ' + experiment_path)
                bin = experiment.get_asset(report['assetId'], 'binary')
                with filename.open('wb') as file:
                    file.write(bin)
            else:
                raise Exception('no report found')

        return filename
