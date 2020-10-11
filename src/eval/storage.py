import os
from comet_ml.api import API


class Storage:

    def __init__(self, api_key):
        # "lL8eJl4CI5wlfif6CuU3WS4FE"
        self.comet_api = API(api_key=api_key)

    def get_checkpoints(self, experiment_path: str):

        filename = f'{experiment_path.replace("/", "_")}.ckpt'
        if experiment_path and not os.path.exists(filename):

            experiment = self.comet_api.get(experiment_path)
            assets = experiment.get_asset_list()
            ckpt = next(asset for asset in assets if '.ckpt' in asset['fileName'])

            if ckpt:
                print('download ckpt from ' + experiment_path)
                bin = experiment.get_asset(ckpt['assetId'], 'binary')
                file = open(filename, 'wb')
                file.write(bin)
                file.close()
            else:
                return False

        return filename

    def get_report(self, experiment_path: str):

        filename = f'{experiment_path.replace("/", "_")}.pt'
        if experiment_path and not os.path.exists(filename):

            experiment = self.comet_api.get(experiment_path)
            assets = experiment.get_asset_list()
            ckpt = next(asset for asset in assets if 'report.pt' in asset['fileName'])

            if ckpt:
                print('download report from ' + experiment_path)
                bin = experiment.get_asset(ckpt['assetId'], 'binary')
                file = open(filename, 'wb')
                file.write(bin)
                file.close()
            else:
                return False

        return filename
