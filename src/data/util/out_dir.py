from datetime import datetime
from pathlib import Path


class OutDir:

    def __init__(self, base_dir: str):
        self.root = Path(base_dir)

        self.root.absolute().mkdir(exist_ok=True)
        self.root.joinpath('stats').mkdir(exist_ok=True)
        self.root.joinpath('samples').mkdir(exist_ok=True)
        assert self.root.exists(), "path does not exists"

    def stats(self):
        return self.root.joinpath('stats')

    def sample(self):
        return self.root.joinpath('samples')

    def metadata(self, date: datetime):
        return self.root.joinpath('video_metadata', f'{date.strftime("%Y%m-%d%H-%M%S")}.pth')
