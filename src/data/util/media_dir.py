from pathlib import Path
import re
from datetime import datetime


class MediaDir:

    def __init__(self, base_dir: str):
        self.root = Path(base_dir)

        self.root.absolute().mkdir(exist_ok=True)
        self.root.joinpath('datasets').mkdir(exist_ok=True)
        self.root.joinpath('video_metadata').mkdir(exist_ok=True)
        assert self.root.exists(), "path does not exists"

    def database(self, filename: str):
        return self.root.joinpath(f'datasets/{filename}.json')

    def datasets(self):
        return self.root.joinpath(f'datasets')

    def metadata(self, date: datetime):
        return self.root.joinpath('video_metadata', f'{date.strftime("%Y%m-%d%H-%M%S")}.pth')

    def video(self, url: str, res: int):
        matches = re.search('(youtube|drive).*\?.*[vid]{1,2}=(.*)', url)
        if len(matches.groups()) != 2:
            raise Exception('cannot parse video url ' + url)
        provider = matches.group(1)
        ext_id = matches.group(2)

        return self.root.joinpath('video', provider, ext_id, f'{res}p.mp4')
