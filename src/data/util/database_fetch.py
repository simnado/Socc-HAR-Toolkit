from urllib import request
from tqdm.auto import tqdm
from pathlib import Path


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class DatabaseFetcher:

    @staticmethod
    def load(version: str, dest_path: Path):

        database_url = f'https://gitlab.com/socc-har/socc-har-32/-/raw/master/data/{version}.json'

        if not dest_path.exists():
            with DownloadProgressBar(unit='B', unit_scale=True,
                                     miniters=1, desc=database_url.split('/')[-1]) as t:
                opener = request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                request.install_opener(opener)
                request.urlretrieve(database_url, filename=str(dest_path), reporthook=t.update_to)

