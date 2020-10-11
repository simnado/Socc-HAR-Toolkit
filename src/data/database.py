import json
from pathlib import Path
from typing import Optional
from itertools import chain


class DatabaseHandle:

    def __init__(self, file: Path):
        """

        @type filename: object path to the json database
        """
        self.file_path = file
        self.filename = file.name

        with file.open('r') as json_file:
            file_content = json.load(json_file)
            self.taxonomy = file_content['taxonomy']
            self.database = file_content['database']
            self.classes = [node['nodeName'] for node in self.taxonomy]

        self._video2keys = dict()

        # build video2keys index
        for key, v in self.database.items():
            url = v['url']
            video_id = url.split('v=')[1] if 'youtube' in url else url.split('id=')[1]
            if video_id not in self._video2keys:
                self._video2keys[video_id] = []
            self._video2keys[video_id].append(key)

    def save(self, out_filename: Optional[str]):
        if not out_filename:
            out_filename = self.filename

        with open(out_filename, 'w') as f:
            json.dump(dict(taxonomy=self.taxonomy, database=self.database), f)

    def get_records_by_path(self, video_path: str, period: Optional[int]):
        video_id = video_path.split('/')[-2]
        records = dict()
        for key in self._video2keys[video_id]:
            match_id, _period = key.split('@')
            if not period or period == int(_period):
                records[key] = self.database[key]
        return records

    @property
    def video_urls(self):
        return list(set([data['url'] for key, data in self.database.items()]))

    @property
    def match_ids(self):
        return list(set([key.split('@')[0] for key in self.database.keys()]))

    @property
    def annotations(self):
        return list(chain(*[half['annotations'] for half in self.database.values()]))

    def get_annotations(self, split: str):
        assert split in ['train', 'val', 'test']
        return list(chain(*[half['annotations'] for half in self.database.values() if half['subset'] == split]))