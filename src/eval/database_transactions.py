import json
from datetime import datetime
import os
from src.data import DatabaseHandle


class Transactions:

    def __init__(self, filename: str, save_dir='.'):
        """

        @type filename: str filename of transaction json file
        @type save_dir: str directory to be saved in
        """
        self.filename = filename
        self.save_dir = save_dir
        self.insertions = dict()
        self.updates = dict()
        self.deletions = dict()

        if filename and os.path.exists(self.path):
            print('loading previous transactions')

            with open(self.path) as json_file:
                self.insertions = json_file['insertions']
                self.updates = json_file['updates']
                self.deletions = json_file['deletions']

    @property
    def path(self):
        return f'{self.save_dir}/{self.filename}'

    def add(self, period_id: str, label: str, segment: [int]):
        if period_id not in self.insertions:
            self.insertions[period_id] = []
        self.insertions[period_id].append({
            "url": datetime.now().strftime('%Y%m-%d%H-%M%S-'),
            "label": label,
            "segment": segment
        })
        self._save()

    def adjust(self, period_id: str, url: str, label: str, segment: [int]):
        if period_id not in self.updates:
            self.updates[period_id] = []
        self.updates[period_id].append({
                "url": url,
                "label": label,
                "segment": segment  # only segment will be updated
        })

    def remove(self, period_id: str, url: str, label: str):
        if period_id not in self.deletions:
            self.deletions[period_id] = []
        self.deletions[period_id].append(dict(url=url, label=label))
        self._save()

    def _save(self):
        with open(self.path, 'w') as f:
            json.dump(dict(insertions=self.insertions, updates=self.updates, deletions=self.deletions), f)

    def apply(self, database: DatabaseHandle):

        for k, v in database.database.items():
            if k in self.updates:
                for update in self.updates[k]:
                    candidates = [idx for idx, record in database.database[k] if record["url"] == update["url"] and record["label"] == update["label"]]
                    if len(candidates) == 0:
                        print(f'no record to update for key={k}, url={update["url"]}, label={update["label"]}')
                    else:
                        # update segment
                        database.database[k]["annotations"][candidates[0]]["segment"] = update["segment"]
            if k in self.deletions:
                for deletion in self.deletions[k]:
                    candidates = [idx for idx, record in database.database[k] if record["url"] == deletion["url"] and record["label"] == deletion["label"]]
                    if len(candidates) == 0:
                        print(f'no record to delete for key={k}, url={deletion["url"]}, label={deletion["label"]}')
                    else:
                        # delete record
                        database.database[k]["annotations"].pop(candidates[0])
            if k in self.insertions:
                for insertion in self.insertions[k]:
                    # add record
                    database.database[k]["annotations"].append(insertion)

            # sort entries by segement
            database.database[k]["annotations"] = sorted(database.database[k]["annotations"], key=lambda x: x['segemnt'][0], reverse=False)
