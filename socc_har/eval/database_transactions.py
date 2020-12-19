import json
from datetime import datetime
import os
from src.data import DatabaseHandle
import pandas as pd


class Transactions:

    def __init__(self, filename: str, save_dir='.'):
        """

        @type filename: str filename of transaction json file
        @type save_dir: str directory to be saved in
        """
        self.filename = filename
        self.save_dir = save_dir
        self.df = pd.DataFrame(
            columns=['period_id', 'url', 'src_label', 'src_segment', 'dest_label', 'dest_segment', 'verified', 'deleted', 'operation'])

        if filename and os.path.exists(self.path):
            print(f'loading previous transactions from {filename}')
            self.df = pd.read_csv(self.path)

    @property
    def path(self):
        return f'{self.save_dir}/{self.filename}'

    def integrity_check(self):
        # for transaction and db (incl subset field)
        # set 'verified': True,
        #             'deleted': False,
        pass

    def verify(self, period_id: str, url: str, label: str, segment: [int]):
        matches = self.df[(self.df.period_id == period_id) & (self.df.url == url) & (self.df.src_label == label)]
        if len(matches):
            print(f'this sample is already verified as {matches.iloc[0].operation}ed')
            return

        self.df = self.df.append({
            'period_id': period_id,
            'url': url,
            'src_label': label,
            'dest_label': label,
            'src_segment': segment,
            'dest_segment': segment,
            'operation': 'pass'
        }, ignore_index=True)
        self._save()

    def add(self, period_id: str, label: str, segment: [int]):
        matches = self.df[(self.df.period_id == period_id) & (self.df.src_segment == segment) & (self.df.dest_label == label)]
        if len(matches):
            print('this action is already added')
            return

        self.df = self.df.append({
            'period_id': period_id,
            'url': None,
            'src_label': None,
            'dest_label': label,
            'src_segment': None,
            'dest_segment': segment,
            'operation': 'add'
        }, ignore_index=True)
        self._save()

    def adjust(self, period_id: str, url: str, label: str, src_segment: [int], dest_segment: [int]):
        matches = self.df[(self.df.period_id == period_id) & (self.df.url == url) & (self.df.src_label == label)]
        if len(matches):
            print(f'this sample is already verified as {matches.iloc[0].operation}ed')
            return

        self.df = self.df.append({
            'period_id': period_id,
            'url': url,
            'src_label': label,
            'dest_label': label,
            'src_segment': src_segment,
            'dest_segment': dest_segment,
            'operation': 'edit'
        }, ignore_index=True)
        self._save()

    def remove(self, period_id: str, url: str, label: str, segment: [int]):
        matches = self.df[(self.df.period_id == period_id) & (self.df.url == url) & (self.df.src_label == label)]
        if len(matches):
            print(f'this sample is already verified as {matches.iloc[0].operation}ed')
            return

        self.df = self.df.append({
            'period_id': period_id,
            'url': url,
            'src_label': label,
            'dest_label': None,
            'src_segment': segment,
            'dest_segment': None,
            'operation': 'delete'
        }, ignore_index=True)
        self._save()

    def _save(self):
        self.integrity_check()
        self.df.to_csv(self.path)

    def apply(self, database: DatabaseHandle):

        df_left = self.df[True]
        for period_id, period_data in database.database.items():
            df = df_left[self.df.period_id == period_id]
            df_left = df_left[~(self.df.period_id == period_id)]
            period_annos = period_data['annotations']

            for index, row in df.iterrows():
                if row.operation in ['pass', 'edit', 'delete']:
                    candidates = [idx for idx, anno in period_annos if anno["url"] == row.url and anno["label"] == row.label]
                    assert len(candidates) == 1, f'{len(candidates)} record to update for key={period_id}, url={row.url}, label={row.label}'
                    record = period_annos[candidates[0]]

                    if row.operation == 'pass':
                        assert record['segment'] == row.src_segment == row.dest_segment
                        record['verified'] = True
                        record['deleted'] = False
                    elif row.operation == 'edit':
                        assert record['segment'] == row.src_segment
                        if 'source' not in record:
                            # store original segment from SBOD, only the first time
                            record['source'] = {'segment': row.src_segment}
                        record['segment'] = row.dest_segment
                        record['verified'] = True
                        record['deleted'] = False
                    elif row.operation == 'delete':
                        assert record['segment'] == row.src_segment
                        assert row.dest_segment is None
                        assert row.dest_label is None
                        record['verified'] = True
                        record['deleted'] = True
                    else:
                        raise RuntimeError('Invalid state')
                elif row.operation == 'add':
                    period_annos.append({
                        "url": None,
                        "label": row.dest_label,
                        "segment": row.dest_segment,
                        "verified": True,
                        "deleted": False
                    })
                else:
                    raise RuntimeError('Invalid state')

            # sort entries by segement
            database.database[period_id]["annotations"] = sorted(period_annos, key=lambda x: x['segment'][0], reverse=False)

        assert len(df_left) == 0, f'{len(df_left)} samples left unprocessed'

    def clear(self):
        self.df = pd.DataFrame(
            columns=['period_id', 'url', 'src_label', 'src_segment', 'dest_label', 'dest_segment', 'verified',
                     'deleted', 'operation'])
