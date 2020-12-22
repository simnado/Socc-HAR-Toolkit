from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional
import torch
from datetime import datetime

from .database import DatabaseHandle
from .util.video_converter import VideoConverter
from .util.video_fetch import VideoFetcher
from .util.media_dir import MediaDir
from .util.out_dir import OutDir


class PreProcessing:

    def __init__(self, db: DatabaseHandle, media_dir: MediaDir, out_dir: OutDir, metadata_path: Optional[Path], res=360):
        self.db = db
        self.fetch = VideoFetcher()
        self.convert = VideoConverter()
        self.not_found = set()
        self.res = res

        self.media_dir = media_dir
        self.out_dir = out_dir

        self.video_metadata = dict(
            train=dict(video_paths=[], video_fps=[], video_pts=[], sac_urls=[], sac_keys=[]),
            val=dict(video_paths=[], video_fps=[], video_pts=[], sac_urls=[], sac_keys=[]),
            test=dict(video_paths=[], video_fps=[], video_pts=[], sac_urls=[], sac_keys=[]))

        self._precomputed_video_metadata = dict(
            train=dict(video_paths=[], video_fps=[], video_pts=[], sac_urls=[], sac_keys=[]),
            val=dict(video_paths=[], video_fps=[], video_pts=[], sac_urls=[], sac_keys=[]),
            test=dict(video_paths=[], video_fps=[], video_pts=[], sac_urls=[], sac_keys=[]))

        if metadata_path and metadata_path.exists():
            self._precomputed_video_metadata = torch.load(metadata_path)
            print(f'found precomputed video metadata.')

    def prepare_data(self, verbose: bool):
        video_metadata = dict(
            train=dict(video_paths=[], video_fps=[], video_pts=[], sac_urls=[], sac_keys=[]),
            val=dict(video_paths=[], video_fps=[], video_pts=[], sac_urls=[], sac_keys=[]),
            test=dict(video_paths=[], video_fps=[], video_pts=[], sac_urls=[], sac_keys=[]))
        self.video_metadata = video_metadata

        raw_path = None

        for key, data in tqdm(self.db.database.items()):
            split = data['subset']
            url = data['url']

            if url in video_metadata[split]['sac_urls']:
                # second half of an already processed video
                index = video_metadata[split]['sac_urls'].index(url)
                video_metadata[split]['sac_keys'][index] = list(
                    {*video_metadata[split]['sac_keys'][index], key})
                if verbose:
                    print(f'[{key}] already processed')
                self.not_found.discard(url)
                continue

            out_path = self.media_dir.video(url, self.res)

            # 1) Download from provider
            if not out_path.exists():
                if verbose:
                    print(f'[{key}] load video {url}')

                raw_path = self.fetch.load_video(url, out_path.parent, self.res)
                if not raw_path:
                    if url not in self.not_found:
                        self.not_found.add(url)
                    continue

            # 2) Convert
            if not out_path.exists() and raw_path:
                if verbose:
                    print(f'[{key}] convert video {raw_path}')
                status = self.convert.resize_videos(Path(raw_path), dest_path=out_path, res=self.res)
                if status != 0:
                    print(f'[{key}] ERROR! Converting failed!')
                    continue
            elif verbose:
                print(f'[{key}] file ok {out_path}')

            # 3) Analyse
            if url not in self._precomputed_video_metadata[split]['sac_urls']:
                if verbose:
                    print(f'[{key}] analyse video {out_path}')
                fps, pts = self.fetch.load_timestamps(out_path)
            else:
                if verbose:
                    print(f'[{key}] found metadata')
                old_idx = self._precomputed_video_metadata[split]['sac_urls'].index(url)
                fps = self._precomputed_video_metadata[split]['video_fps'][old_idx]
                pts = self._precomputed_video_metadata[split]['video_pts'][old_idx]

            video_metadata[split]['video_paths'].append(str(out_path))
            video_metadata[split]['video_fps'].append(fps)
            video_metadata[split]['video_pts'].append(torch.tensor(pts))
            video_metadata[split]['sac_keys'].append([key])
            video_metadata[split]['sac_urls'].append(url)

            assert len(video_metadata[split]['video_fps']) == len(
                video_metadata[split]['video_paths']), "length of fps should match length of paths"

            assert out_path.exists(), f"File not found: {out_path}"

            assert len(video_metadata[split]['video_pts'][-1]) > 100, "analysing failed"

            # 4) Delete Download
            if raw_path and raw_path.exists() and f'{self.res}p' not in raw_path.name:
                if verbose:
                    print(f'[{key}] free storage for {raw_path}')
                raw_path.unlink()

            self.video_metadata = video_metadata

        # check for video availability
        if len(self.not_found) > 0:
            not_found_path: Path = self.out_dir.root.joinpath('not_found.txt')
            with not_found_path.open("w+") as f:
                f.writelines([url + '\n' for url in self.not_found])

            print(f'{len(self.not_found)} videos not found. a complete list is saved to `{not_found_path}`.')
        else:
            print(f'all videos found')

        return video_metadata

    def save(self):
        if not self.video_metadata:
            print('no metadata loaded. run .prepare_data()')
        metadata_out_path = self.out_dir.metadata(datetime.now())
        torch.save(self.video_metadata, metadata_out_path)
        print(f'video metadata saved to `{metadata_out_path}`')

    def reanalyze(self, split: str, key: str):
        idx = [idx for idx, keys in enumerate(self.video_metadata[split]['sac_keys']) if key in keys]
        if len(idx) != 1:
            print(f'found {len(idx)} matching entries')
            return
        idx = idx[0]

        path = self.video_metadata[split]['video_paths'][idx]
        old_fps = self.video_metadata[split]['video_fps'][idx]
        old_pts = self.video_metadata[split]['video_pts'][idx]

        print(f'[{key}] analyse video {path}')
        fps, pts = self.fetch.load_timestamps(path)

        print(f'finished analyzing: fps={fps} (was {old_fps}), {len(pts)} pts with max={pts[-1]} (was {len(old_pts)} pts with max={old_pts[-1]})')

        self.video_metadata[split]['video_fps'][idx] = fps
        self.video_metadata[split]['video_pts'][idx] = torch.tensor(pts)

