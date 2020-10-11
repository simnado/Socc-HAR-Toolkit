import os
import re

import torch
from pytube import YouTube
import gdown
from urllib.error import HTTPError
from torchvision import io
from pathlib import Path


class VideoFetcher:

    def load_video(self, url: str, path: Path, res=360) -> Path:
        if path.exists():
            files = [str(vid) for vid in path.iterdir() if path.is_dir()]
            if f'{res}.mp4' in files:
                return path.joinpath(f'{res}.mp4')
            if 'complete.mkv' in files:
                return path.joinpath('complete.mkv')
            if '720p.mkv' in files:
                return path.joinpath('720p.mkv')
            if '1080.mkv' in files:
                return path.joinpath('1080p.mkv')

        if 'youtube' in url:
            return self._load_from_youtube(url, res, path)
        elif 'drive' in url:
            return self._load_from_drive(url, path)
        else:
            raise Exception('unknown video provider ')

    @staticmethod
    def load_timestamps(video_path: Path):
        pts, video_fps = io.read_video_timestamps(str(video_path))
        pts = torch.IntTensor(pts)
        return video_fps, pts

    @staticmethod
    def _load_from_youtube(url: str, res: int, path: Path) -> Path:
        try:
            video = YouTube(url)
        except BaseException as err:
            print(f'YouTube Error for {url}: {err}')
            return False

        video_sources = video.streams \
            .filter(resolution=f'{res}p', only_video=True, subtype='mp4') \
            .order_by('fps')
        if len(video_sources) == 0:
            print('no data source available for video %s' % id)
            return False

        video_src = video_sources[0]
        video_src_id = video_src.itag
        try:
            video.streams \
                .get_by_itag(video_src_id) \
                .download(output_path=str(path.absolute()), filename=f'{res}p')
        except HTTPError as err:
            print(f'YouTube Error for {url}: {err}')
            return False
        return path.joinpath(f'{res}p.mp4')

    @staticmethod
    def _load_from_drive(url: str, path: Path) -> Path:
        try:
            os.makedirs(path, exist_ok=True)
            filename = gdown.download(url, f'{path}/complete.mkv', quiet=False)
        except BaseException as err:
            print(f'Google Drive Error for {url}: {err}')
            return False
        if not filename:
            return False
        return path.joinpath('complete.mkv')
