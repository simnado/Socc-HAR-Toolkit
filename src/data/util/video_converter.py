import os
import subprocess
import sys
from pathlib import Path


class VideoConverter:
    """
    stolen from open-mmlab/mmaction2
    """

    def resize_videos(self, src_path: Path, dest_path: Path, res=360, remove_dup=False, dense=False):
        """Generate resized video cache.
        Args:
            vid_item (list): Video item containing video full path,
                video relative path.
        Returns:
            bool: Whether generate video cache successfully.
        """
        result = os.popen(
            f'ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "{src_path}"'
            # noqa:E501
        )
        w, h = [int(d) for d in result.readline().rstrip().split(',')]
        if w > h:
            cmd = (f'ffmpeg -hide_banner -loglevel error -i "{src_path}" '
                   f'-vf {"mpdecimate," if remove_dup else ""}'
                   f'scale=-1:{res},fps=25 '
                   f'{"-vsync vfr" if remove_dup else ""} '
                   f'-c:v libx264 {"-g 16" if dense else ""} '
                   f'-an "{dest_path}" -y')
        else:
            cmd = (f'ffmpeg -hide_banner -loglevel error -i "{src_path}" '
                   f'-vf {"mpdecimate," if remove_dup else ""}'
                   f'scale=-1:{res},fps=25 '
                   f'{"-vsync vfr" if remove_dup else ""} '
                   f'-c:v libx264 {"-g 16" if dense else ""} '
                   f'-an "{dest_path}" -y')

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()

        # This makes the wait possible
        p_status = p.wait()

        # This will give you the output of the command being executed
        print("Command output: " + output.decode('utf-8'))
        if p_status != 0:
            print(f'ERROR: {err.decode("utf-8")}')
        #os.popen(cmd)
        print(f'encoded {dest_path}')
        sys.stdout.flush()
        return p_status
