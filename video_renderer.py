from env.renderer import BreakoutRenderer

from subprocess import Popen, PIPE

import matplotlib.pyplot as plt


class VideoTrajectoryRenderer:
    def __init__(self,
                 episode_record,
                 renderer: BreakoutRenderer,
                 fps: int = 30):
        self.episode_record = episode_record
        self.renderer = renderer
        self.fps = fps
        self.current_state = None

    def render(self):
        ffmpeg_process = Popen(
            ['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(self.fps), '-i', '-',
             '-vcodec', 'mpeg4', '-qscale', '5', '-r', str(self.fps), 'video.avi'], stdin=PIPE)

        for step_record in self.episode_record:
            self.renderer.render(step_record)
            plt.savefig(ffmpeg_process.stdin)
            plt.close()

        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
