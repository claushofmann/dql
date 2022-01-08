from typing import Tuple, Optional

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from .state import BreakoutState


class BreakoutRenderer:

    def __init__(self, breakout_env):
        super().__init__()
        self.environment = breakout_env

    def render(self, state: BreakoutState, ax=None) -> None:
        if ax is None:
            fig, ax = plt.subplots()
        # Your drawing code goes here
        ax.add_patch(plt.Circle((state.ball_pos[0],
                                 state.ball_pos[1]),
                                0.01))

        for (pos_x, pos_y), block_exists in np.ndenumerate(state.blocks):
            if block_exists:
                box = self.environment.get_block_box(pos_x, pos_y)
                ax.add_patch(self.draw_box(box))

        paddle_box = self.environment.get_paddle_box_with(state.paddle_x, self.environment.paddle_y)
        ax.add_patch(self.draw_box(paddle_box))
        # plt.show()

    def draw_box(self, box):
        return patches.Rectangle((box.x - box.w / 2, box.y - box.h / 2), box.w, box.h)
