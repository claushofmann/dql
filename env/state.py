import numpy as np

class BreakoutState:
    def __init__(self, blocks, ball_pos,
                 paddle_x, ball_dir):
        self.blocks = blocks.copy()
        self.ball_pos = ball_pos
        self.paddle_x = paddle_x
        self.ball_dir = ball_dir

    def get_observation(self):
        return np.concatenate([self.blocks.reshape(-1),
                               self.ball_pos.reshape(-1), [self.paddle_x], [self.ball_dir]])

class MultiStepState:
    def __init__(self, states):
        self.states = states

    def get_observation(self):
        return np.stack(self.states, axis=-2)