from typing import Union, Tuple, Dict, Any

import numpy as np
import copy
from env.state import BreakoutState, MultiStepState


class Env:
    def step(self, action):
        pass

    def get_action_size(self):
        pass

    def get_observation_size(self):
        pass

    def reset(self):
        pass


class Box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.limits = self.__get_limits()

    def __get_limits(self):
        return np.array([[self.x - self.w / 2, self.x + self.w / 2],
                         [self.y - self.h / 2, self.y + self.h / 2]])

    def is_inside_box(self, point):
        return np.all(point > self.limits[:, 0]) and np.all(point < self.limits[:, 1])

    def angle(self, point):
        return np.arctan2(point[1] - self.y, point[0] - self.x)

    def angle2(self, point, in_angle):
        point_resized = point.reshape((1, -1))
        point_relative = np.abs(point_resized - self.limits.T)
        min_coord = np.argmin(point_relative)
        if min_coord == 0 or min_coord == 2:
            return np.pi - in_angle
        else:
            return -in_angle


class BreakoutEnv:
    def __init__(self, no_blocks, time_coef=0.05):
        super().__init__()
        self.no_blocks = no_blocks

        self.time_coef = time_coef

        self.paddle_y = 0.1
        self.paddle_width = 0.2
        self.paddle_height = 0.05
        self.blocks_start_y = 0.5
        self.blocks_end_y = 0.95

        self.block_width = 1 / no_blocks
        self.block_height = (self.blocks_end_y - self.blocks_start_y) / no_blocks

        self.create_env()


    def create_env(self):
        self.blocks = np.ones([self.no_blocks, self.no_blocks], dtype=bool)
        self.ball_pos = np.array([0.4, 0.2])
        self.ball_dir = np.random.uniform(0, np.pi)
        self.paddle_x = 0.5

    def get_paddle_box(self):
        return self.get_paddle_box_with(self.paddle_x, self.paddle_y)

    def get_paddle_box_with(self, paddle_x, paddle_y):
        return Box(paddle_x, paddle_y, self.paddle_width, self.paddle_height)

    def get_block_box(self, x_pos, y_pos):
        return Box(x_pos * self.block_width + self.block_width / 2,
                   y_pos * self.block_height + self.blocks_start_y + self.block_height / 2,
                   self.block_width,
                   self.block_height)

    def step(self, action: int):
        # action: 0: left, 1: stay, 2:right
        paddle_x = self.paddle_x + (-0.1 if action==0 else 0.1 if action==2 else 0.)
        self.paddle_x = max(0, min(1, paddle_x))

        paddle_box = self.get_paddle_box()

        reward = 0.

        if self.ball_pos[0] < 0:
            self.ball_dir = np.pi - self.ball_dir

        if self.ball_pos[0] > 1:
            self.ball_dir = np.pi - self.ball_dir

        if self.ball_pos[1] > 1:
            self.ball_dir = -self.ball_dir

        if self.ball_pos[1] < 0:
            return self.get_state(), 0., True

        if paddle_box.is_inside_box(self.ball_pos):
            #reward += 1.
            self.ball_dir = paddle_box.angle(self.ball_pos)

        for (x_pos, y_pos), block_exist in np.ndenumerate(self.blocks):
            if block_exist:
                block_box = self.get_block_box(x_pos, y_pos)
                if block_box.is_inside_box(self.ball_pos):
                    self.blocks[x_pos, y_pos] = False
                    reward += 1.
                    self.ball_dir = block_box.angle2(self.ball_pos, self.ball_dir)

        if not np.any(self.blocks):
            return self.get_state(), reward, True

        angle = self.ball_dir

        self.ball_pos = self.ball_pos + np.array([np.cos(angle) * self.time_coef * 0.5, np.sin(angle) * self.time_coef * 0.5])

        if self.ball_dir < 0.01 and self.ball_dir > 0:
            self.ball_dir = 0.01
        if self.ball_dir < 0 and self.ball_dir > -0.01:
            self.ball_dir = -0.01

        return self.get_state(), reward, False

    def get_state(self) -> BreakoutState:
        return BreakoutState(self.blocks, self.ball_pos, self.paddle_x, self.ball_dir)

    def get_action_size(self):
        return 3

    def get_observation_size(self):
        return self.get_state().get_observation().shape

    def reset(self) -> BreakoutState:
        self.create_env()
        return self.get_state()


class EnvironmentDelegatorMixin:
    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError as e:
            try:
                return self.env.__getattribute__(item)
            except AttributeError:
                raise e


class MultiStepExecutingEvironmentWrapper(Env, EnvironmentDelegatorMixin):
    def __init__(self, env, no_steps):
        self.env = env
        self.no_steps = no_steps

    def step(self, action):
        rewards = 0
        done = False
        for step in range(self.no_steps):
            if not done:
                a = action if step == 0 else 1
                state, reward, done = self.env.step(a)
                rewards += reward
        return state, rewards, done


class MultiStepExecutingEvironmentWrapperMixin:
    def __init__(self, *args, no_steps=8):
        #super(Env, self).__init__()
        super().__init__(*args)
        self.no_steps = no_steps

    def step(self, action):
        #states = []
        rewards = 0
        done = False
        for step in range(self.no_steps):
            if not done:
                a = action if step == 0 else 1
                state, reward, done = super().step(a)
                #states.append(obs.get_observation())
                rewards += reward
        #states = MultiStepState(states)
        return state, rewards, done


class BreakoutMultiStepExecutingEnvironmentWrapper(MultiStepExecutingEvironmentWrapperMixin, BreakoutEnv):
    pass