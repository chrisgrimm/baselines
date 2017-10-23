import pygame
import gym
from gym import utils, spaces
import numpy as np
import sys
import cv2


class ContinuousGridworld(gym.Env, utils.EzPickle):
    """ The environment """

    def __init__(self, game, image_size=64, visualize=False, init_goal_coord=(0.5,0), max_steps=1000):
        utils.EzPickle.__init__(self, game, 'image')
        self.image_size = image_size
        self.visualize = visualize
        if visualize:
            self.screen = pygame.display.set_mode((image_size, image_size),)
        else:
            self.screen = pygame.Surface((image_size, image_size))
        pygame.init()
        self.goal = np.array(list(init_goal_coord))
        self.agent_position = np.array([0., 0.])
        self.agent_direction = 0.
        self.observation_space = spaces.Box(-1, 1, shape=(4*3 + 2,))
        self.action_space = spaces.Discrete(4)
        self.render_agent()
        self.max_steps = max_steps
        self.step_counter = 0.
        self.buffer = ObsBuffer(4, 64)

    def render_agent(self):
        x = (self.agent_position[0] + 1) / 2.
        y = (self.agent_position[1] + 1) / 2.
        self.screen.fill((255,255,255))
        x_int = int(x*self.image_size)
        y_int = int(y*self.image_size)
        pygame.draw.circle(self.screen, (0, 0, 0), (x_int, y_int), 5)
        if self.visualize:
            pygame.display.update()
            pygame.event.get()
        imgdata = pygame.surfarray.array3d(self.screen)
        imgdata.swapaxes(0,1)
        return imgdata

    def draw_target(self):
        x = (self.goal[0] + 1) / 2.
        y = (self.goal[1] + 1) / 2.
        x_int = int(x * self.image_size)
        y_int = int(y * self.image_size)
        pygame.draw.circle(self.screen, (255, 0, 255), (x_int, y_int), 5)
        if self.visualize:
            pygame.display.update()
            pygame.event.get()

    @property
    def pos_buffer(self):
        xy, img = self.buffer.get()
        return xy

    @property
    def normalized_goal(self):
        return self.goal.copy()

    def _step(self, action):
        forward, backward, turn_right, turn_left = range(4)
        self.step_counter += 1
        delta = 0.1
        if action == forward:
            scaled_action = np.array([delta*np.cos(self.agent_direction), delta*np.sin(self.agent_direction)])
        elif action == backward:
            scaled_action = -np.array([delta*np.cos(self.agent_direction), delta*np.sin(self.agent_direction)])
        elif action == turn_right:
            self.agent_direction = (self.agent_direction + delta) % (2*np.pi)
            scaled_action = np.array([0,0])
        else:
            self.agent_direction = (self.agent_direction - delta) % (2*np.pi)
            scaled_action = np.array([0,0])


        closeness_cutoff = 0.15
        self.agent_position = np.clip(self.agent_position + scaled_action, -1, 1)
        observation = self.render_agent()
        self.buffer.update(self.agent_position, self.agent_direction, observation.copy())
        xy, img = self.buffer.get()
        self.draw_target()

        goal_dist = np.sqrt(np.sum(np.square(self.agent_position - self.goal)))
        reward  = 1 if goal_dist < closeness_cutoff else 0
        subtask_complete = False
        old_goal = self.goal
        if reward == 1:
            self.goal = np.random.uniform(-1, 1, size=2)
            subtask_complete = True
        if self.step_counter > self.max_steps:
            subtask_complete = True
        return np.concatenate([xy, self.goal.copy()]), reward, subtask_complete, {}

    def _reset(self):
        self.buffer.reset()
        self.agent_position = np.random.uniform(-1, 1, size=2)
        self.agent_direction = 0.
        self.goal = np.random.uniform(-1, 1, size=2)
        self.step_counter = 0
        return np.concatenate([self.buffer.get()[0], self.goal.copy()])

    def _get_obs(self):
        return self.render_agent()




class ObsBuffer(object):
    def __init__(self, history_len, image_size):
        self.hl = history_len
        self.image_shape = [image_size, image_size, 3]
        self.reset()

    def update(self, xy, dir, img):
        self.xy_buffer = self.xy_buffer[1:] + [np.concatenate([xy, [dir]])]
        self.img_buffer = self.img_buffer[1:] + [img]

    def reset(self):
        self.xy_buffer = [np.zeros(3) for _ in range(self.hl)]
        self.img_buffer = [np.zeros(self.image_shape) for _ in range(self.hl)]

    def get(self):
        assert len(self.xy_buffer) == self.hl
        assert len(self.img_buffer) == self.hl
        xy = np.concatenate(self.xy_buffer, axis=0)
        img = np.concatenate([np.reshape(x, self.image_shape)
                              for x in self.img_buffer], axis=2)
        return xy, img



grid = ContinuousGridworld('grid')
print(grid.observation_space)