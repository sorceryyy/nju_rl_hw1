import gym
import numpy as np
import cv2
from PIL import Image
from collections import deque

from copy import copy

from utils import unwrap


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms))

        if done or truncated:
            self.visited_rooms.clear()
        return obs, rew, done, truncated, info

    def reset(self):
        return self.env.reset()

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render=False, num_stacks=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = num_stacks
        self.is_render = is_render

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            if self.is_render:
                self.env.render()
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class Env(gym.Wrapper):
    def __init__(self, env_name, num_stacks, history_size=4, h=84, w=84):
        self.env = MaxAndSkipEnv(gym.make(env_name, max_episode_steps=5000, render_mode="rgb_array"), num_stacks=num_stacks)
        if 'Montezuma' in env_name:
            self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in env_name else 1)

        # num_stacks: the agent acts every num_stacks frames
        self.num_stacks = num_stacks
        self.history_size = history_size
        self.h = h  # height of preprocessed image
        self.w = w  # width of preprocessed image

        # Initialize history with zeros: shape [history_size, h, w]
        self.history = np.zeros([self.history_size, h, w], dtype=np.float32)

        # Save observation space and action space from original environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        reward_sum = 0
        done = False
        obs_next = None

        obs_next, reward, done, truncated, info = self.env.step(action)
        reward_sum += reward

        # Preprocess the observation and update history
        obs_next = self.pre_proc(obs_next)
        self.history[:self.history_size-1, :, :] = self.history[1:, :, :]
        self.history[self.history_size-1, :, :] = obs_next

        return self.history, reward_sum, done, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        # Preprocess the observation and fill the history with it
        obs = self.pre_proc(obs)
        self.history = np.stack([obs] * self.history_size, axis=0)
        return self.history, info

    def pre_proc(self, obs):
        """
        Preprocess the observation by converting it to grayscale and resizing it.
        """
        # Convert to grayscale using PIL
        obs = np.array(Image.fromarray(obs).convert('L')).astype('float32')
        # Resize the observation to target height and width using OpenCV
        obs = cv2.resize(obs, (self.h, self.w))
        # Normalize the observation to [0, 1] range
        obs /= 255.0
        return obs

