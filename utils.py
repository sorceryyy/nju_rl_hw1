import numpy as np
import stable_baselines3
from PIL import Image
import abc

from copy import deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def human_play(envs, action_shape):
    obs = envs.reset()
    while True:
        im = Image.fromarray(obs)
        im.save('imgs/' + str('screen') + '.jpeg')
        action = int(input('input action'))
        while action < 0 or action >= action_shape:
            action = int(input('re-input action'))
        obs_next, reward, done, _ = envs.step(action)
        obs = obs_next
        if done:
            obs = envs.reset()

class Data_Collector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect(self, num_steps):
        pass

class Manual_Data_Collector():
    def __init__(self, envs, agent,) -> None:
        self.envs = deepcopy(envs)
        self.agent = agent

    def collect(self, num_steps, save_img=True):
        """
        Returns:
            data: List[obs]
            label: List[actions]
        """
        envs = self.envs
        agent = self.agent
        data, label = [], []
        obs = envs.reset()
        # we get a trajectory with the length of args.num_steps
        for step in range(num_steps):
            # Sample actions
            epsilon = 0.05
            if np.random.rand() < epsilon:
                # we choose a random action
                action = envs.action_space.sample()
            else:
                # we choose a special action according to our model
                action = agent.select_action(obs)

            # interact with the environment
            # we input the action to the environments and it returns some information
            # obs_next: the next observation after we do the action
            # reward: (float) the reward achieved by the action
            # down: (boolean)  whether it’s time to reset the environment again.
            #           done being True indicates the episode has terminated.
            obs_next, reward, done, _ = envs.step(action)
            # we view the new observation as current observation
            obs = obs_next
            # if the episode has terminated, we need to reset the environment.
            if done:
                envs.reset()

            # an example of saving observations
            if save_img:
                im = Image.fromarray(obs)
                im.save('imgs/' + str(step) + '.jpeg')
            data.append(obs)

        # You need to label the images in 'imgs/' by recording the right actions in label.txt

        # After you have labeled all the images, you can load the labels
        # for training a model
        with open('/imgs/label.txt', 'r') as f:
            for label_tmp in f.readlines():
                label.append(label_tmp)
        
        return data, label

class SB3_Data_Collector():
    def __init__(self, env, epsilon=0.3, reset_every_collection=False) -> None:
        self.envs = make_vec_env(deepcopy(env))
        self.agent = PPO("CnnPolicy", self.envs, verbose=1)
        self.epsilon = epsilon
        self.prev_obs = None
        self.reset_every_collection = reset_every_collection

    def collect(self, envs, num_steps):
        envs = self.envs
        agent = self.agent
        data, label = [], []
        if self.reset_every_collection and self.prev_obs is not None:
            obs = envs.reset()
        else:
            obs = self.prev_obs

        epsilon = self.epsilon
        for step in range(num_steps):
            epsilon = 0.05
            if np.random.rand() < epsilon:
                # we choose a random action
                action = envs.action_space.sample()
            else:
                # we choose a special action according to our model
                action = agent.predict(obs)
            obs_next, reward, done, _ = envs.step(action)
            data.append(obs)
            label.append(action)
            # we view the new observation as current observation
            obs = obs_next
            # if the episode has terminated, we need to reset the environment.
            if done:
                envs.reset()
        
        self.prev_obs = obs

        return data, label


    def model_lean(self, timesteps):
        self.agent.learn(total_timesteps=25000)