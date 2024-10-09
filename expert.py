import abc
import os
import pickle
import stable_baselines3
import numpy as np

from PIL import Image
from copy import deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm


from RLA import logger


class Data_Collector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect(self, num_steps):
        pass

    @abc.abstractmethod
    def save(self, data_dir):
        pass

    @abc.abstractmethod
    def load(self, data_dir):
        pass



class ManualDataCollector():
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
                action = agent.predict(obs)

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

class ExpertDataCollector(Data_Collector):
    def __init__(self, env, agent, args) -> None:
        self.env = deepcopy(env)  # Create vectorized environment
        self.save_dir = args.data_dir  # Directory to save the collected data
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.agent = agent  # The agent to interact with the environment
        self.epsilon = args.epsilon  # Exploration rate (epsilon)
        self.reset_collection = args.reset_collection  # Whether to reset the environment for every collection
        self.obses = []  # List to store observations
        self.actions = []  # List to store actions
        self.rewards = []
        self.ends = []
        self.length = 0  # Counter for total data points collected
    
    def info(self):
        final_reward = []
        r_episode = 0
        for r, e in zip(self.rewards, self.ends):
            r_episode += r
            if e:
                final_reward.append(r_episode)
                r_episode = 0
        
        data_m, data_std = np.mean(final_reward), np.std(final_reward)
        
        logger.info(f"dataset info: mean {data_m}, std: {data_std} ")
                


    def collect(self, num_steps=None, target_buffer_step=None):
        """
        Collect data by interacting with the environment, and store in `self.obses` and `self.actions`.
        """
        if num_steps is None:
            num_steps = target_buffer_step - self.length


        obs, info = self.env.reset()  
        epsilon = self.epsilon

        logger.info(f"Start collecting {num_steps} steps")
        for _ in tqdm(range(num_steps), desc="data collection"):
            # Epsilon-greedy strategy to select action
            if np.random.rand() < epsilon:
                action = self.env.action_space.sample()  # Choose a random action
            else:
                action = self.agent.predict(obs)  # Choose an action predicted by the agent

            obs_next, reward, done, truncated, _ = self.env.step(action)  # Step in the environment

            # Store the current observation and action
            self.obses.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.ends.append((done or truncated))

            # Update the current observation
            obs = obs_next
            self.length += 1  # Increase the counter

            # Reset environment if the episode is done
            if done or truncated:
                obs, _ = self.env.reset()


    def save(self, filename=None):
        """
        Save the collected data. The filename includes the current epsilon value.
        """
        if filename is None:
            filename = os.path.join(self.save_dir, f"expert_data_epsilon{self.epsilon:.1f}.pkl")
        
        data = {
            'obses': self.obses,
            'actions': self.actions,
            'length': self.length,
            'rewards': self.rewards,
            'ends': self.ends,
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to {filename}")

    def load(self, filename=None):
        """
        Load previously saved data.
        """
        if filename is None:
            filename = os.path.join(self.save_dir, f"expert_data_epsilon{self.epsilon:.1f}.pkl")
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.obses = data['obses']
            self.actions = data['actions']
            self.length = data['length']
            self.rewards = data["rewards"]
            self.ends = data["ends"]
            logger.info(f"Data loaded from {filename}")
        else:
            logger.info(f"Previous data unexist in: {filename} ")

    def sample(self, batch_size=None, ratio=1):
        """
        Sample a batch of observations and actions from the collected data.
        """
        assert ratio > 0 and ratio <= 1
        if batch_size is None:
            batch_size = self.length * ratio

        indices = np.random.choice(self.length, batch_size, replace=False)
        sampled_obses = [self.obses[i] for i in indices]
        sampled_actions = [self.actions[i] for i in indices]
        return sampled_obses, sampled_actions
    