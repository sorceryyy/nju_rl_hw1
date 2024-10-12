import abc
import os
import pickle
import stable_baselines3
import numpy as np
import torch

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Generator, NamedTuple, Optional, Union
from gym import spaces
import torch as th
from sklearn.model_selection import train_test_split
from RLA import logger, time_step_holder

from PIL import Image
from copy import copy, deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.vec_env import VecNormalize
from tqdm import tqdm
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from utils import turn_grey

from utils import get_dataloader

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None



class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]], env: Optional[VecNormalize] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class Trainer(metaclass=abc.ABCMeta):
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

class DaggerTrainer(Trainer):
    def __init__(self, env, agent, expert_agent, args) -> None:
        self.env = deepcopy(env)  # Create vectorized environment
        self.save_dir = args.data_dir  # Directory to save the collected data
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.agent = agent  # The agent to interact with the environment
        self.expert_agent = expert_agent
        self.epsilon = args.epsilon  # Exploration rate (epsilon)
        self.max_explore_epsilon = args.max_explore_epsilon
        self.max_buffer_size = args.max_buffer_size

        # Arrays to store observations, actions, rewards, ends, etc.
        self.obses = None
        self.actions = None
        self.expert_actions = None
        self.rewards = None
        self.ends = None
        self.length = 0  # Counter for total data points collected
        self.train_bs = args.train_bs
        self.eval_bs = args.eval_bs

    def info(self):
        final_reward = []
        r_episode = 0
        traj_num = 0
        for r, e in zip(self.rewards, self.ends):
            r_episode += r
            if e:
                final_reward.append(r_episode)
                r_episode = 0
                traj_num += 1
        
        data_m, data_std = np.mean(final_reward), np.std(final_reward)
        
        logger.info(f"dataset info: traj num {traj_num}, mean {data_m}, std: {data_std} ")
    
    def drop_buffer(self, drop_num):
            # Ensure that we don't drop more elements than we have
        if drop_num >= len(self.obses):
            raise ValueError(f"Cannot drop {drop_num} elements from buffer, as it only contains {len(self.obses)} elements.")

        # Slice each buffer to drop the first `drop_num` elements
        self.obses = self.obses[drop_num:]
        self.actions = self.actions[drop_num:]
        self.expert_actions = self.expert_actions[drop_num:]
        self.rewards = self.rewards[drop_num:]
        self.ends = self.ends[drop_num:]

    def collect(self, num_steps=None, target_buffer_step=None, time_step=0):
        """
        Collect data by interacting with the environment, and store in `self.obses` and `self.actions`.
        """
        rewards = []  # List to store total reward for each episode
        episode_reward = 0.0
        if num_steps is None:
            num_steps = max(target_buffer_step - self.length, 0)

        obs, info = self.env.reset()
        assert time_step - 1 >= 0
        epsilon = min(self.epsilon**(time_step-1), self.max_explore_epsilon)

        # Ensure buffer size does not exceed max buffer size
        if num_steps + self.length > self.max_buffer_size:
            self.drop_buffer(num_steps + self.length - self.max_buffer_size)

        logger.info(f"Start collecting {num_steps} steps")
        for _ in tqdm(range(num_steps), desc="data collection"):
            # Epsilon-greedy strategy to select action
            action = self.agent.predict(obs)  # Choose an action predicted by the agent
            expert_action = self.expert_agent.predict(obs)
            if np.random.rand() < epsilon:
                final_action = expert_action  # Choose a random action
            else:
                final_action = action  # Choose a random action

            obs_next, reward, done, truncated, _ = self.env.step(final_action)  # Step in the environment
            episode_reward += reward  # Accumulate reward for this episode

            # Initialize lists if empty
            if self.length == 0:
                self.obses = [copy(obs[-1])]  # List of observations
                self.actions = [action]  # List of actions
                self.expert_actions = [expert_action]  # List of expert actions
                self.rewards = [reward]  # List of rewards
                self.ends = [done or truncated]  # List of done flags
            else:
                # Append new observation, action, expert action, reward, and end flag to lists
                self.obses.append(copy(obs[-1]))
                self.actions.append(action)
                self.expert_actions.append(expert_action)
                self.rewards.append(reward)
                self.ends.append(done or truncated)

            # Update the current observation
            obs = obs_next
            self.length += 1  # Increase the counter

            # Reset environment if the episode is done
            if done or truncated:
                obs, _ = self.env.reset()
                rewards.append(episode_reward)
                episode_reward = 0.0
        
        mean_reward = np.mean(rewards) if rewards else -1 # ensure not empty
        std_reward = np.std(rewards) if rewards else -1
        logger.logkv(f"collect/r_m", mean_reward)
        logger.logkv(f"collect/r_std", std_reward)
        time_step_holder.set_time(time_step)
        logger.dumpkvs()


    # def collect_array(self, num_steps=None, target_buffer_step=None):
    #     """
    #     Collect data by interacting with the environment, and store in `self.obses` and `self.actions`.
    #     """
    #     if num_steps is None:
    #         num_steps = target_buffer_step - self.length

    #     obs, info = self.env.reset()  
    #     epsilon = self.epsilon

    #     if num_steps + self.length > self.max_buffer_size:
    #         self.drop_buffer(num_steps + self.length - self.max_buffer_size)

    #     logger.info(f"Start collecting {num_steps} steps")
    #     for _ in tqdm(range(num_steps), desc="data collection"):
    #         # Epsilon-greedy strategy to select action
    #         if np.random.rand() < epsilon:
    #             action = self.env.action_space.sample()  # Choose a random action
    #         else:
    #             action = self.agent.predict(obs)  # Choose an action predicted by the agent
    #         expert_action = self.expert_agent.predict(obs)

    #         obs_next, reward, done, truncated, _ = self.env.step(action)  # Step in the environment

    #         if self.length == 0:
    #             self.obses, self.actions, self.expert_actions, self.rewards, self.ends = \
    #                 np.expand_dims(copy(obs[-1]), axis=0), np.array([action]), np.array([expert_action]), np.array([reward]), np.array([done or truncated])
    #         else:
    #             # Store the current observation, action, reward, and end flag
    #             self.obses = np.concatenate([self.obses, np.expand_dims(copy(obs[-1]), axis=0)], axis=0)  # Append new observation
    #             self.actions = np.concatenate([self.actions, np.array([action])], axis=0)  # Append new action
    #             self.expert_actions = np.concatenate([self.expert_actions, np.array([expert_action])], axis=0)
    #             self.rewards = np.concatenate([self.rewards, np.array([reward])], axis=0)  # Append new reward
    #             self.ends = np.concatenate([self.ends, np.array([done or truncated])], axis=0)  # Append new end flag

    #         # Update the current observation
    #         obs = obs_next
    #         self.length += 1  # Increase the counter

    #         # Reset environment if the episode is done
    #         if done or truncated:
    #             obs, _ = self.env.reset()
    
    def train(self, epoch, train_ratio=1):
        self._construct_traj_len()
        traj_data, traj_label = self.sample()
        train_data, eval_data, train_label, eval_label = train_test_split(
            traj_data, traj_label, test_size=1-train_ratio
        )
        train_dataloader = get_dataloader(train_data, train_label, data_type=torch.float32, label_type=torch.long, bs=self.train_bs)
        eval_dataloader = get_dataloader(eval_data, eval_label, data_type=torch.float32, label_type=torch.long, bs=self.eval_bs)
        train_epoch_loss = self.agent.train(train_dataloader)
        logger.logkv("train/epoch_loss", train_epoch_loss)
        time_step_holder.set_time(epoch)
        logger.dumpkvs()

        # eval
        eval_loss, eval_accuracy = self.agent.evaluate(eval_dataloader)
        logger.logkv("eval/epoch_loss", eval_loss)
        logger.logkv("eval/epoch_accuracy", eval_accuracy)
        time_step_holder.set_time(epoch)
        logger.dumpkvs()


    def save(self, filename=None):
        """
        Save the collected data. The filename includes the current epsilon value.
        """
        if filename is None:
            filename = os.path.join(self.save_dir, f"expert_data_epsilon{self.epsilon:.1f}.pkl")
        
        data = {
            'obses': self.obses,
            'actions': self.actions,
            'expert_actions': self.expert_actions,
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
            self.expert_actions = data['expert_actions']
            self.length = data['length']
            self.rewards = data["rewards"]
            self.ends = data["ends"]
            logger.info(f"Data loaded from {filename}")
        else:
            logger.info(f"Previous data unexist in: {filename} ")


    def get_history(self, obs_index, horizon):
        # Get the historical observations for the given index
        start_idx = self.traj_starts[obs_index]
        first_obs = self.obses[start_idx]
        padding = [first_obs] * max(0, horizon - (obs_index - start_idx + 1))
        real_obses = self.obses[max(obs_index - horizon + 1, start_idx): obs_index+1]
        return np.stack(padding + real_obses, axis=0)

    def _construct_traj_len(self):
        traj_start = 0
        self.traj_starts = []
        for idx, end in enumerate(self.ends):
            self.traj_starts.append(traj_start)
            if end:
                traj_start = idx


    def sample(self, batch_size=None, ratio=1, horizon=4):
        """
        Sample a batch of observations and actions from the collected data.
        """
        assert ratio > 0 and ratio <= 1
        if batch_size is None:
            batch_size = int(self.length * ratio)

        logger.info(f"Datasize: {batch_size}")
        indices = np.random.choice(self.length, batch_size, replace=False)

         # Use list comprehension to extract samples by index
        sampled_obses = [self.get_history(i, horizon=horizon) for i in indices]
        sampled_actions = [self.expert_actions[i] for i in indices]
        # sampled_obses = self.obses[indices]
        # sampled_actions = self.expert_actions[indices]
        return sampled_obses, sampled_actions