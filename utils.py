import numpy as np
from PIL import Image
import os
import gym
import matplotlib.pyplot as plt
import random

import torch
from torch import inf
from torch.utils.data import DataLoader, TensorDataset

from RLA import logger, time_step_holder, exp_manager

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

def set_seed_everywhere(seed):
    """
    Set seed for reproducibility across different random modules.
    
    Args:
        seed (int): The seed value to set for all relevant random modules.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy random
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in GPU operations
    torch.backends.cudnn.benchmark = False  # Disable cudnn auto-tuning for deterministic results

def get_device():
    """
    Returns the device to be used for PyTorch operations.
    If a GPU is available, it returns 'cuda', otherwise it returns 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env

def plot(record):
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(record['steps'], record['mean'],
	        color='blue', label='reward')
	ax.fill_between(record['steps'], record['min'], record['max'],
	                color='blue', alpha=0.2)
	ax.set_xlabel('number of steps')
	ax.set_ylabel('Average score per episode')
	ax1 = ax.twinx()
	ax1.plot(record['steps'], record['query'],
	         color='red', label='query')
	ax1.set_ylabel('queries')
	reward_patch = mpatches.Patch(lw=1, linestyle='-', color='blue', label='score')
	query_patch = mpatches.Patch(lw=1, linestyle='-', color='red', label='query')
	patch_set = [reward_patch, query_patch]
	ax.legend(handles=patch_set)
	fig.savefig('performance.png')


# class RenderGymEnv(gym.Wrapper):
#     def __init__(self, env, img_dir='imgs', use_plt=False):
#         super().__init__(env)
#         self.img_dir = img_dir
#         os.makedirs(self.img_dir, exist_ok=True)  # Create directory to save images
#         self.img_count = 0  # To ensure unique filenames
#         self.use_plt = use_plt

#     def render(self, **kwargs):
#         env = self.unwrapped
#         image = env.get_images()[0]
        
#         if self.use_plt:
#              # 使用 matplotlib 显示图像
#             plt.imshow(image)
#             plt.title(f"Step: {self.img_count}")
#             plt.axis('off')  # 不显示坐标轴
#             plt.show()  
#         else:
#             # Save the frame as a JPEG image
#             img_path = os.path.join(self.img_dir, f'ppo_{self.img_count}.jpeg')
            
#             # Convert the NumPy array (frame) to a PIL Image and save it
#             im = Image.fromarray(image)
#             im.save(img_path)
            
#             print(f"Image saved at {img_path}")
        
#         self.img_count += 1  # Increment image count for unique file names

def plot_and_save_fig(env, img_path):
    env = unwrap(env)
    image = env.render()
    # Convert the NumPy array (frame) to a PIL Image and save it
    im = Image.fromarray(image)
    im.save(img_path)
    print(f"Image saved at {img_path}")

def evaluate_episode(agent, env, eval_episode=5, time_step=-1, log_prefix="eval"):
    """
    Evaluate the agent over multiple episodes and log the mean and standard deviation of rewards.

    Args:
        agent: The agent being evaluated, which has a policy for selecting actions.
        env: The environment in which the agent is evaluated.
        eval_episode: Number of evaluation episodes (default: 5).
        logger: Logger for logging key-value pairs (optional).

    Returns:
        float: The average reward obtained over all episodes.
    """
    rewards = []  # List to store total reward for each episode
    
    for episode in range(eval_episode):
        state, info = env.reset()  # Reset the environment for a new episode
        episode_reward = 0.0
        done, truncated = False, False
        step = 0
        
        while not (done or truncated):
            action = agent.predict(state)  # Agent selects an action
            next_state, reward, done, truncated, _ = env.step(action)  # Environment responds
            episode_reward += reward  # Accumulate reward for this episode
            state = next_state  # Move to the next state

            step += 1
            if step % 1000 == 0:
                print(f"step: {step}, reward:{episode_reward}")
                plot_and_save_fig(env=env, img_path=os.path.join(exp_manager.results_dir, f"{step}.jpeg"))
        
        rewards.append(episode_reward)  # Append the total reward for this episode
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # If a logger is provided, log the mean and standard deviation of rewards
    logger.logkv(f'{log_prefix}/mean_reward', mean_reward)
    logger.logkv(f'{log_prefix}/std_reward', std_reward)
    logger.logkv(f'{log_prefix}/episodes', eval_episode)
    time_step_holder.set_time(time_step)
    logger.dumpkvs()  # Dump all key-value logs
    
    return mean_reward

def get_dataloader(data, label, data_type, label_type, bs):
    if isinstance(data, list):
        data = np.stack(data, axis=0)
    if isinstance(label, list):
        label = np.stack(label, axis=0)
    data_batch = torch.from_numpy(data).to(data_type)
    label_batch = torch.from_numpy(label).to(label_type)
    dataset = TensorDataset(data_batch, label_batch)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    return data_loader



def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm
