import gymnasium as gym
from stable_baselines3 import PPO, DQN  # 你可以换成其他算法，如 DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper

from utils import RenderGymEnv


# 创建Montezuma's Revenge环境
def make_env():
    env = gym.make('MontezumaRevengeNoFrameskip-v0', render_mode="rgb_array") # "rgb_array"
    env = AtariWrapper(env)  # 使用 AtariWrapper 进行环境预处理
    return env

if __name__ == "__main__":
    # 使用 DummyVecEnv 和 VecFrameStack 包装环境
    env = DummyVecEnv([make_env])  # 单环境包装成矢量化环境
    # env = RenderGymEnv(env, use_plt=True)  # 单环境包装成矢量化环境
    env = VecFrameStack(env, n_stack=4)  # 堆叠 4 帧作为输入

    # 创建PPO模型，当然你也可以使用DQN、A2C等算法
    # model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./ppo_montezuma_tensorboard/")
    model = DQN('CnnPolicy', env, verbose=1, tensorboard_log="./dqn_montezuma_tensorboard/", 
                buffer_size=10000, learning_starts=50000, target_update_interval=1000, 
                train_freq=4, gradient_steps=1, exploration_fraction=0.1, exploration_final_eps=0.01,
                learning_rate=1e-4, gamma=0.99)

    # 渲染频率设为每 10000 个 steps 渲染一次
    render_callback = EvalCallback(env, render=False, eval_freq=10000)

    # 训练模型
    model.learn(total_timesteps=1000000, callback=render_callback)

    # 保存模型
    model.save("ppo_montezuma")

    # 测试模型表现
    obs = env.reset()
    for i in range(10000):  # 测试10000步
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()  # 渲染每一步的表现
