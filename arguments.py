import argparse

# import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--info', type=str, default="")
    parser.add_argument('--env-name', type=str, default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--num-stacks', type=int, default=8)
    parser.add_argument('--num-steps', type=int, default=400)
    parser.add_argument('--test-steps', type=int, default=2000)
    parser.add_argument('--num-frames', type=int, default=100000)
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-img', type=bool, default=True)
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval, one eval per n updates (default: None)')
    parser.add_argument('--play-game', type=bool, default=False)
    parser.add_argument('--epsilon', type=float, default=0, help="Exploration rate (epsilon)")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train-bs', type=int, default=64)
    parser.add_argument('--eval-bs', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--reset-collection', type=bool, default=False)

    parser.add_argument('--data-collector', type=str, default='rnd')
    parser.add_argument('--data-dir', type=str, default="./data")
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--rl-log-interval', type=int, default=5)

    # rnd params
    parser.add_argument('--rnd-model-dir', type=str, default="./rnd")
    parser.add_argument('--lam', type=float, default=0.95, help='Lambda for GAE')
    parser.add_argument('--num_worker', type=int, default=1, help='Number of workers')
    parser.add_argument('--num_step', type=int, default=128, help='Number of steps per worker')
    parser.add_argument('--ppo_eps', type=float, default=0.1, help='PPO epsilon')
    parser.add_argument('--epoch', type=int, default=4, help='Number of PPO epochs')
    parser.add_argument('--mini_batch', type=int, default=4, help='Number of mini-batches per update')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--entropy_coef', type=float, default=0.001, help='Entropy coefficient')
    parser.add_argument('--gamma', type=float, default=0.999, help='Discount factor for rewards')
    parser.add_argument('--clip_grad_norm', type=float, default=0.5, help='Max norm of gradient clipping')
    
    parser.add_argument('--sticky_action', action='store_true', default=True, help='Use sticky actions in the environment')
    parser.add_argument('--action_prob', type=float, default=0.25, help='Probability of repeating action when using sticky actions')
    parser.add_argument('--life_done', action='store_true', default=False, help='Terminate episode when life is lost')
    
    parser.add_argument('--env_type', type=str, default='atari', help='Type of environment')
    parser.add_argument('--env_id', type=str, default='MontezumaRevengeNoFrameskip-v4', help='Environment ID')

    parser.add_argument('--max_step_per_episode', type=int, default=4500, help='Max steps per episode')
    parser.add_argument('--ext_coef', type=float, default=2.0, help='External reward coefficient')
    parser.add_argument('--int_coef', type=float, default=1.0, help='Intrinsic reward coefficient')
    
    parser.add_argument('--int_gamma', type=float, default=0.99, help='Discount factor for intrinsic rewards')
    parser.add_argument('--stable_eps', type=float, default=1e-8, help='Epsilon for numerical stability')
    
    parser.add_argument('--state_stack_size', type=int, default=4, help='Number of stacked frames as input')
    parser.add_argument('--preproc_height', type=int, default=84, help='Height of preprocessed frames')
    parser.add_argument('--preproc_width', type=int, default=84, help='Width of preprocessed frames')
    
    parser.add_argument('--use_gae', action='store_true', default=True, help='Use Generalized Advantage Estimation')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU for training')
    parser.add_argument('--use_norm', action='store_true', default=False, help='Use normalization')
    parser.add_argument('--use_noisy_net', action='store_true', default=False, help='Use noisy networks')
    
    parser.add_argument('--update_proportion', type=float, default=0.25, help='Proportion of updates for intrinsic rewards')
    parser.add_argument('--obs_norm_step', type=int, default=50, help='Number of steps for observation normalization')


    args = parser.parse_args()


    return args