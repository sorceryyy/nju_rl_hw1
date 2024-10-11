import matplotlib
import torch
matplotlib.use('Agg')  # For faster non-interactive plotting
import matplotlib.pyplot as plt

from arguments import get_args
from Dagger import MyAgent
import numpy as np
from tqdm import tqdm

from RLA import logger, time_step_holder, exp_manager
from expert import DaggerTrainer, Trainer
from env import Env
from rnd.agents import RNDAgent
from utils import human_play, get_device, evaluate_episode, set_seed_everywhere, get_dataloader


def main():
	# load hyper parameters
	args = get_args()

	# set exp manager
	task_name = 'dagger'
	exp_manager.set_hyper_param(**vars(args))
	exp_manager.add_record_param(["info", "seed", 'env_name', 'lr', 'train_bs', 'epsilon'])
	exp_manager.configure(task_name, rla_config='./rla_config.yaml', data_root='./rslts')
	exp_manager.log_files_gen()
	exp_manager.print_args()

	set_seed_everywhere(args.seed)

	# environment initial
	env = Env(args.env_name, args.num_stacks)
	# action_shape is the size of the discrete action set, here is 18
	action_shape = env.action_space.n
	# observation_shape is the shape of the observation
	# here is (210,160,3)=(height, weight, channels)
	observation_shape = env.observation_space.shape
	print(action_shape, observation_shape)


	# You can play this game yourself for fun
	if args.play_game:
		human_play(envs=env, action_shape=action_shape)

	# set data collector
	dagger_trainer: Trainer = None
	if args.data_collector == "sb3":
		dagger_trainer = DaggerTrainer(env=env, epsilon=args.epsilon)
	elif args.data_collector == "rnd":
		input_size = observation_shape  # 4
		output_size = action_shape  # 2
		expert_agent = RNDAgent(
			input_size,
			output_size,
			args.num_worker,
			args.num_step,
			args.gamma,
			lam=args.lam,
			learning_rate=args.learning_rate,
			ent_coef=args.entropy_coef,
			clip_grad_norm=args.clip_grad_norm,
			epoch=args.epoch,
			batch_size=args.mini_batch,
			ppo_eps=args.ppo_eps,
			use_cuda=args.use_gpu,
			use_gae=args.use_gae,
			use_noisy_net=args.use_noisy_net
		)

	# start train your agent
	device = get_device()
	logger.info(device)

	agent = MyAgent(env=env, args=args, device=device)
	evaluate_episode(agent, env=env, eval_episode=2, time_step=-1, log_prefix="debug/dagger_agent")  # 对当前 agent 的表现进行测试和日志记录	for epoch in tqdm(range(epochs), desc="Train Dagger"):
	expert_agent.load(args.rnd_model_dir, env_id=args.env_name)
	evaluate_episode(agent=expert_agent, env=env, eval_episode=1, log_prefix="debug/rnd_agent")	# test
	dagger_trainer = DaggerTrainer(env=env, agent=agent, expert_agent=expert_agent, args=args)
		

	epochs = args.epochs
	exp_manager.new_saver(max_to_keep=1000)
	dagger_trainer.load()
	collect_times = 0
	for epoch in tqdm(range(epochs), desc="Train Dagger"):
		if epoch % args.collect_interval == 0:
			collect_times += 1
			dagger_trainer.collect(target_buffer_step=args.epoch_data_num*collect_times)
			dagger_trainer.info()
			dagger_trainer.save()
		dagger_trainer.train(epoch=epoch, train_ratio=args.train_ratio)

		if (epoch + 1) % args.rl_log_interval == 0:
			agent.model.eval()
			evaluate_episode(agent, env=env, eval_episode=10, time_step=epoch, save_fig=True)  # 对当前 agent 的表现进行测试和日志记录	for epoch in tqdm(range(epochs), desc="Train Dagger"):
		
		
		if (epoch + 1) % args.save_interval == 0:
			checkpoint, related_variable = agent.get_checkpoint()
			exp_manager.save_checkpoint(checkpoint, related_variable=related_variable)

	evaluate_episode(agent, env=env, eval_episode=20, time_step=epochs)

	checkpoint, related_variable = agent.get_checkpoint()
	exp_manager.save_checkpoint(checkpoint, related_variable=related_variable)


if __name__ == "__main__":
	main()
