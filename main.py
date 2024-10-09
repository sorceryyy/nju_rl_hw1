import matplotlib
import torch
matplotlib.use('Agg')  # For faster non-interactive plotting
import matplotlib.pyplot as plt

from arguments import get_args
from Dagger import MyAgent
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from RLA import logger, time_step_holder, exp_manager
from expert import ExpertDataCollector, Data_Collector
from env import Env
from rnd.agents import RNDAgent
from utils import human_play, get_device, evaluate_episode, set_seed_everywhere, get_dataloader


def main():
	# load hyper parameters
	args = get_args()

	# set exp manager
	task_name = 'dagger'
	exp_manager.set_hyper_param(**vars(args))
	exp_manager.add_record_param(["info", "seed", 'env_name'])
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
	data_collector: Data_Collector = None
	if args.data_collector == "sb3":
		data_collector = ExpertDataCollector(env=env, epsilon=args.epsilon)
	elif args.data_collector == "rnd":
		input_size = observation_shape  # 4
		output_size = action_shape  # 2
		agent = RNDAgent(
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
		agent.load(args.rnd_model_dir, env_id=args.env_name)
		evaluate_episode(agent=agent, env=env, eval_episode=1, log_prefix="debug/rnd_agent")	# test
		data_collector = ExpertDataCollector(env=env, agent=agent, args=args)
		data_collector.load()
		data_collector.collect(target_buffer_step=100000)
		data_collector.save()
		data_collector.info()
		

	# start train your agent
	device = get_device()
	dagger = MyAgent(env=env, args=args, device=device)
	epochs = args.epochs
	traj_data, traj_label = data_collector.sample()
	train_data, eval_data, train_label, eval_label = train_test_split(
		traj_data, traj_label, test_size=1-args.train_ratio
	)
	train_dataloader = get_dataloader(train_data, train_label, data_type=torch.float32, label_type=torch.long, bs=args.train_bs)
	eval_dataloader = get_dataloader(eval_data, eval_label, data_type=torch.float32, label_type=torch.long, bs=args.eval_bs)
	evaluate_episode(dagger, env=env, eval_episode=2, time_step=-1, log_prefix="debug/dagger_agent")  # 对当前 agent 的表现进行测试和日志记录	for epoch in tqdm(range(epochs), desc="Train Dagger"):
	for epoch in tqdm(range(epochs), desc="Train Dagger"):

		step = 0
		train_epoch_loss = dagger.train(train_dataloader)

		logger.logkv("train_epoch_loss", train_epoch_loss)
		time_step_holder.set_time(epoch)
		logger.dumpkvs()

		eval_loss, eval_accuracy = dagger.evaluate(eval_dataloader)
		logger.logkv("eval_epoch_loss", eval_loss)
		logger.logkv("eval_epoch_accuracy", eval_accuracy)
		time_step_holder.set_time(epoch)
		logger.dumpkvs()

		if (epoch + 1) % args.rl_log_interval == 0:
			dagger.model.eval()
			evaluate_episode(dagger, env=env, eval_episode=10, time_step=epoch)  # 对当前 agent 的表现进行测试和日志记录	for epoch in tqdm(range(epochs), desc="Train Dagger"):

	evaluate_episode(dagger, eval_episode=20, time_step=epochs)


if __name__ == "__main__":
	main()
