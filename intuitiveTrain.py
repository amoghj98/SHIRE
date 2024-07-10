#!/usr/bin/python

import sys
import numpy as np
import torch
import time
import os
import readline

# import gym
import gymnasium as gym
# sys.modules["gym"] = gym

# from stable_baselines3 import PPO
# from algorithms import *
# from algorithms import *
import algorithms
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from argConfig import config
from util import *

from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
	# get args
	args = config()
	# init tensorboard
	writer = SummaryWriter()
	logger = configure(os.path.join(args.logPath, args.env + '_' + args.t), ['tensorboard'])
	# construct lr schedule
	lrSched = compose_schedule(args.learning_rate, args.lr_schedule)
	# construct clip schedule
	clipSched = compose_schedule(args.clip_range, args.clip_schedule)
	# construct entropy loss coef schedule
	entropySched = compose_schedule(args.beta, args.beta_schedule)
	# construct intuitionCoef schedule
	intuitionSched = compose_schedule(args.intuitionCoef, args.intuitionCoef_schedule)
	# need this to get rid of a weird error "torch does not contain member pi"
	torch.pi = np.pi
	# Create the vectorized environment
	if args.mode == "train":
		if not os.path.exists(os.path.join("./best_model", args.t)):
			os.mkdir(os.path.join("./best_model", args.t))
		if args.set_seeds:
			vec_env = SubprocVecEnv([make_env(args.env, args, i, args.seeds) for i in range(args.nCores)])
		else:
			vec_env = SubprocVecEnv([make_env(args.env, args, i, torch.randint(0, 2**20, (args.nCores,))) for i in range(args.nCores)])
	elif args.mode == "test":
		vec_env = SubprocVecEnv([make_env(args.env, args, i, torch.randint(0, 2**20, (1,))) for i in range(1)])
	# construct rl algo kwargs dict
	algo_kwargs = {'ppo': {'policy': "MlpPolicy", 'env':vec_env, 'learning_rate':parameter_scheduler(lrSched, total_tsteps=args.totalSteps), 'clip_range':parameter_scheduler(clipSched, total_tsteps=args.totalSteps), 'n_steps':args.n_steps, 'batch_size':args.batch_size, 'n_epochs':args.n_epochs, 'gamma': args.gamma, 'ent_coef': args.beta, 'ent_coef_sched': parameter_scheduler(entropySched, total_tsteps=args.totalSteps), 'gae_lambda':args.lambd, 'verbose':1, 'tensorboard_log':args.logPath, 'use_intuition':args.intuitiveEncouragement, 'intuition_coef':parameter_scheduler(intuitionSched, total_tsteps=args.totalSteps), 'mean_reward_thresh': args.intuition_stop_thresh, 'env_name':args.env, 'pgm_kwargs': args.pgm_kwargs},
				'trpo': {'policy': "MlpPolicy", 'env':vec_env, 'learning_rate':parameter_scheduler(lrSched, total_tsteps=args.totalSteps), 'n_steps':args.n_steps, 'batch_size':args.batch_size, 'target_kl': args.target_kl, 'n_critic_updates':args.n_epochs, 'gamma': args.gamma, 'ent_coef': args.beta, 'gae_lambda':args.lambd, 'verbose':1, 'tensorboard_log':args.logPath, 'use_intuition':args.intuitiveEncouragement, 'intuition_coef':parameter_scheduler(intuitionSched, total_tsteps=args.totalSteps), 'mean_reward_thresh': args.intuition_stop_thresh, 'env_name':args.env, 'pgm_kwargs': args.pgm_kwargs},
				'dqn': {'policy': "MlpPolicy", 'env':vec_env, 'learning_rate':parameter_scheduler(lrSched, total_tsteps=args.totalSteps), 'train_freq':args.n_steps, 'target_update_interval': args.n_steps, 'gradient_steps':args.n_epochs * args.nCores, 'batch_size':args.batch_size, 'verbose': 1, 'tensorboard_log':args.logPath, 'use_intuition':args.intuitiveEncouragement, 'intuition_coef':parameter_scheduler(intuitionSched, total_tsteps=args.totalSteps), 'mean_reward_thresh': args.intuition_stop_thresh, 'env_name':args.env, 'pgm_kwargs': args.pgm_kwargs},
				'td3': {'policy': "MlpPolicy", 'env':vec_env, 'learning_rate':parameter_scheduler(lrSched, total_tsteps=args.totalSteps), 'train_freq':args.n_steps, 'gradient_steps':args.n_epochs * args.nCores, 'batch_size':args.batch_size, 'verbose': 1, 'tensorboard_log':args.logPath, 'use_intuition':args.intuitiveEncouragement, 'intuition_coef':parameter_scheduler(intuitionSched, total_tsteps=args.totalSteps), 'mean_reward_thresh': args.intuition_stop_thresh, 'env_name':args.env, 'pgm_kwargs': args.pgm_kwargs},
				'sac': {'policy': "MlpPolicy", 'env':vec_env, 'learning_rate':parameter_scheduler(lrSched, total_tsteps=args.totalSteps), 'buffer_size': args.buffer_size, 'learning_starts': args.learning_starts, 'batch_size':args.batch_size, 'tau': args.tau, 'gamma': args.gamma, 'train_freq':args.n_steps, 'target_update_interval': args.n_steps, 'gradient_steps':args.n_steps, 'verbose': 1, 'target_entropy': 'auto', 'tensorboard_log':args.logPath, 'use_intuition':args.intuitiveEncouragement, 'intuition_coef':parameter_scheduler(intuitionSched, total_tsteps=args.totalSteps), 'mean_reward_thresh': args.intuition_stop_thresh, 'env_name':args.env, 'pgm_kwargs': args.pgm_kwargs}}
	#
	policy_kwargs = dict(
	# features_extractor_class=naiveEncoder,
	# features_extractor_kwargs=dict(features_dim=300),
	# activation_fn=torch.nn.Sigmoid,
	net_arch=dict(pi=[64, 64, 32, 32], qf=[64, 64, 1])
	)
	# Create an evaluation callback with the same env, called every 10000 iterations
	callbacks = []
	#
	terminatorCallback = stopTraining()
	callbacks.append(terminatorCallback)
	#
	solvedStateSaver = saveSolvedState(args.env, args.evalFreq, args.t, args.solved_state_reward)
	#
	eval_callback = EvalCallback(
		vec_env,
		callback_after_eval=None, #solvedStateSaver,
		callback_on_new_best=solvedStateSaver,
		n_eval_episodes=args.nEpisodes,
		best_model_save_path="./best_model/"+args.t,
		log_path=args.logPath,
		eval_freq=int(args.evalFreq / args.nCores)
	)
	callbacks.append(eval_callback)
	#
	kwargs = {}
	kwargs["callback"] = callbacks
	#
	killed = False
	delLog = False
	#
	try:
		if args.mode == "train":
			model = algorithms.__algorithms__[args.algorithm](**algo_kwargs[args.algorithm])#, policy_kwargs=policy_kwargs)
			model.set_logger(logger)
			# print(model.policy)
			model.learn(total_timesteps=args.totalSteps, progress_bar=args.progressGUI, tb_log_name="ppo_run_" + str(time.time()), **kwargs)
			terminatorCallback.external_stop()
			# time.sleep(4)
			# sys.exit(0)
		elif args.mode == "test":
			file = os.path.join(args.loadPath, args.loadName)
			model = algorithms.__algorithms__[args.algorithm].load(file, env=vec_env, env_name=args.env)
			mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=args.nEpisodes)
			vec_env.render(args.guiMode)
	except KeyError:
		console_out(consoleMsg=f'Unsupported optimisation algorithm "{args.algorithm}". Check arg "algorithm" for typos, or add support', msgCat=msgCategory.FATAL)
		sys.exit(-1)
	except KeyboardInterrupt:
		killed = True
	if killed:
		console_out(consoleMsg='Script killed by user', msgCat=msgCategory.INFO, msgCol=msgColour.CYAN, flushPrintBuffer=True)
		console_out(consoleMsg=f'This action can not be undone'.upper()+f'{msgColour.YELLOW} Delete logs? (y/n)\t', msgCat=msgCategory.WARNING, msgCol=msgColour.RED, flushPrintBuffer=True)
		c = input()
		delLog = True if c.lower()=='y' else False
	console_out(consoleMsg=f'Exiting...')
	# perform post-run cleanup
	cleanup(vec_env, args, killed=killed, deleteLogs=delLog)
	console_out(consoleMsg=f'[{green}DONE EXITING{nc}]', suppressMsgCategory=True)
