#!/usr/bin/python

import sys
import numpy as np
import torch
import time
import os

# import gym
import gymnasium as gym
# sys.modules["gym"] = gym

# from stable_baselines3 import PPO
from algorithms import intuitivePPO
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
	# construct intuitionCoef schedule
	intuitionSched = compose_schedule(args.intuitionCoef, args.intuitionCoef_schedule)
	# need this to get rid of a weird error "torch does not contain member pi"
	torch.pi = np.pi
	# Create the vectorized environment
	if args.mode == "train":
		vec_env = SubprocVecEnv([make_env(args.env, args, i, torch.randint(0, 2**20, (args.nCores,))) for i in range(args.nCores)])
		eval_vec_env = SubprocVecEnv([make_env(args.env, args, i, torch.randint(0, 2**20, (args.nCores,))) for i in range(args.nCores)])
	elif args.mode == "test":
		vec_env = SubprocVecEnv([make_env(args.env, args, i, torch.randint(0, 2**20, (1,))) for i in range(1)])
	# policy_kwargs = dict(
	# features_extractor_class=naiveEncoder,
	# features_extractor_kwargs=dict(features_dim=300),
	# activation_fn=torch.nn.Sigmoid,
	# )
	# Create an evaluation callback with the same env, called every 10000 iterations
	callbacks = []
	eval_callback = EvalCallback(
		vec_env,
		callback_on_new_best=None,
		n_eval_episodes=args.nEpisodes,
		best_model_save_path="./best_model/"+args.t,
		log_path=args.logPath,
		eval_freq=int(args.evalFreq / args.nCores)
	)
	callbacks.append(eval_callback)
	#
	kwargs = {}
	kwargs["callback"] = callbacks
	if args.mode == "train":
		model = intuitivePPO.intuitivePPO("MlpPolicy", env=vec_env, learning_rate=parameter_scheduler(lrSched, total_tsteps=args.totalSteps), clip_range=parameter_scheduler(clipSched, total_tsteps=args.totalSteps), n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs, verbose=1, tensorboard_log=args.logPath, use_intuition=args.intuitiveEncouragement, intuition_coef=parameter_scheduler(intuitionSched, total_tsteps=args.totalSteps))
		model.set_logger(logger)
		model.learn(total_timesteps=args.totalSteps, progress_bar=args.progressGUI, tb_log_name="ppo_run_" + str(time.time()), **kwargs)
	elif args.mode == "test":
		file = os.path.join(args.loadPath, args.loadName)
		model = intuitivePPO.intuitivePPO.load(file, env=vec_env)
		mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=args.nEpisodes)
		vec_env.render(args.guiMode)
	console_out(f'Exiting...', terminalChar='\t', suppressCarriageReturn=True)
	# perform post-run cleanup
	cleanup(vec_env, args)
	console_out(f'[{green}DONE{nc}]')
