#!/usr/bin/python

import sys
import numpy as np
import torch
import time
import os
import glob

import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor


# define console out colours
white='\033[1;37m'
red='\033[1;31m'
green='\033[1;32m'
yellow='\033[1;33m'
nc='\033[0m'


# util functions
def parameter_scheduler(sched, initial_tstep=0, total_tsteps=1e6):
	def func(progress_remaining: float) -> float:
		tstep = (total_tsteps * (1 - progress_remaining)) + initial_tstep
		pmt= -1
		for pair in sched:
			if tstep < pair[0]:
				pmt = pair[1]
				break
		if pmt < 0:
			pmt = sched[-1][1]
		return pmt
	return func

def compose_schedule(vals, tsteps):
	sched = []
	for i in range(len(vals)):
		sched.append((tsteps[i], vals[i]))
	return sched

def make_env(env_id: str, args, rank: int, seed: int = 0):
	"""
	Utility function for multiprocessed env.
	:param env_id: the environment ID
	:param num_env: the number of environments you wish to have in subprocesses
	:param seed: the inital seed for RNG
	:param rank: index of the subprocess
	"""
	cont_envs = ["LunarLander-v2"]
	def _init():
		env = Monitor(
			env = gym.make(env_id, **args.env_kwargs)
		)
		# if env_id in cont_envs:
		# 	env = Monitor(
		# 				env = gym.make(env_id, continuous=args.continuous, render_mode=args.guiMode)
		# 			)
		# else:
		# 	env = Monitor(
		# 			env = gym.make(env_id, render_mode=args.guiMode)
		# 		)
		env.reset(seed=int(seed[rank]) + rank)
		return env
	set_random_seed(int(seed[rank]))
	return _init

def cleanup(env, args):
	# close env
	env.close()
	# remove logs if running test
	if args.mode == "test":
		files = glob.glob(os.path.join("./tensorboard_PPO/*"))
		current_logfile = max(files, key=os.path.getctime)
		os.remove(current_logfile)

def console_out(formattedStr, terminalChar=None, suppressCarriageReturn=False):
	if terminalChar is not None:
		print(formattedStr, end=terminalChar, flush=suppressCarriageReturn)
	else:
		print(formattedStr, flush=suppressCarriageReturn)
