#!/usr/bin/python

import sys
import numpy as np
import torch
import time
import os
import glob
from enum import Enum
import readline

import argparse
from argparse import RawTextHelpFormatter

import gymnasium as gym
import maze_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# define colour escape codes
red='\033[1;91m'
green='\033[1;92m'
yellow='\033[1;93m'
blue='\033[1;94m'
magenta='\033[1;95m'
cyan='\033[1;96m'
white='\033[1;97m'
nc='\033[0m'


# cosmetic enums
class msgCategory(Enum):
	FATAL = 0
	WARNING = 1
	INFO = 2
	CUSTOM = 3

class msgColour(str, Enum):
	RED = red
	GREEN = green
	YELLOW = yellow
	BLUE = blue
	MAGENTA = magenta
	CYAN = cyan
	WHITE = white
	NO_COLOUR = nc

colourCode = {msgCategory.FATAL: msgColour.RED, msgCategory.WARNING: msgColour.YELLOW, msgCategory.INFO: msgColour.GREEN, msgCategory.CUSTOM: msgColour.CYAN}


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

def _log_seeds(args, seed, rank):
	with open(os.path.join("./best_model", args.t, 'seeds.txt'), 'a') as seed_log_file:
		seed_log_file.write(f'Thread {rank} Seeds: {seed}\n')
		seed_log_file.write(f'Thread {rank} Reset seed: {int(seed[rank]) + rank}\n')
		seed_log_file.write(f'Thread {rank} Random seed: {int(seed[rank])}\n')
	console_out(f'Thread {rank} Seeds: {seed}')
	console_out(f'Thread {rank} Reset seed: {int(seed[rank]) + rank}')
	console_out(f'Thread {rank} Random seed: {int(seed[rank])}')

def _log_args(args):
	with open(os.path.join("./best_model", args.t, 'args.txt'), 'a') as arg_log_file:
		arg_log_file.write('\n'.join(sys.argv[1:]))

def _clear_dir(dir):
	files = os.listdir(dir)
	for file in files:
		os.remove(os.path.join(dir, file))
	os.rmdir(dir)

def read_seeds(nCores, dir, path='./best_model'):
	success = True
	filename = os.path.join(path, dir, 'seeds.txt')
	with open(filename) as seed_file:
		seeds = [0] * nCores
		for line in seed_file:
			if 'Random seed' in line:
				elems = line.split(' ')
				n = int(elems[1])
				try:
					seeds[n] = int(elems[-1])
				except IndexError:
					success = False
	if 0 in seeds:
		success = False
	return seeds, success

def make_env(env_id: str, args, rank: int, seed: int = 0):
	"""
	Utility function for multiprocessed env.
	:param env_id: the environment ID
	:param num_env: the number of environments you wish to have in subprocesses
	:param seed: the inital seed for RNG
	:param rank: index of the subprocess
	"""
	def _init():
		if 'Maze' in env_id:
			env_ID = 'maze_env/Maze-v0'
		else:
			env_ID = env_id
		env = Monitor(
			env = gym.make(env_ID, **args.env_kwargs)
		)
		if args.set_seeds:
			env.reset(seed=int(args.seeds[rank]) + rank)
		else:
			env.reset(seed=int(seed[rank]) + rank)
		if args.mode == 'train':
			_log_seeds(args, seed, rank)
			if rank == 0:
				_log_args(args)
		return env
	if args.set_seeds:
		set_random_seed(int(args.seeds[rank]))
	else:
		set_random_seed(int(seed[rank]))
	return _init

def cleanup(env, args, killed=False, deleteLogs=False):
	console_out(consoleMsg='Cleaning up...', terminalChar='\t', suppressMsgCategory=True)
	# close env
	if not killed:
		env.close()
	else:
		modelDir = os.path.join("./best_model", args.t)
		_clear_dir(modelDir)
	# delete run (useless extra log)
	runDir = os.path.join("./runs", (time.strftime("%b%d_%T", args._raw_time)+'_'+os.uname().nodename).replace(':', '-'))
	_clear_dir(runDir)
	# remove logs if running test
	if deleteLogs or args.mode == "test":
		# delete log
		logDir = os.path.join("./tensorboard_PPO", args.env+"_"+args.t)
		_clear_dir(logDir)
	console_out(consoleMsg=f'[{green}DONE{nc}]', suppressMsgCategory=True)

def console_out(consoleMsg='sample msg', msgCat=msgCategory.INFO, categoryStr=None, categoryCol=msgColour.WHITE, msgCol=msgColour.NO_COLOUR, terminalChar='\n', suppressMsgCategory=False, flushPrintBuffer=False):
	catStr = categoryStr.upper() if (msgCat==msgCategory.CUSTOM and categoryStr is not None) else msgCat.name
	catCol = colourCode[msgCat] if (msgCat in colourCode.keys() and msgCat!=msgCategory.CUSTOM) else categoryCol
	category = f'' if suppressMsgCategory else f'[{catCol}{catStr}{nc}] '
	msg = f'{msgCol}{consoleMsg}{nc}'
	print(f'{category}{msg}', end=terminalChar, flush=flushPrintBuffer)

def manualCleanup(env: str, t: str) -> int:
	console_out(consoleMsg='Cleaning up...', terminalChar='\t', suppressMsgCategory=True)
	# clean everything in runs dir, its useless anyways
	runDir = os.path.join("./runs")
	dirs = os.listdir(runDir)
	for dir in dirs:
		_clear_dir(dir)
	# clean model dir
	modelDir = os.path.join("./best_model", t)
	_clear_dir(modelDir)
	# delete log dir
	logDir = os.path.join("./tensorboard_PPO", env+"_"+t)
	_clear_dir(logDir)
	# done!
	console_out(consoleMsg=f'[{green}DONE{nc}]', suppressMsgCategory=True)


# custom callback class to save models that solve the environment
class saveSolvedState(BaseCallback):
	def __init__(self, env, eval_freq, t, solved_state_reward = None, verbose: int = 0):
		super().__init__(verbose)
		self.envName = env
		self.eval_freq = eval_freq
		self.model_save_path="./best_model/"+t
		self.solved_state_reward = solved_state_reward

	def _on_step(self) -> bool:
		assert self.parent is not None, ("`StopTrainingOnRewardThreshold` callback must be used with an `EvalCallback`")
		#
		if (self.solved_state_reward is not None) and (self.parent.last_mean_reward > self.solved_state_reward) and (self.parent.num_timesteps % self.eval_freq == 0):
			console_out(consoleMsg=f'New solution model with mean reward {self.parent.last_mean_reward}', msgCat=msgCategory.CUSTOM, categoryStr='update', categoryCol=msgColour.BLUE, msgCol=msgColour.CYAN)
			self.model.save(os.path.join(self.model_save_path, self.envName+'_step_'+str(self.parent.num_timesteps)))
		#
		continue_training = True
		return continue_training
	
# custom callback class to stop training
class stopTraining(BaseCallback):
	def __init__(self, verbose: int = 0):
		super().__init__(verbose)
		self.continueTraining = True

	def _on_step(self) -> bool:
		return self.continueTraining
	
	def external_stop(self):
		self.continueTraining = False



if __name__ == "__main__":
	desc=("Utility script.")
	epilog=("For further documentation, refer the IntuitiveRL framework documentation page at https://github.com/")
	parser=argparse.ArgumentParser(description=desc, epilog=epilog, formatter_class=RawTextHelpFormatter)
	# args
	parser.add_argument('env', type=str, default=None, help="Task Environment of dir to be cleaned")
	parser.add_argument('t', type=str, default=None, help="Timestamp string of dir to be cleaned")
	# parse all args
	args = parser.parse_args()
	# sanity checks
	if args.env is None or args.t is None:
		console_out('All arguments are required!', msgCategory.FATAL)
	#
	console_out(consoleMsg=f'This action can not be undone'.upper()+f'{msgColour.YELLOW} Delete logs? (y/n)\t', msgCat=msgCategory.WARNING, msgCol=msgColour.RED, flushPrintBuffer=True)
	c = input()
	delLog = True if c.lower()=='y' else False
	if delLog:
		manualCleanup(env=args.env, t=args.t)
	else:
		console_out('Exiting...', suppressMsgCategory=True)
