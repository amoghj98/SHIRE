#!/usr/bin/python

import sys
import time

import argparse
from argparse import RawTextHelpFormatter

from util import *

import intuitionNets as inets


def config():
	desc=("RL Training args configuration script.")
	epilog=("For further documentation, refer the IntuitiveRL framework documentation page at https://github.com/")
	parser=argparse.ArgumentParser(description=desc, epilog=epilog, formatter_class=RawTextHelpFormatter)
	# policy training args
	parser.add_argument('-a', '--algorithm', type=str, default='ppo', help="Policy optimisation algorithm")
	parser.add_argument('-lr', '--learning-rate', type=float, nargs='+', default=[5e-3, 1e-4, 8e-5, 5e-5, 8e-6], help="Policy learning rates")
	parser.add_argument('-lrs', '--lr-schedule', type=float, nargs='+', default=[50000, 100000, 150000, 250000, 500000], help="Policy learning rate schedule")
	parser.add_argument('-bfsz', '--buffer-size', type=int, default=int(1e6), help="Replay Buffer size")
	parser.add_argument('-ns', '--n-steps', type=int, default=512, help="Number of environment interactions per policy backprop")
	parser.add_argument('-bsz', '--batch-size', type=int, default=4, help="Policy training batch size")
	parser.add_argument('-e', '--n-epochs', type=int, default=10, help="Number of training epochs per policy backprop")
	parser.add_argument('-cr', '--clip-range', type=float, nargs='+', default=[0.4, 0.3, 0.2], help="Clipping parameter values")
	parser.add_argument('-crs', '--clip-schedule', type=float, nargs='+', default=[100000, 250000, 1000000], help="Clip parameter decay schedule")
	parser.add_argument('-ef', '--evalFreq', type=int, default=2000, help="Policy evaluation frequency")
	parser.add_argument('-ls', '--learning-starts', type=int, default=1000, help="Number of initial exploration steps")
	parser.add_argument('--tau', type=float, default=0.001, help="Soft update co-efficient")
	parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
	parser.add_argument('--lambd', type=float, default=0.95, help="Bias v/s Variance trade-off factor")
	parser.add_argument('--beta', type=float, nargs='+', default=[0.0], help="Entropy Loss co-efficient")
	parser.add_argument('--beta-schedule', type=float, nargs='+', default=[1e6], help="Entropy Loss co-efficient schedule")
	parser.add_argument('-tkl', '--target_kl', type=float, default=0.01, help="Target KL-Divergence Limit for policy updates")
	# training loop args
	parser.add_argument('--totalSteps', type=int, default=int(1e6), help="Total number of training interactions with env")
	# eval args
	parser.add_argument('-m', '--mode', type=str, default='train', help="Script execution mode (train/test)")
	parser.add_argument('--loadPath', type=str, default='./best_model/', help="Model load path")
	parser.add_argument('--loadName', type=str, default='best_model', help="Saved model filename")
	parser.add_argument('-nep', '--nEpisodes', type=int, default=100, help="Number of episodes to run in eval mode")
	# log args
	parser.add_argument('--logPath', type=str, default="./tensorboard_PPO", help="Tensorboard log path")
	# RL Environment args
	parser.add_argument('--env', type=str, default='LunarLander-v2', help="Task Environment")
	parser.add_argument('--nCores', type=int, default=4, help="Number of parallel training threads")
	parser.add_argument('--theta-margin', type=float, default=0.4, help="Angle margin for Lunar Lander PGM")
	parser.add_argument('--continuous', action='store_true', default=False, help="Flag to set if continuous action space is desired")
	parser.add_argument('-ssr', '--solved-state-reward', type=float, default=None, help="Solved State Reward Threshold")
	# Intuition Encouragement framework args
	parser.add_argument('--intuitiveEncouragement', action='store_true', default=False, help="Flag for enabling intuition encouragement")
	parser.add_argument('-icf', '--intuitionCoef', type=float, nargs='+', default=[1000.0], help="Multiplicative hyperparamater for tuning effect of intuition net")
	parser.add_argument('-icfs', '--intuitionCoef-schedule', type=float, nargs='+', default=[int(1e6)], help="Intuition COefficient decay schedule")
	parser.add_argument('-ist', '--intuition-stop-thresh', type=float, default=100.0, help="Mean reward at which to stop intuition encouragement")
	# GUI args
	parser.add_argument('--guiMode', type=str, default=None, help="Environment GUI mode")
	parser.add_argument('--progressGUI', action='store_false', default=True, help="Progress bar display flag")
	# physics constants
	parser.add_argument('-g', '--gravity', type=float, default=-9.80665, help="Acceleration due to gravity (m/s^2)")
	parser.add_argument('-wind', action='store_true', default=False, help="Enable wind flag")
	parser.add_argument('-wp', '--windPower', type=float, default=15.0, help="Max. magnitutde of applied linear wind")
	parser.add_argument('-tp', '--turbulencePower', type=float, default=1.5, help="Max. magnitude of applied rotary wind")
	#
	parser.add_argument('-frw', '--fwd-rew-wt', type=float, default=1.0, help="Swimmer env forward reward term weight")
	parser.add_argument('-ccw', '--ctrl-cost-wt', type=float, default=1e-4, help="Swimmer env control cost weight")
	parser.add_argument('-rns', '--rst-noise-scale', type=float, default=0.1, help="Initial obs perturbation scaling factor")
	parser.add_argument('--exclude-curr-pos-obs', action='store_true', default=False, help="Flag to control x- y-coordinate omission from observation")
	# parse all args
	args = parser.parse_args()
	# sanity checks
	if len(args.learning_rate) != len(args.lr_schedule):
		console_out(consoleMsg='Args learning-rate and lr-schedule have different lengths!', msgCategory=msgCategory.FATAL)
		sys.exit(-1)
	if len(args.clip_range) != len(args.clip_schedule):
		console_out(consoleMsg='Args clip-range and clip-schedule have different lengths!', msgCategory=msgCategory.FATAL)
		sys.exit(-1)
	if len(args.beta) != len(args.beta_schedule):
		console_out(consoleMsg='Args beta and beta-schedule have different lengths!', msgCategory=msgCategory.FATAL)
		sys.exit(-1)
	if len(args.intuitionCoef) != len(args.intuitionCoef_schedule):
		console_out(consoleMsg='Args intuitionCoef and intuitionCoef-schedule have different lengths!', msgCategory=msgCategory.FATAL)
		sys.exit(-1)
	if args.continuous and args.algorithm == 'dqn':
		console_out(consoleMsg='DQN does not support continuous action spaces. Remove the --continuous flag or change the optimisation algorithm!', msgCategory=msgCategory.FATAL)
		sys.exit(-1)
	if args.target_kl <= 0:
		console_out(consoleMsg='Arg target_kl can not be <= 0. Modify this arg and try again!', msgCategory=msgCategory.FATAL)
		sys.exit(-1)
	# forced arg behaviour handling
	if args.mode == "test":
		args.guiMode = "human"
	if args.continuous and 'MountainCar' in args.env:
		args.env = 'MountainCarContinuous-v0'
	# check env support
	supported_envs = ['LunarLander-v2', 'CartPole-v1', 'MountainCar-v0', 'Pendulum-v1', 'InvertedPendulum-v4', 'Acrobot-v1', 'Swimmer-v4']
	args.env_kwargs = {}
	if args.env not in supported_envs:
		console_out(consoleMsg=f'Unsupported environment "{args.env}". Check arg "env" for typos, or add support', msgCategory=msgCategory.FATAL)
		sys.exit(-1)
	# construct env kwargs dict
	argList = {'LunarLander-v2': ['continuous', 'gravity', 'enable_wind', 'wind_power', 'turbulence_power'],
			   'Pendulum-v1': ['g'],
			   'Swimmer-v4': ['forward_reward_weight', 'ctrl_cost_weight', 'reset_noise_scale', 'exclude_current_positions_from_observation']}
	if args.env in argList.keys():
		argList[args.env].append("render_mode")
	else:
		argList[args.env] = ["render_mode"]
	env_kwargs = {'render_mode': args.guiMode, 'continuous':args.continuous, 'gravity': args.gravity, 'g': -args.gravity, 'enable_wind': args.wind, 'wind_power': args.windPower, 'turbulence_power': args.turbulencePower, 'forward_reward_weight': args.fwd_rew_wt, 'ctrl_cost_weight': args.ctrl_cost_wt, 'reset_noise_scale': args.rst_noise_scale, 'exclude_current_positions_from_observation': args.exclude_curr_pos_obs}
	for key in argList[args.env]:
		args.env_kwargs[key] = env_kwargs[key]
	#
	args.pgm_kwargs = {}
	if args.intuitiveEncouragement:
		if args.env not in inets.__envs__.keys():
			console_out(consoleMsg=f'Unsupported intuition net "{args.env}". Check arg "env" for typos, or add support', msgCategory=msgCategory.FATAL)
			sys.exit(-1)
		argList = {'LunarLander-v2': ['theta_margin'],
				'MountainCarContinuous-v0': ['continuous']}
		if args.env in argList.keys():
			argList[args.env].append("debug")
		else:
			argList[args.env] = ["debug"]
		pgm_kwargs = {'debug': False, 'continuous':args.continuous, 'theta_margin': args.theta_margin}
		for key in argList[args.env]:
			args.pgm_kwargs[key] = pgm_kwargs[key]
	# get time for creating log and other dir names later on
	args._raw_time = time.localtime()
	args.t = "".join([i for i in time.strftime("%D %T", args._raw_time) if i not in [" ", "/", ":"]])
	return args