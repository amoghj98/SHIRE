#!/usr/bin/python

import sys

import argparse
from argparse import RawTextHelpFormatter


def config():
    desc=("RL Training args configuration script.")
    epilog=("For further documentation, refer the IntuitiveRL framework documentation page at https://github.com/")
    parser=argparse.ArgumentParser(description=desc, epilog=epilog, formatter_class=RawTextHelpFormatter)
    # policy training args
    parser.add_argument('-lr', '--learning-rate', type=float, nargs='+', default=[5e-3, 1e-4, 8e-5, 5e-5, 8e-6], help="Policy learning rates")
    parser.add_argument('-lrs', '--lr-schedule', type=float, nargs='+', default=[50000, 100000, 150000, 250000, 500000], help="Policy learning rate schedule")
    parser.add_argument('-ns', '--n-steps', type=int, default=512, help="Number of environment interactions per policy backprop")
    parser.add_argument('-bsz', '--batch-size', type=int, default=4, help="Policy training batch size")
    parser.add_argument('-e', '--n-epochs', type=int, default=10, help="Number of training epochs per policy backprop")
    parser.add_argument('-cr', '--clip-range', type=float, nargs='+', default=[0.4, 0.3, 0.2], help="Clipping parameter values")
    parser.add_argument('-crs', '--clip-schedule', type=float, nargs='+', default=[100000, 250000, 1000000], help="Clip parameter decay schedule")
    parser.add_argument('-ef', '--evalFreq', type=int, default=2000, help="Policy evaluation frequency")
    # training loop args
    parser.add_argument('--totalSteps', type=int, default=int(1e6), help="Total number of training interactions with env")
    # eval args
    parser.add_argument('-m', '--mode', type=str, default='train', help="Script execution mode (train/test)")
    parser.add_argument('--loadPath', type=str, default='./best_model/', help="Model load path")
    parser.add_argument('--loadName', type=str, default='best_model', help="Saved model filename")
    parser.add_argument('-nep', '--nEpisodes', type=int, default=10, help="Number of episodes to run in eval mode")
    # log args
    parser.add_argument('--logPath', type=str, default="./tensorboard_PPO", help="Tensorboard log path")
    # RL Environment args
    parser.add_argument('--env', type=str, default='LunarLander-v2', help="Task Environment")
    parser.add_argument('--nCores', type=int, default=4, help="Number of parallel training threads")
    parser.add_argument('--continuous', action='store_true', default=False, help="Flag to set if continuous action space is desired")
    # GUI args
    parser.add_argument('--guiMode', type=str, default=None, help="Environment GUI mode")
    parser.add_argument('--progressGUI', action='store_false', default=True, help="Progress bar display flag")
    # physics constants
    parser.add_argument('-g', '--gravity', type=float, default=-9.80665, help="Acceleration due to gravity (m/s^2)")
    parser.add_argument('-wind', action='store_true', default=False, help="Enable wind flag")
    parser.add_argument('-wp', '--windPower', type=float, default=15.0, help="Max. magnitutde of applied linear wind")
    parser.add_argument('-tp', '--turbulencePower', type=float, default=1.5, help="Max. magnitude of applied rotary wind")
    # parse all args
    args = parser.parse_args()
    # kwargs dict
    args.env_kwargs = {'render_mode': args.guiMode, 'continuous':args.continuous, 'gravity': args.gravity, 'enable_wind': args.wind, 'wind_power': args.windPower, 'turbulence_power': args.turbulencePower}
    # sanity checks
    if len(args.learning_rate) != len(args.lr_schedule):
        print(f'[\033[1;31mFATAL\033[0m] Args learning-rate and lr-schedule have different lengths!')
        sys.exit(-1)
    if len(args.clip_range) != len(args.clip_schedule):
        print(f'[\033[1;31mFATAL\033[0m] Args clip-range and clip-schedule have different lengths!')
        sys.exit(-1)
    # forced arg behaviour handling
    if args.mode == "test":
        args.guiMode = "human"
    return args