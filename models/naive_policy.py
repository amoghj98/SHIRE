#!/usr/bin/python

import torch
import torch.nn as nn
import snntorch
from snntorch import utils
from model_utils import *
from torchinfo import summary
import gymnasium as gym
# import gym
from gym import spaces
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


class naivePolicy(nn.Module):
	"""
		Custom Actor-Critic class.
		NOTE: Custom actor-critics must accept an observation encoding of shape = features_dim of the encoder class
		NOTE: This class only describes the behaviour of the ActorCritic. It must be wrapped in a 
	"""
	def __init__(self, feature_dim, last_layer_dim_pi, last_layer_dim_vf) -> None:
		super().__init__()
		self.latent_dim_pi = last_layer_dim_pi
		self.latent_dim_vf = last_layer_dim_vf


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)
	

if __name__ == "__main__":
	obs = spaces.Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], shape=(4,), dtype=np.float32)
	# obs = spaces.Dict({'img':spaces.Box(low=0.0, high=255, dtype=np.uint8, shape=(5, 1, 256, 256)), \
											#   'imu':spaces.Box(low=-50.0, high=50.0, dtype=np.float64, shape=(2, 3))})
	# obs={}
	# obs['img'] = np.random.randint(255, size = (5, 1, 256, 256))
	# obs['imu'] = np.random.uniform(low=-50, high=50, size = (2, 3))
	# test_module = naiveEncoder(observation_space=obs, debug=True)#.cuda()
	test_module = naivePolicy(debug=True)
	summary(test_module, [(1, 256, 256, 5), (2, 3)], batch_dim=0)
	cnt = count_parameters(test_module)
	print(cnt)