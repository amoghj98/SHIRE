#!/usr/bin/python

import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

# from stable_baselines3.common.buffers import RolloutBuffer
from .buffers import RolloutBuffer
# from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from .on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

SelfPPO = TypeVar("SelfPPO", bound="PPO")

import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import BeliefPropagation

from os.path import expanduser
import sys

sys.path.insert(1, expanduser("~")+'/SHIRE/intuitionNets')
import intuitionNets as inets

sys.path.insert(1, expanduser("~")+'/SHIRE')
from util import *

warnings.simplefilter("error", RuntimeWarning)

class intuitivePPO(OnPolicyAlgorithm):
	"""
	Proximal Policy Optimization algorithm (PPO) (clip version)

	Paper: https://arxiv.org/abs/1707.06347
	Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
	https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
	Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

	Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

	:param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
	:param env: The environment to learn from (if registered in Gym, can be str)
	:param learning_rate: The learning rate, it can be a function
		of the current progress remaining (from 1 to 0)
	:param n_steps: The number of steps to run for each environment per update
		(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
		NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
		See https://github.com/pytorch/pytorch/issues/29372
	:param batch_size: Minibatch size
	:param n_epochs: Number of epoch when optimizing the surrogate loss
	:param gamma: Discount factor
	:param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
	:param clip_range: Clipping parameter, it can be a function of the current progress
		remaining (from 1 to 0).
	:param clip_range_vf: Clipping parameter for the value function,
		it can be a function of the current progress remaining (from 1 to 0).
		This is a parameter specific to the OpenAI implementation. If None is passed (default),
		no clipping will be done on the value function.
		IMPORTANT: this clipping depends on the reward scaling.
	:param normalize_advantage: Whether to normalize or not the advantage
	:param ent_coef: Entropy coefficient for the loss calculation
	:param vf_coef: Value function coefficient for the loss calculation
	:param max_grad_norm: The maximum value for the gradient clipping
	:param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
		instead of action noise exploration (default: False)
	:param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
		Default: -1 (only sample at the beginning of the rollout)
	:param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
	:param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
	:param target_kl: Limit the KL divergence between updates,
		because the clipping is not enough to prevent large update
		see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
		By default, there is no limit on the kl div.
	:param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
		the reported success rate, mean episode length, and mean reward over
	:param tensorboard_log: the log location for tensorboard (if None, no logging)
	:param policy_kwargs: additional arguments to be passed to the policy on creation
	:param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
		debug messages
	:param seed: Seed for the pseudo random generators
	:param device: Device (cpu, cuda, ...) on which the code should be run.
		Setting it to auto, the code will be run on the GPU if possible.
	:param _init_setup_model: Whether or not to build the network at the creation of the instance
	"""
	policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
		"MlpPolicy": ActorCriticPolicy,
		"CnnPolicy": ActorCriticCnnPolicy,
		"MultiInputPolicy": MultiInputActorCriticPolicy,
	}

	def __init__(
		self,
		policy: Union[str, Type[ActorCriticPolicy]],
		env: Union[GymEnv, str],
		learning_rate: Union[float, Schedule] = 3e-4,
		n_steps: int = 2048,
		batch_size: int = 64,
		n_epochs: int = 10,
		gamma: float = 0.99,
		gae_lambda: float = 0.95,
		clip_range: Union[float, Schedule] = 0.2,
		clip_range_vf: Union[None, float, Schedule] = None,
		normalize_advantage: bool = True,
		ent_coef: float = 0.0,
		ent_coef_sched: Union[float, Schedule] = 0.0,
		vf_coef: float = 0.5,
		theta_margin: float = 0.4,
		use_intuition: bool = False,
		intuition_coef: Union[float, Schedule] = 0.5,
		mean_reward_thresh = 100,
		env_name = 'LunarLander-v2',
		pgm_kwargs: Optional[Dict[str, Any]] = None,
		max_grad_norm: float = 0.5,
		use_sde: bool = False,
		sde_sample_freq: int = -1,
		rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
		rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
		target_kl: Optional[float] = None,
		stats_window_size: int = 100,
		tensorboard_log: Optional[str] = None,
		policy_kwargs: Optional[Dict[str, Any]] = None,
		verbose: int = 0,
		seed: Optional[int] = None,
		device: Union[th.device, str] = "auto",
		_init_setup_model: bool = True,
	):
		super().__init__(
			policy,
			env,
			learning_rate=learning_rate,
			n_steps=n_steps,
			gamma=gamma,
			gae_lambda=gae_lambda,
			ent_coef=ent_coef,
			vf_coef=vf_coef,
			max_grad_norm=max_grad_norm,
			use_sde=use_sde,
			sde_sample_freq=sde_sample_freq,
			rollout_buffer_class=rollout_buffer_class,
			rollout_buffer_kwargs=rollout_buffer_kwargs,
			stats_window_size=stats_window_size,
			tensorboard_log=tensorboard_log,
			policy_kwargs=policy_kwargs,
			verbose=verbose,
			device=device,
			seed=seed,
			_init_setup_model=False,
			supported_action_spaces=(
				spaces.Box,
				spaces.Discrete,
				spaces.MultiDiscrete,
				spaces.MultiBinary,
			),
		)
		# Sanity check, otherwise it will lead to noisy gradient and NaN
		# because of the advantage normalization
		if normalize_advantage:
			assert (batch_size > 1), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"
		if self.env is not None:
			# Check that `n_steps * n_envs > 1` to avoid NaN
			# when doing advantage normalization
			buffer_size = self.env.num_envs * self.n_steps
			assert buffer_size > 1 or (not normalize_advantage), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
			# Check that the rollout buffer size is a multiple of the mini-batch size
			untruncated_batches = buffer_size // batch_size
			if buffer_size % batch_size > 0:
				warnings.warn(
					f"You have specified a mini-batch size of {batch_size},"
					f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
					f" after every {untruncated_batches} untruncated mini-batches,"
					f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
					f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
					f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
				)
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.clip_range = clip_range
		self.clip_range_vf = clip_range_vf
		self.normalize_advantage = normalize_advantage
		self.target_kl = target_kl
		self.ent_coef_sched = ent_coef_sched
		#
		self.use_intuition = use_intuition
		if self.use_intuition:
			self.intuition_coef = intuition_coef
			self.low_intuition_loss = False
			self.env_name = env_name
			self.rew_smoothing_factor = 0.5
			self.mean_reward = None
			self.mean_reward_thresh = mean_reward_thresh
			# if 'CartPole' in self.env_name:
			self.pgm_kwargs = pgm_kwargs
			self.net = inets.__envs__[self.env_name](**self.pgm_kwargs)
		#
		if _init_setup_model:
			self._setup_model()

	def _setup_model(self) -> None:
		super()._setup_model()
		# Initialize schedules for policy/value clipping
		self.clip_range = get_schedule_fn(self.clip_range)
		if self.use_intuition:
			self.intuition_coef = get_schedule_fn(self.intuition_coef)
		if self.clip_range_vf is not None:
			if isinstance(self.clip_range_vf, (float, int)):
				assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
			self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

	def train(self) -> None:
		"""
		Update policy using the currently gathered rollout buffer.
		"""
		# Switch to train mode (this affects batch norm / dropout)
		self.policy.set_training_mode(True)
		# Update optimizer learning rate
		self._update_learning_rate(self.policy.optimizer)
		# Compute current clip range
		clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
		# Optional: clip range for the value function
		if self.clip_range_vf is not None:
			clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

		if self.use_intuition:
			intuition_coef = self.intuition_coef(self._current_progress_remaining)
			# self.net = inets.__envs__[self.env_name](**self.pgm_kwargs)
			# self.net.generate_memory_states(self.rollout_buffer)

		entropy_losses = []
		pg_losses, value_losses = [], []
		intuition_losses = []
		clip_fractions = []

		self.ent_coef = self.ent_coef_sched(self._current_progress_remaining)

		# disable intuition after basics are learned
		if self.use_intuition:
			epi_rew_mean_across_envs = np.zeros((self.n_envs, 1))
			for env_idx in range(self.n_envs):
				start_idxs = np.where(self.rollout_buffer.episode_starts[:, env_idx] > 0)[0]
				end_idxs = np.append(np.copy(start_idxs)[1:], self.rollout_buffer.buffer_size)
				env_rews = np.array([self.rollout_buffer.rewards[start_idxs[i] : end_idxs[i], env_idx].sum() for i in range(len(start_idxs))])
				try:
					epi_rew_mean_across_envs[env_idx] = env_rews.mean()
				except:
					epi_rew_mean_across_envs[env_idx] = self.mean_reward
			epi_rew_mean = epi_rew_mean_across_envs.mean()
			if self.mean_reward is not None:
				self.mean_reward = self.rew_smoothing_factor * self.mean_reward + (1 - self.rew_smoothing_factor) * epi_rew_mean
			else:
				self.mean_reward = epi_rew_mean if epi_rew_mean is not np.NaN else self.mean_reward
			if (self.mean_reward > self.mean_reward_thresh) or self.low_intuition_loss:
				if self.low_intuition_loss and self.use_intuition:
					console_out(consoleMsg=f'Intuition encouragement stopped due to high policy agreement', msgCategory=msgCategory.CUSTOM, categoryStr='alert')
				elif self.use_intuition:
					console_out(consoleMsg=f'Intuition encouragement stopped due to high mean reward', msgCategory=msgCategory.CUSTOM, categoryStr='alert')
				self.use_intuition = False

		continue_training = True
		# train for n_epochs epochs
		for epoch in range(self.n_epochs):
			approx_kl_divs = []
			# Do a complete pass on the rollout buffer
			for rollout_data in self.rollout_buffer.get(self.batch_size):
				actions = rollout_data.actions
				if isinstance(self.action_space, spaces.Discrete):
					# Convert discrete action from float to long
					actions = rollout_data.actions.long().flatten()

				# Re-sample the noise matrix because the log_std has changed
				if self.use_sde:
					self.policy.reset_noise(self.batch_size)

				values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
				values = values.flatten()
				# Normalize advantage
				advantages = rollout_data.advantages
				# Normalization does not make sense if mini batchsize == 1, see GH issue #325
				if self.normalize_advantage and len(advantages) > 1:
					advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

				# ratio between old and new policy, should be one at the first iteration
				ratio = th.exp(log_prob - rollout_data.old_log_prob)

				# clipped surrogate loss
				policy_loss_1 = advantages * ratio
				policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
				policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
				# print(policy_loss.shape)

				# Logging
				pg_losses.append(policy_loss.item())
				clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
				clip_fractions.append(clip_fraction)

				if self.clip_range_vf is None:
					# No clipping
					values_pred = values
				else:
					# Clip the difference between old and new value
					# NOTE: this depends on the reward scaling
					values_pred = rollout_data.old_values + th.clamp(
						values - rollout_data.old_values, -clip_range_vf, clip_range_vf
					)
				# Value loss using the TD(gae_lambda) target
				value_loss = F.mse_loss(rollout_data.returns, values_pred)
				# print(value_loss.shape)
				value_losses.append(value_loss.item())

				# Entropy loss favor exploration
				if entropy is None:
					# Approximate entropy when no analytical form
					entropy_loss = -th.mean(-log_prob)
				else:
					entropy_loss = -th.mean(entropy)

				entropy_losses.append(entropy_loss.item())
				# print(entropy_loss.shape)

				# Intuition Loss Computation
				if self.use_intuition:
					intuitive_action_loss = self.net.forward(rollout_data=rollout_data)
					# sys.exit(0)
					intuition_losses.append(intuitive_action_loss.item())
					loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + intuition_coef * intuitive_action_loss
				else:
					# std loss computation
					loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

				# Calculate approximate form of reverse KL Divergence for early stopping
				# see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
				# and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
				# and Schulman blog: http://joschu.net/blog/kl-approx.html
				with th.no_grad():
					log_ratio = log_prob - rollout_data.old_log_prob
					approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
					approx_kl_divs.append(approx_kl_div)

				if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
					continue_training = False
					if self.verbose >= 1:
						print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
					break

				# Optimization step
				self.policy.optimizer.zero_grad()
				loss.backward()
				# Clip grad norm
				th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
				self.policy.optimizer.step()

			self._n_updates += 1
			if not continue_training:
				break

		explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

		# Logs
		self.logger.record("train/entropy_loss", np.mean(entropy_losses))
		self.logger.record("train/entropy_loss_coef", self.ent_coef)
		self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
		self.logger.record("train/value_loss", np.mean(value_losses))
		if self.use_intuition:
			self.logger.record("train/intuition_loss", np.mean(intuition_losses))
		self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
		self.logger.record("train/clip_fraction", np.mean(clip_fractions))
		self.logger.record("train/loss", loss.item())
		self.logger.record("train/explained_variance", explained_var)
		if self.use_intuition:
			self.logger.record("train/smoothed_mean_reward", self.mean_reward)
		if hasattr(self.policy, "log_std"):
			self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

		self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
		self.logger.record("train/clip_range", clip_range)
		if self.clip_range_vf is not None:
			self.logger.record("train/clip_range_vf", clip_range_vf)

	def learn(
		self: SelfPPO,
		total_timesteps: int,
		callback: MaybeCallback = None,
		log_interval: int = 1,
		tb_log_name: str = "intuitivePPO",
		reset_num_timesteps: bool = True,
		progress_bar: bool = False,
	) -> SelfPPO:
		return super().learn(
			total_timesteps=total_timesteps,
			callback=callback,
			log_interval=log_interval,
			tb_log_name=tb_log_name,
			reset_num_timesteps=reset_num_timesteps,
			progress_bar=progress_bar,
		)