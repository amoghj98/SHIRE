#!/usr/bin/python

import sys
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import BeliefPropagation, VariableElimination

import torch as th
import numpy as np

# define console out colours
white='\033[1;37m'
red='\033[1;31m'
green='\033[1;32m'
yellow='\033[1;33m'
nc='\033[0m'


class Acrobot(th.autograd.Function):
	def __init__(self, debug=False):
		self.debug = debug
		self.torque_net = BayesianNetwork()
		self.torque_net.add_nodes_from(['theta', 'flag', 't'])
		self.torque_net.add_edges_from([('theta', 't'), ('flag', 't')])
		#
		theta_cpd = TabularCPD('theta', 2, [[0.5], [0.5]],	# uniform for now
							state_names={'theta': ['neg_ang', 'pos_ang']})
		flag_cpd = TabularCPD('flag', 2, [[0.5], [0.5]],	# uniform for now
							state_names={'flag': ['torque', 'no_torque']})
		t_cpd = TabularCPD('t', 3 , [[0.15, 0.05, 0.8, 0.05],
									[0.05, 0.9, 0.05, 0.9],
									[0.8, 0.05, 0.15, 0.05]],#[[0.8, 0.2], [0.2, 0.8]],
							evidence=['theta', 'flag'], evidence_card=[2, 2],
							state_names={'t': ['neg_t', 'no_t', 'pos_t'],
										'theta': ['neg_ang', 'pos_ang'],
										'flag': ['torque', 'no_torque']})
		self.torque_net.add_cpds(theta_cpd, flag_cpd, t_cpd)
		#
		self.bp = {'torque_net': BeliefPropagation(self.torque_net)}
		self.bp = {'torque_net': VariableElimination(self.torque_net)}

	def __reduce__(self) -> str | tuple[any, ...]:
		return (self.__class__, (self.debug, ))

	def check_pgm(self, net):
		assert net.check_model() == True, f'ERROR: Inconsistent probability assignments detected in network "{net}"!'
		if self.debug:
			print(self.torque_net.get_cpds()[1])

	def marginalise(self, net, vars_to_marginalise, var_to_compute):
		idx = list(net.nodes()).index(var_to_compute)
		marginal = net.get_cpds()[idx].marginalize(vars_to_marginalise, inplace=False)
		if self.debug:
			print(marginal.get_values())

	def exact_inference(self, net, var, evidence):
		# self.bp[net].calibrate()
		var_max = self.bp[net].map_query(variables=[var], evidence=evidence)[var]
		if self.debug:
			print(var_max)
		return var_max
	
	def generate_memory_states(self, rollout_buffer):
		action_triggers = th.zeros((rollout_buffer.buffer_size, rollout_buffer.n_envs), dtype=th.float32)
		epsilon = 1e-3
		obs = rollout_buffer.observations
		theta1_trig = th.as_tensor(np.abs(np.arctan2(obs[:, :, 1], obs[:, :, 0])) < epsilon)
		theta2_trig = th.as_tensor(np.abs(np.arctan2(obs[:, :, 3], obs[:, :, 2])) < epsilon)
		omega1_trig = th.as_tensor(np.abs(obs[:, :, 4]) < epsilon)
		# forced torque application at episode start
		action_triggers = th.logical_or(action_triggers, th.as_tensor(rollout_buffer.episode_starts))
		# trigger conditions
		action_triggers = th.logical_xor(action_triggers, theta1_trig)
		action_triggers = th.logical_xor(action_triggers, theta2_trig)
		action_triggers = th.logical_xor(action_triggers, omega1_trig)
		rollout_buffer.action_triggers = np.asarray(action_triggers)
		# return action_triggers
	
	def encode_abstract_states(self, rollout_data):
		theta_evi = []
		flag_evi = []
		theta1 = th.arctan2(rollout_data.observations[:, 1], rollout_data.observations[:, 0])
		w = rollout_data.observations[:, 4]
		for i in range(rollout_data.observations.shape[0]):
			theta_evi.append('pos_ang' if theta1[i] > 0 else 'neg_ang')
			flag_evi.append('torque' if rollout_data.action_triggers[i] else 'no_torque')
		return [theta_evi, flag_evi]
	
	def compute_intuition_diffs(self, rollout_data):
		actions = rollout_data.actions
		[theta_evi, flag_evi] = self.encode_abstract_states(rollout_data=rollout_data)
		t_diff = th.zeros(rollout_data.observations.shape[0])
		for i in range(rollout_data.observations.shape[0]):
			evi = {'theta': theta_evi[i], 'flag': flag_evi[i]}
			t_diff[i] = 1.0 if ((self.exact_inference('torque_net', 't', evi) == 'neg_t') and (actions[i] != 0)
								or (self.exact_inference('torque_net', 't', evi) == 'no_t') and (actions[i] != 1)
								or (self.exact_inference('torque_net', 't', evi) == 'pos_t') and (actions[i] != 2)) else 0.0
		return [t_diff]
	
	def forward(self, rollout_data):
		[t_diff] = self.compute_intuition_diffs(rollout_data)
		# convert to hinge loss
		actions = rollout_data.actions
		t_diff_max = th.maximum((1 - (t_diff * th.abs(actions.cpu()))), th.zeros(rollout_data.observations.shape[0]))
		intuition_loss = t_diff_max.sum()
		self.save_for_backward(t_diff)
		return intuition_loss

	def backward(self, grad_output):
		t_diff = self.saved_tensors
		return -grad_output * (t_diff)


if __name__ == "__main__":
	net = Acrobot(debug=True)
	net.check_pgm(net.torque_net)
	net.marginalise(net.torque_net, ['theta', 'flag'], 't')
	evi = {'theta': 'neg_ang', 'flag': 'no_torque'}
	net.exact_inference('torque_net', 't', evi)
