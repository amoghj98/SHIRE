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


class Swimmer(th.autograd.Function):
	def __init__(self, debug=False):
		self.debug = debug
		self.t1_net = BayesianNetwork()
		self.t2_net = BayesianNetwork()
		self.t1_net.add_nodes_from(['theta1', 't1'])
		self.t2_net.add_nodes_from(['theta2', 't2'])
		self.t1_net.add_edges_from([('theta1', 't1')])
		self.t2_net.add_edges_from([('theta2', 't2')])
		#
		theta1_cpd = TabularCPD('theta1', 3, [[0.33], [0.34], [0.33]],	# uniform for now
							state_names={'theta1': ['neg_ang', 'lt_thresh', 'pos_ang']})
		theta2_cpd = TabularCPD('theta2', 3, [[0.33], [0.34], [0.33]],	# uniform for now
							state_names={'theta2': ['neg_ang', 'lt_thresh', 'pos_ang']})
		t1_cpd = TabularCPD('t1', 2 , [[0.8, 0.5, 0.2],
								 	   [0.2, 0.5, 0.8]],
							evidence=['theta1'], evidence_card=[3],
							state_names={'t1': ['neg_t', 'pos_t'],
										'theta1': ['neg_ang', 'lt_thresh', 'pos_ang']})
		t2_cpd = TabularCPD('t2', 2 , [[0.8, 0.5, 0.2],
								 	   [0.2, 0.5, 0.8]],
							evidence=['theta2'], evidence_card=[3],
							state_names={'t2': ['neg_t', 'pos_t'],
										'theta2': ['neg_ang', 'lt_thresh', 'pos_ang']})
		self.t1_net.add_cpds(theta1_cpd, t1_cpd)
		self.t2_net.add_cpds(theta2_cpd, t2_cpd)
		#
		self.bp = {'t1_net': BeliefPropagation(self.t1_net), 't2_net': BeliefPropagation(self.t2_net)}
		self.ve = {'t1_net': VariableElimination(self.t1_net), 't2_net': VariableElimination(self.t2_net)}
		
	def __reduce__(self) -> str | tuple[any, ...]:
		return (self.__class__, (self.debug, ))

	def check_pgm(self, net):
		assert net.check_model() == True, f'ERROR: Inconsistent probability assignments detected in network "{net}"!'
		if self.debug:
			print(net.get_cpds()[1])

	def marginalise(self, net, vars_to_marginalise, var_to_compute):
		idx = list(net.nodes()).index(var_to_compute)
		marginal = net.get_cpds()[idx].marginalize(vars_to_marginalise, inplace=False)
		if self.debug:
			print(marginal.get_values())

	def exact_inference(self, net, var, evidence):
		# self.bp[net].calibrate()
		# var_max = self.ve[net].map_query(variables=[var], evidence=evidence)[var]
		var_max = 'pos_t' if np.argmax(self.ve[net].query(variables=[var], evidence=evidence, joint=False, show_progress=False)[var].values) else 'neg_t'
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
		theta_thresh = 4e-1
		theta1_evi, theta2_evi = [], []
		theta1 = rollout_data.observations[:, 1] - rollout_data.observations[:, 0]
		theta2 = rollout_data.observations[:, 2] - rollout_data.observations[:, 0]
		for i in range(rollout_data.observations.shape[0]):
			theta1_evi.append('lt_thresh' if th.abs(theta1[i]) <= theta_thresh else ('neg_ang' if theta1[i] < theta_thresh else 'pos_ang'))
			theta2_evi.append('lt_thresh' if th.abs(theta2[i]) <= theta_thresh else ('neg_ang' if theta2[i] < theta_thresh else 'pos_ang'))
		return [theta1_evi, theta2_evi]
	
	def compute_intuition_diffs(self, rollout_data):
		epsilon = 3e-1
		actions = rollout_data.actions
		# l = [0, 1, 2, 5, 6, 7]
		# for i in l:
		# 	print(f'obs[{i}]: {rollout_data.observations[:, i]}')
		[theta1_evi, theta2_evi] = self.encode_abstract_states(rollout_data=rollout_data)
		t_diff = th.zeros(rollout_data.observations.shape[0])
		for i in range(rollout_data.observations.shape[0]):
			evi1 = {'theta1': theta1_evi[i]}
			evi2 = {'theta2': theta2_evi[i]}
			# t_diff[i] = 1.0 if ((vx[i] <= 0) or th.any(th.abs(actions[i]) < epsilon*th.ones(size=actions[i].shape).to(device='cuda'))) else 0.0
			t_diff[i] = 1.0 if (((actions[i, 0] < 0 and self.exact_inference('t1_net', 't1', evi1) == 'pos_t') or (actions[i, 0] > 0 and self.exact_inference('t1_net', 't1', evi1) == 'neg_t'))
					   			or ((actions[i, 1] < 0 and self.exact_inference('t2_net', 't2', evi2) == 'pos_t') or (actions[i, 1] > 0 and self.exact_inference('t2_net', 't2', evi2) == 'neg_t'))) else 0.0
			# t_diff[i] += 0.25 if (th.prod(actions[i]) < 0) else 0.0
		return [t_diff]
	
	def forward(self, rollout_data):
		[t_diff] = self.compute_intuition_diffs(rollout_data)
		# convert to hinge loss
		actions = rollout_data.actions
		t_diff_max = th.maximum((1 - (t_diff * th.abs(actions).sum(dim=1).cpu())), th.zeros(rollout_data.observations.shape[0]))
		intuition_loss = t_diff_max.sum()
		self.save_for_backward(t_diff)
		return intuition_loss

	def backward(self, grad_output):
		t_diff = self.saved_tensors
		return -grad_output * (t_diff)


if __name__ == "__main__":
	net = Swimmer(debug=True)
	net.check_pgm(net.t1_net)
	net.check_pgm(net.t2_net)
	net.marginalise(net.t1_net, ['theta1'], 't1')
	evi = {'theta1': 'pos_ang'}
	net.exact_inference('t1_net', 't1', evi)
	evi = {'theta2': 'pos_ang'}
	net.exact_inference('t2_net', 't2', evi)
