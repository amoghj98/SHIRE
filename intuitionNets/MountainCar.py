#!/usr/bin/python

import sys
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import BeliefPropagation, VariableElimination

import torch as th
import numpy as np

import time


class MountainCar(th.autograd.Function):
	def __init__(self, continuous=False, debug=False):
		self.continuous = continuous
		self.debug = debug
		self.firstStep = True
		self.acc_net = BayesianNetwork()
		self.acc_net.add_nodes_from(['v', 'a'])
		self.acc_net.add_edges_from([('v', 'a')])
		#
		v_cpd = TabularCPD('v', 2, [[0.5], [0.5]],	# uniform for now
							state_names={'v': ['neg_v', 'pos_v']})
		a_cpd = TabularCPD('a', 2 , [[0.8, 0.2], [0.2, 0.8]], evidence=['v'], evidence_card=[2],
							state_names={'a': ['neg_a', 'pos_a'],
										'v': ['neg_v', 'pos_v']})
		self.acc_net.add_cpds(v_cpd, a_cpd)
		#
		# self.bp = {'acc_net': BeliefPropagation(self.acc_net)}
		self.ve = {'acc_net': VariableElimination(self.acc_net)}

	def __reduce__(self) -> str | tuple[any, ...]:
		return (self.__class__, (self.debug, ))

	def check_pgm(self, net):
		assert net.check_model() == True, f'ERROR: Inconsistent probability assignments detected in network "{net}"!'
		if self.debug:
			print(self.acc_net.get_cpds()[1])

	def marginalise(self, net, vars_to_marginalise, var_to_compute):
		idx = list(net.nodes()).index(var_to_compute)
		marginal = net.get_cpds()[idx].marginalize(vars_to_marginalise, inplace=False)
		if self.debug:
			print(marginal.get_values())

	def exact_inference(self, net, var, evidence):
		# self.bp[net].calibrate()
		# var_max = self.bp[net].map_query(variables=[var], evidence=evidence)[var]
		# var_max = 'pos_a' if np.argmax(self.ve[net].query(variables=[var], evidence=evidence, joint=False, show_progress=False)[var].values) else 'neg_a'
		var_max = 'pos_a' if np.argmax(self.ve[net].query(variables=[var], evidence=evidence, joint=True, show_progress=False).values) else 'neg_a'
		if self.debug:
			print(var_max)
		return var_max
	
	def encode_abstract_states(self, rollout_data):
		# v_evi = []
		# v = rollout_data.observations[:, 1]
		# v_evi = ['neg_v' if rollout_data.observations[i, 1] < 0 else 'pos_v' for i in range(rollout_data.observations.shape[0])]
		# t1=time.time()
		# v_evi = list(map(lambda i: 'neg_v' if rollout_data.observations[i, 1] < 0 else 'pos_v', range(rollout_data.observations.shape[0])))
		# t2=time.time()
		# print(t2-t1)
		# print(v_evi)
		# t1=time.time()
		v_evi = []
		for i in range(rollout_data.observations.shape[0]):
			v_evi.append('neg_v' if rollout_data.observations[i, 1] < 0 else 'pos_v')
		# t2=time.time()
		# print(v_evi)
		# print(t2-t1)
		return v_evi
	
	def compute_intuition_diffs(self, rollout_data):
		actions = rollout_data.actions
		v_evi = self.encode_abstract_states(rollout_data=rollout_data)
		# a_diff = th.zeros(rollout_data.observations.shape[0])
		# t1=time.time()
		# a_diff = th.tensor(list(map(lambda v, act: 1.0 if ((self.exact_inference('acc_net', 'a', {'v': v}) == 'neg_a') and (act > 0)
										# or (self.exact_inference('acc_net', 'a', {'v': v}) == 'pos_a') and (act < 2)) else 0.0, v_evi, actions)))
		# t2=time.time()
		# print(t2-t1)
		# print(a_diff)
		# t1=time.time()
		if self.continuous:
			a_diff = th.tensor(list(map(lambda i: 1.0 if ((self.exact_inference('acc_net', 'a', {'v': v_evi[i]}) == 'neg_a') and (actions[i] >= 0)
										or (self.exact_inference('acc_net', 'a', {'v': v_evi[i]}) == 'pos_a') and (actions[i] <= 0)) else 0.0, range(rollout_data.observations.shape[0]))))
		else:
			a_diff = th.tensor(list(map(lambda i: 1.0 if ((self.exact_inference('acc_net', 'a', {'v': v_evi[i]}) == 'neg_a') and (actions[i] > 0)
										or (self.exact_inference('acc_net', 'a', {'v': v_evi[i]}) == 'pos_a') and (actions[i] < 2)) else 0.0, range(rollout_data.observations.shape[0]))))
		# t2=time.time()
		# print(a_diff)
		# print(t2-t1)
		# for i in range(rollout_data.observations.shape[0]):
		# 	evi = {'v': v_evi[i]}
		# 	a_diff[i] = 1.0 if ((self.exact_inference('acc_net', 'a', evi) == 'neg_a') and (actions[i] > 0)
		# 						or (self.exact_inference('acc_net', 'a', evi) == 'pos_a') and (actions[i] < 2)) else -1.0
		return [a_diff]
	
	def forward(self, rollout_data):
		[a_diff] = self.compute_intuition_diffs(rollout_data)
		# convert to hinge loss
		actions = rollout_data.actions
		a_diff_max = th.maximum((1 - (a_diff * th.abs(actions.cpu()))), th.zeros(rollout_data.observations.shape[0]))
		intuition_loss = a_diff_max.sum()
		self.save_for_backward(a_diff)
		return intuition_loss
	
	# def forward(self, rollout_data):
	# 	actions = rollout_data.actions
	# 	v_evi = list(map(lambda i: 'neg_v' if rollout_data.observations[i, 1] < 0 else 'pos_v', range(rollout_data.observations.shape[0])))
	# 	a_diff = th.tensor(list(map(lambda i: 1.0 if ((self.exact_inference('acc_net', 'a', {'v': v_evi[i]}) == 'neg_a') and (actions[i] > 0)
	# 									or (self.exact_inference('acc_net', 'a', {'v': v_evi[i]}) == 'pos_a') and (actions[i] < 2)) else -1.0, range(rollout_data.observations.shape[0]))))
	# 	a_diff_max = th.maximum((1 - (a_diff * th.abs(actions.cpu()))), th.zeros(rollout_data.observations.shape[0]))
	# 	intuition_loss = a_diff_max.sum()
	# 	self.save_for_backward(a_diff)
	# 	return intuition_loss

	def backward(self, grad_output):
		a_diff = self.saved_tensors
		return -grad_output * (a_diff) * 0.5


if __name__ == "__main__":
	net = MountainCar(debug=True)
	net.check_pgm(net.acc_net)
	# l_red = net.l_ori_net.get_cpds()[3].reduce([('theta')])
	# net.marginalise(net.l_ori_net, ['theta', 'py', 'px'], 'l')
	net.marginalise(net.acc_net, ['v'], 'a')
	evi = {'v': 'pos_v'}
	t1 = time.time()
	net.exact_inference('acc_net', 'a', evi)
	t2 = time.time()
	print(t2-t1)
	# for ls in net.l_ori_net._node['l'].states:
		# print(net.l_ori_net.get_state_probability({'theta': 'safe', 'py':'caution', 'px':'lDanger', 'l':ls}))
