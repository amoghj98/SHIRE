#!/usr/bin/python

import sys
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import BeliefPropagation

import torch as th


class MountainCar(th.autograd.Function):
	def __init__(self, debug=False):
		self.debug = debug
		self.firstStep = True
		self.acc_net = BayesianNetwork()
		self.acc_net.add_nodes_from(['v', 'a'])
		self.acc_net.add_edges_from([('v', 'a')])
		#
		theta_cpd = TabularCPD('v', 2, [[0.5], [0.5]],	# uniform for now
							state_names={'v': ['neg_v', 'pos_v']})
		a_cpd = TabularCPD('a', 2 , [[0.8, 0.2], [0.2, 0.8]], evidence=['v'], evidence_card=[2],
							state_names={'a': ['neg_a', 'pos_a'],
										'v': ['neg_v', 'pos_v']})
		self.acc_net.add_cpds(theta_cpd, a_cpd)
		#
		self.bp = {'acc_net': BeliefPropagation(self.acc_net)}

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
		var_max = self.bp[net].map_query(variables=[var], evidence=evidence)[var]
		if self.debug:
			print(var_max)
		return var_max
	
	def encode_abstract_states(self, rollout_data):
		v_evi = []
		v = rollout_data.observations[:, 1]
		for i in range(rollout_data.observations.shape[0]):
			v_evi.append('neg_v' if v[i] < 0 else 'pos_v')
		return [v_evi]
	
	def compute_intuition_loss(self, rollout_data):
		actions = rollout_data.actions
		v_evi = self.encode_abstract_states(rollout_data=rollout_data)[0]
		a_diff = th.zeros(rollout_data.observations.shape[0])
		for i in range(rollout_data.observations.shape[0]):
			evi = {'v': v_evi[i]}
			a_diff[i] = 1.0 if ((self.exact_inference('acc_net', 'a', evi) == 'neg_a') and (actions[i] > 0)
								or (self.exact_inference('acc_net', 'a', evi) == 'pos_a') and (actions[i] < 2)) else 0.0
		return [a_diff]
	
	def forward(self, rollout_data):
		[a_diff] = self.compute_intuition_diffs(rollout_data)
		# convert to hinge loss
		actions = rollout_data.actions
		a_diff_max = th.maximum((1 - (a_diff * th.abs(actions[:, 1].cpu()))), th.zeros(rollout_data.observations.shape[0]))
		intuition_loss = a_diff_max.sum()
		self.save_for_backward(a_diff)
		return intuition_loss

	def backward(self, grad_output):
		a_diff = self.saved_tensors
		return -grad_output * (a_diff.sum())


if __name__ == "__main__":
	net = MountainCar(debug=True)
	net.check_pgm(net.acc_net)
	# l_red = net.l_ori_net.get_cpds()[3].reduce([('theta')])
	# net.marginalise(net.l_ori_net, ['theta', 'py', 'px'], 'l')
	net.marginalise(net.acc_net, ['v'], 'a')
	evi = {'v': 'pos_v'}
	net.exact_inference('acc_net', 'a', evi)
	# for ls in net.l_ori_net._node['l'].states:
		# print(net.l_ori_net.get_state_probability({'theta': 'safe', 'py':'caution', 'px':'lDanger', 'l':ls}))
