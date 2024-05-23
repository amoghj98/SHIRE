#!/usr/bin/python

import sys
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import BeliefPropagation

import torch as th


class Pendulum(th.autograd.Function):
	def __init__(self, debug=False):
		self.debug = debug
		self.torque_net = BayesianNetwork()
		self.torque_net.add_nodes_from(['omega', 't'])
		self.torque_net.add_edges_from([('omega', 't')])
		#
		omega_cpd = TabularCPD('omega', 2, [[0.5], [0.5]],	# uniform for now
							state_names={'omega': ['neg_w', 'pos_w']})
		t_cpd = TabularCPD('t', 2 , [[0.8, 0.2], [0.2, 0.8]], evidence=['omega'], evidence_card=[2],
							state_names={'t': ['neg_t', 'pos_t'],
										'omega': ['neg_w', 'pos_w']})
		self.torque_net.add_cpds(omega_cpd, t_cpd)
		#
		self.bp = {'torque_net': BeliefPropagation(self.torque_net)}

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
	
	def encode_abstract_states(self, rollout_data):
		omega_evi = []
		w = rollout_data.observations[:, 1]
		for i in range(rollout_data.observations.shape[0]):
			omega_evi.append('pos_w' if w[i] > 0 else 'neg_w')
		return [omega_evi]
	
	def compute_intuition_diffs(self, rollout_data):
		actions = rollout_data.actions
		omega_evi = self.encode_abstract_states(rollout_data=rollout_data)[0]
		t_diff = th.zeros(rollout_data.observations.shape[0])
		for i in range(rollout_data.observations.shape[0]):
			evi = {'omega': omega_evi[i]}
			t_diff[i] = 1.0 if ((self.exact_inference('torque_net', 't', evi) == 'neg_t') and (actions[i] >= 0)
								or (self.exact_inference('torque_net', 't', evi) == 'pos_t') and (actions[i] < 0)) else 0.0
		return [t_diff]
	
	def forward(self, rollout_data):
		[t_diff] = self.compute_intuition_diffs(rollout_data)
		# convert to hinge loss
		actions = rollout_data.actions
		t_diff_max = th.maximum((1 - (t_diff * th.abs(actions[:, 1].cpu()))), th.zeros(rollout_data.observations.shape[0]))
		intuition_loss = t_diff_max.sum()
		self.save_for_backward(t_diff)
		return intuition_loss

	def backward(self, grad_output):
		t_diff = self.saved_tensors
		return -grad_output * (t_diff.sum())


if __name__ == "__main__":
	net = Pendulum(debug=True)
	net.check_pgm(net.torque_net)
	# l_red = net.l_ori_net.get_cpds()[3].reduce([('omega')])
	# net.marginalise(net.l_ori_net, ['omega', 'py', 'px'], 'neg_w')
	net.marginalise(net.torque_net, ['omega'], 't')
	evi = {'omega': 'neg_w'}
	net.exact_inference('torque_net', 't', evi)
	# for ls in net.l_ori_net._node['neg_w'].states:
		# print(net.l_ori_net.get_state_probability({'omega': 'safe', 'py':'caution', 'px':'lDanger', 'neg_w':ls}))
