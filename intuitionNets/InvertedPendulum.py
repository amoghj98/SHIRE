#!/usr/bin/python

import sys
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import BeliefPropagation, ApproxInference, VariableElimination

import torch as th
import numpy as np

import time


class InvertedPendulum(th.autograd.Function):
	def __init__(self, debug=False):
		self.debug = debug
		self.force_net = BayesianNetwork()
		self.force_net.add_nodes_from(['theta', 'f'])
		self.force_net.add_edges_from([('theta', 'f')])
		#
		theta_cpd = TabularCPD('theta', 2, [[0.5], [0.5]],	# uniform for now
							state_names={'theta': ['l', 'r']})
		f_cpd = TabularCPD('f', 2 , [[0.8, 0.2], [0.2, 0.8]], evidence=['theta'], evidence_card=[2],
							state_names={'f': ['goLeft', 'goRight'],
										'theta': ['l', 'r']})
		self.force_net.add_cpds(theta_cpd, f_cpd)
		#
		# self.bp = {'force_net': BeliefPropagation(self.force_net)}
		self.ve = {'force_net': VariableElimination(self.force_net)}

	def __reduce__(self) -> str | tuple[any, ...]:
		return (self.__class__, (self.debug, ))

	def check_pgm(self, net):
		assert net.check_model() == True, f'ERROR: Inconsistent probability assignments detected in network "{net}"!'
		if self.debug:
			print(self.force_net.get_cpds()[1])

	def marginalise(self, net, vars_to_marginalise, var_to_compute):
		idx = list(net.nodes()).index(var_to_compute)
		marginal = net.get_cpds()[idx].marginalize(vars_to_marginalise, inplace=False)
		if self.debug:
			print(marginal.get_values())

	def exact_inference(self, net, var, evidence):
		# self.bp[net].calibrate()
		# var_max = self.bp[net].map_query(variables=[var], evidence=evidence)[var]
		var_max = 'goRight' if np.argmax(self.ve[net].query(variables=[var], evidence=evidence, joint=False, show_progress=False)[var].values) else 'goLeft'
		if self.debug:
			print(var_max)
		return var_max
	
	def encode_abstract_states(self, rollout_data):
		theta_evi = []
		ori = rollout_data.observations[:, 1]
		for i in range(rollout_data.observations.shape[0]):
			theta_evi.append('r' if ori[i] < 0 else 'l')
		return [theta_evi]
	
	def compute_intuition_diffs(self, rollout_data):
		actions = rollout_data.actions
		theta_evi = self.encode_abstract_states(rollout_data=rollout_data)[0]
		f_diff = th.zeros(rollout_data.observations.shape[0])
		for i in range(rollout_data.observations.shape[0]):
			evi = {'theta': theta_evi[i]}
			f_diff[i] = 1.0 if ((self.exact_inference('force_net', 'f', evi) == 'goLeft') and (actions[i] >= 0)
								or (self.exact_inference('force_net', 'f', evi) == 'goRight') and (actions[i] <= 0)) else 0.0
		return [f_diff]
	
	def forward(self, rollout_data):
		[f_diff] = self.compute_intuition_diffs(rollout_data)
		# convert to hinge loss
		actions = rollout_data.actions
		f_diff_max = th.maximum((1 - (f_diff * th.abs(actions.cpu()))), th.zeros(rollout_data.observations.shape[0]))
		intuition_loss = f_diff_max.sum()
		self.save_for_backward(f_diff)
		return intuition_loss

	def backward(self, grad_output):
		f_diff = self.saved_tensors
		return -grad_output * (f_diff)


if __name__ == "__main__":
	net = InvertedPendulum(debug=True)
	net.check_pgm(net.force_net)
	# l_red = net.l_ori_net.get_cpds()[3].reduce([('theta')])
	# net.marginalise(net.l_ori_net, ['theta', 'py', 'px'], 'l')
	net.marginalise(net.force_net, ['theta'], 'f')
	evi = {'theta': 'r'}
	t1 = time.time()
	net.exact_inference('force_net', 'f', evi)
	t2 = time.time()
	print(t2-t1)
	# for ls in net.l_ori_net._node['l'].states:
		# print(net.l_ori_net.get_state_probability({'theta': 'safe', 'py':'caution', 'px':'lDanger', 'l':ls}))
