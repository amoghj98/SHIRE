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


class Taxi(th.autograd.Function):
	def __init__(self, debug=False):
		self.debug = debug
		self.locations = {0: [0, 0], 1: [0, 4], 3: [4, 3], 2: [4, 0], 4: [4, 4]}	# r,g,b,y locations in (row, col) format
		self.a_net = BayesianNetwork()
		self.a_net.add_nodes_from(['action', 'h', 'v', 'a'])
		self.a_net.add_edges_from([('action', 'a'), ('h', 'a'), ('v', 'a')])
		action_cpd = TabularCPD('action', 3, [[0.33], [0.34], [0.33]],	# uniform for now
						   		state_names={'action': ['pickup', 'move', 'drop']})
		h_cpd  =TabularCPD('h', 3, [[0.33], [0.34], [0.33]],
							state_names={'h': ['l', 'n', 'r']})
		v_cpd  =TabularCPD('v', 3, [[0.33], [0.34], [0.33]],
							state_names={'v': ['u', 'n', 'd']})
		a_cpd = TabularCPD('a', 6, [[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.41, 0.04, 0.04, 0.80, 0.04, 0.04, 0.41, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
									[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.41, 0.04, 0.04, 0.80, 0.04, 0.04, 0.41, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
									[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.41, 0.04, 0.43, 0.80, 0.43, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
									[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.43, 0.80, 0.43, 0.04, 0.43, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
									[0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
									[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80]],
							evidence=['action', 'h', 'v'], evidence_card=[3, 3, 3],
							state_names={'a': ['d', 'u', 'r', 'l', 'pickup', 'drop'],
										'action': ['pickup', 'move', 'drop'],
										'h': ['l', 'n', 'r'],
										'v': ['u', 'n', 'd']})
		self.a_net.add_cpds(action_cpd, h_cpd, v_cpd, a_cpd)
		#
		self.bp = {'a_net': BeliefPropagation(self.a_net)}
		self.ve = {'a_net': VariableElimination(self.a_net)}
		
	def __reduce__(self) -> str | tuple[any, ...]:
		return (self.__class__, (self.locations, self.debug, ))

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
		v = {0: 'd', 1: 'u', 2: 'r', 3: 'l', 4: 'pickup', 5: 'drop'}
		var_max = v[np.argmax(self.ve[net].query(variables=[var], evidence=evidence, joint=False, show_progress=False)[var].values)]
		if self.debug:
			print(var_max)
		return var_max
	
	def generate_memory_states(self, rollout_buffer):
		pass
	
	def encode_abstract_states(self, rollout_data):
		obs = rollout_data.observations.cpu()
		obs = obs % 4
		dest = th.tensor([self.locations[int((obs)[i])] for i in range(obs.shape[0])])
		obs = th.floor(obs / 4).int()
		obs = obs % 5
		p_loc = th.tensor([self.locations[int((obs)[i])] for i in range(obs.shape[0])])
		in_taxi = th.eq(p_loc, th.tensor([4, 4]))
		in_taxi = th.logical_and(in_taxi[:, 0], in_taxi[:, 1])
		obs = th.floor(obs / 5).int()
		col = obs % 5
		obs = obs /5
		t_loc = th.cat((th.floor(obs), col.clone().detach()), dim=1).int()
		for i in range(obs.shape[0]):
			p_loc[i] = t_loc[i] if p_loc[i] == [4, 4] else p_loc[i]
		hp = (p_loc - t_loc)[:, 0]
		vp = (p_loc - t_loc)[:, 1]
		hd = (dest - t_loc)[:, 0]
		vd = (dest - t_loc)[:, 1]
		action_evi = []
		h_evi = []
		v_evi = []
		for i in range(rollout_data.observations.shape[0]):
			action_evi.append('pickup' if (not in_taxi[i] and th.equal(t_loc[i], p_loc[i])) else ('drop' if (in_taxi[i] and th.equal(t_loc[i], dest[i])) else 'move'))
			h_evi.append('l' if (action_evi[i] == 'move' and ((in_taxi[i] and hd[i] < 0) or (not in_taxi[i] and hp[i] < 0))) else ('r' if (action_evi[i] == 'move' and ((in_taxi[i] and hd[i] > 0) or (not in_taxi[i] and hp[i] > 0))) else 'n'))
			v_evi.append('u' if (action_evi[i] == 'move' and ((in_taxi[i] and vd[i] < 0) or (not in_taxi[i] and vp[i] < 0))) else ('d' if (action_evi[i] == 'move' and ((in_taxi[i] and vd[i] > 0) or (not in_taxi[i] and vp[i] > 0))) else 'n'))
		return [action_evi, h_evi, v_evi]
	
	def compute_intuition_diffs(self, rollout_data):
		actions = rollout_data.actions
		[action_evi, h_evi, v_evi] = self.encode_abstract_states(rollout_data=rollout_data)
		a_diff = th.zeros(rollout_data.observations.shape[0])
		for i in range(rollout_data.observations.shape[0]):
			evi = {'action': action_evi[i], 'h': h_evi[i], 'v': v_evi[i]}
			a = self.exact_inference('a_net', 'a', evi)
			a_diff[i] = 2.0 if ((a == 'pickup' and actions[i] != 4) or (a == 'drop' and actions[i] != 5)) else (1.0 if ((a == 'u' and actions[i] != 1) or (a == 'd' and actions[i] != 0) or (a == 'l' and actions[i] != 3) or (a == 'r' and actions[i] != 2)) else 0.0)
		return [a_diff]
	
	def forward(self, rollout_data):
		[a_diff] = self.compute_intuition_diffs(rollout_data)
		# convert to hinge loss
		actions = rollout_data.actions
		t_diff_max = th.maximum((1 - (a_diff * th.abs(actions).cpu())), th.zeros(rollout_data.observations.shape[0]))
		intuition_loss = t_diff_max.sum()
		# self.save_for_backward(t_diff)
		return intuition_loss

	def backward(self, grad_output):
		t_diff = self.saved_tensors
		return -grad_output * (t_diff)


if __name__ == "__main__":
	x = 0.04 + 0.04 + 0.04 + 0.04 + 0.80 + 0.04
	print(x)
	print(x==1.0)
	net = Taxi(debug=True)
	net.check_pgm(net.a_net)
	# net.marginalise(net.a_net, ['action'], 'a')
	evi = {'action': 'move', 'h': 'l', 'v': 'n'}
	net.exact_inference('a_net', 'a', evi)
