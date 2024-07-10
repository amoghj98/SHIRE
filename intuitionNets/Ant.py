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


class Ant(th.autograd.Function):
	def __init__(self, safe_z_range, debug=False):
		self.z_min = safe_z_range[0]
		self.z_max = safe_z_range[1]
		self.debug = debug
		self.t_net = BayesianNetwork()
		self.t_net.add_nodes_from(['z', 't_fl', 't_fr', 't_bl', 't_br'])
		self.t_net.add_edges_from([('z', 't_fl'), ('z', 't_fr'), ('z', 't_bl'), ('z', 't_br')])
		z_cpd = TabularCPD('z', 3, [[0.33], [0.34], [0.33]],	# uniform for now
							state_names={'z': ['low_danger', 'safe', 'high_danger']})
		t_fl_cpd = TabularCPD('t_fl', 2, [[0.2, 0.5, 0.8],
										  [0.8, 0.5, 0.2]],
							evidence=['z'], evidence_card=[3],
							state_names={'t_fl': ['neg_t', 'pos_t'],
										 'z': ['low_danger', 'safe', 'high_danger']})
		t_fr_cpd = TabularCPD('t_fr', 2, [[0.2, 0.5, 0.8],
										  [0.8, 0.5, 0.2]],
							evidence=['z'], evidence_card=[3],
							state_names={'t_fr': ['neg_t', 'pos_t'],
										 'z': ['low_danger', 'safe', 'high_danger']})
		t_bl_cpd = TabularCPD('t_bl', 2, [[0.2, 0.5, 0.8],
										  [0.8, 0.5, 0.2]],
							evidence=['z'], evidence_card=[3],
							state_names={'t_bl': ['neg_t', 'pos_t'],
										 'z': ['low_danger', 'safe', 'high_danger']})
		t_br_cpd = TabularCPD('t_br', 2, [[0.2, 0.5, 0.8],
										  [0.8, 0.5, 0.2]],
							evidence=['z'], evidence_card=[3],
							state_names={'t_br': ['neg_t', 'pos_t'],
										 'z': ['low_danger', 'safe', 'high_danger']})
		# self.t_net.add_nodes_from(['theta_fl', 'theta_fr', 'ori', 't_fl', 't_fr', 'comparison'])
		# self.t_net.add_edges_from([('theta_fl', 't_fl'), ('theta_fr', 't_fr'), ('ori', 't_fl'), ('ori', 't_fr'), ('comparison', 't_fl'), ('comparison', 't_fr')])
		#
		# theta_fl_cpd = TabularCPD('theta_fl', 2, [[0.5], [0.5]],	# uniform for now
		# 					state_names={'theta_fl': ['neg_ang', 'pos_ang']})
		# theta_fr_cpd = TabularCPD('theta_fr', 2, [[0.5], [0.5]],	# uniform for now
		# 					state_names={'theta_fr': ['neg_ang', 'pos_ang']})
		# ori_cpd = TabularCPD('ori', 3, [[0.33], [0.34], [0.33]],	# uniform for now
		# 					state_names={'ori': ['neg_ang', 'lt_thresh', 'pos_ang']})
		# comparison_cpd = TabularCPD('comparison', 2, [[0.5], [0.5]],	# uniform for now
		# 					state_names={'comparison': ['l_gt_r', 'l_lt_r']})
		# t_fl_cpd = TabularCPD('t_fl', 2, [[],
		# 							      []],
		# 					evidence=['ori', 'theta_fl', 'comparison'], evidence_card=[3, 2, 2],
		# 					state_names={'t_fl': ['neg_t', 'pos_t'],
        #                                 'ori': ['neg_ang', 'lt_thresh', 'pos_ang'],
		# 			                    'theta_fl': ['neg_ang', 'pos_ang'],
		# 								'comparison': ['l_gt_r', 'l_lt_r']})
		# t_fr_cpd = TabularCPD('t_fl', 2, [[],
		# 							      []],
		# 					evidence=['ori', 'theta_fr', 'comparison'], evidence_card=[3, 2, 2],
		# 					state_names={'t_fr': ['neg_t', 'pos_t'],
        #                                 'ori': ['neg_ang', 'lt_thresh', 'pos_ang'],
		# 			                    'theta_fr': ['neg_ang', 'pos_ang'],
		# 								'comparison': ['l_gt_r', 'l_lt_r']})
		self.t_net.add_cpds(z_cpd, t_fl_cpd, t_fr_cpd, t_bl_cpd, t_br_cpd)
		#
		self.bp = {'t_net': BeliefPropagation(self.t_net)}
		self.ve = {'t_net': VariableElimination(self.t_net)}
		
	def __reduce__(self) -> str | tuple[any, ...]:
		return (self.__class__, (tuple([self.z_min, self.z_max]), self.debug, ))

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
		var_max = 'pos_t' if np.argmax(self.ve[net].query(variables=[var], evidence=evidence, joint=False, show_progress=False)[var].values) else 'neg_t'
		if self.debug:
			print(var_max)
		return var_max
	
	def generate_memory_states(self, rollout_buffer):
		pass
	
	def encode_abstract_states(self, rollout_data):
		z_thresh = 2e-1
		z_evi = []
		z = rollout_data.observations[:, 0]
		for i in range(rollout_data.observations.shape[0]):
			z_evi.append('low_danger' if (z[i] - self.z_min < z_thresh) else ('high_danger' if (self.z_max - z[i] < z_thresh) else 'safe'))
		return [z_evi]
	
	def compute_intuition_diffs(self, rollout_data):
		actions = rollout_data.actions
		# l = [0, 1, 2, 5, 6, 7]
		# for i in l:
		# 	print(f'obs[{i}]: {rollout_data.observations[:, i]}')
		[z_evi] = self.encode_abstract_states(rollout_data=rollout_data)
		t_diff = th.zeros(rollout_data.observations.shape[0])
		for i in range(rollout_data.observations.shape[0]):
			evi = {'z': z_evi[i]}
			# t_diff[i] = 1.0 if ((vx[i] <= 0) or th.any(th.abs(actions[i]) < epsilon*th.ones(size=actions[i].shape).to(device='cuda'))) else 0.0
			t_diff[i] = 1.0 if ((self.exact_inference('t_net', 't_fl', evi) == 'pos_t' and actions[i, 3] < 0)
					   		 or (self.exact_inference('t_net', 't_fl', evi) == 'neg_t' and actions[i, 3] > 0)) else 0.0
			t_diff[i] += 1.0 if ((self.exact_inference('t_net', 't_fr', evi) == 'pos_t' and actions[i, 5] < 0)
					   		 or (self.exact_inference('t_net', 't_fr', evi) == 'neg_t' and actions[i, 5] > 0)) else 0.0
			t_diff[i] += 1.0 if ((self.exact_inference('t_net', 't_bl', evi) == 'pos_t' and actions[i, 7] < 0)
					   		 or (self.exact_inference('t_net', 't_bl', evi) == 'neg_t' and actions[i, 7] > 0)) else 0.0
			t_diff[i] += 1.0 if ((self.exact_inference('t_net', 't_br', evi) == 'pos_t' and actions[i, 1] < 0)
					   		 or (self.exact_inference('t_net', 't_br', evi) == 'neg_t' and actions[i, 1] > 0)) else 0.0
		return [t_diff]
	
	def forward(self, rollout_data):
		[t_diff] = self.compute_intuition_diffs(rollout_data)
		# convert to hinge loss
		actions = rollout_data.actions
		t_diff_max = th.maximum((1 - (t_diff * th.abs(actions).sum(dim=1).cpu())), th.zeros(rollout_data.observations.shape[0]))
		intuition_loss = t_diff_max.sum()
		# self.save_for_backward(t_diff)
		return intuition_loss

	def backward(self, grad_output):
		t_diff = self.saved_tensors
		return -grad_output * (t_diff)


if __name__ == "__main__":
	net = Ant(safe_z_range=(0.2, 1), debug=True)
	net.check_pgm(net.t_net)
	net.marginalise(net.t_net, ['z'], 't_fl')
	evi = {'z': 'low_danger'}
	net.exact_inference('t_net', 't_fl', evi)
