#!/usr/bin/python

import sys
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import BeliefPropagation, ApproxInference, VariableElimination
import time

import torch as th
import numpy as np


class LunarLander(th.autograd.Function):
	def __init__(self, debug=False, theta_margin=0.4):
		self.debug = debug
		self.theta_margin = theta_margin
		self.l_ori_net = BayesianNetwork()
		self.l_ori_net.add_nodes_from(['theta', 'py', 'px', 'l'])
		self.l_ori_net.add_edges_from([('theta', 'l'), ('px', 'l'), ('py', 'l')])
		#
		self.r_ori_net = BayesianNetwork()
		self.r_ori_net.add_nodes_from(['theta', 'py', 'px', 'r'])
		self.r_ori_net.add_edges_from([('theta', 'r'), ('px', 'r'), ('py', 'r')])
		theta_cpd = TabularCPD('theta', 3, [[19/48], [5/24], [19/48]],	# uniform for now
							state_names={'theta': ['lDanger', 'safe', 'rDanger']})
		px_cpd = TabularCPD('px', 3, [[1/3], [1/3], [1/3]],	# uniform for now
							state_names={'px': ['lDanger', 'safe', 'rDanger']})
		py_cpd = TabularCPD('py', 3, [[1/3], [1/3], [1/3]],	# uniform for now
							state_names={'py': ['safe', 'caution', 'danger']})
		l_ori_cpd = TabularCPD('l', 2, [[0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.9, 0.5, 0.3, 0.85, 0.4, 0.15, 0.7, 0.5, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
									[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.5, 0.7, 0.15, 0.6, 0.85, 0.3, 0.5, 0.9, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]],
						evidence=['theta', 'px', 'py'], evidence_card=[3, 3, 3],
						state_names={'l': ['fire', 'idle'],
										'theta': ['lDanger', 'safe', 'rDanger'],
										'py': ['safe', 'caution', 'danger'],
										'px': ['lDanger', 'safe', 'rDanger']})
		r_cpd = TabularCPD('r', 2, [[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.5, 0.7, 0.15, 0.6, 0.85, 0.3, 0.5, 0.9, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98],
									[0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.9, 0.5, 0.3, 0.85, 0.4, 0.15, 0.7, 0.5, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]],
						evidence=['theta', 'py', 'px'], evidence_card=[3, 3, 3],
						state_names={'r': ['fire', 'idle'],
										'theta': ['lDanger', 'safe', 'rDanger'],
										'py': ['safe', 'caution', 'danger'],
										'px': ['lDanger', 'safe', 'rDanger']})
		self.l_ori_net.add_cpds(theta_cpd, px_cpd, py_cpd, l_ori_cpd)
		self.r_ori_net.add_cpds(theta_cpd, px_cpd, py_cpd, r_cpd)
		#
		self.vel_net = BayesianNetwork()
		self.vel_net.add_nodes_from(['theta', 'acc', 'l', 'm', 'r'])
		self.vel_net.add_edges_from([('acc', 'l'), ('theta', 'l'), ('acc', 'm'), ('theta', 'm'), ('acc', 'r'), ('theta', 'r')])
		#
		acc_cpd = TabularCPD('acc', 2, [[0.5], [0.5]], state_names={'acc': ['pos_a', 'neg_a']})
		theta_cpd2 = TabularCPD('theta', 4, [[0.4], [0.4], [0.1], [0.1]], state_names={'theta': ['quad1', 'quad2', 'quad3', 'quad4']}) #['ltPB2', 'gtPB2']
		l_v_cpd = TabularCPD('l', 2,
									# [[0.4, 0.8, 0.4, 0.6],
									# [0.6, 0.2, 0.6, 0.4]],
									[[0.7, 0.7, 0.9, 0.1, 0.3, 0.3, 0.9, 0.1],
									[0.3, 0.3, 0.1, 0.9, 0.7, 0.7, 0.1, 0.9]],
									evidence=['acc', 'theta'], evidence_card=[2, 4],
									# evidence_card=[2, 2]
									state_names={'l': ['fire', 'idle'],
					  							'acc':['pos_a', 'neg_a'],
					  							# 'theta':['ltPB2', 'gtPB2']
												'theta':['quad1', 'quad2', 'quad3', 'quad4']})
		m_v_cpd = TabularCPD('m', 2, 
									# [[0.8, 0.2, 0.2, 0.8],
									#   [0.2, 0.8, 0.8, 0.2]],
									[[0.8, 0.2, 0.05, 0.05, 0.2, 0.8, 0.05, 0.05],
									[0.2, 0.8, 0.95, 0.95, 0.8, 0.2, 0.95, 0.95]],
									evidence=['acc', 'theta'], evidence_card=[2, 4],
									# evidence_card=[2, 2]
									state_names={'m': ['fire', 'idle'],
					  							'acc':['pos_a', 'neg_a'],
					  							# 'theta':['ltPB2', 'gtPB2']
												'theta':['quad1', 'quad2', 'quad3', 'quad4']})
		r_v_cpd = TabularCPD('r', 2, 
									# [[0.6, 0.4, 0.8, 0.4],
									#   [0.4, 0.6, 0.2, 0.6]],
									[[0.3, 0.3, 0.1, 0.9, 0.7, 0.7, 0.1, 0.9],
									[0.7, 0.7, 0.9, 0.1, 0.3, 0.3, 0.9, 0.1]],
									evidence=['acc', 'theta'], evidence_card=[2, 4],
									# evidence_card=[2, 2]
									state_names={'r': ['fire', 'idle'],
					  							'acc':['pos_a', 'neg_a'],
					  							# 'theta':['ltPB2', 'gtPB2']
												'theta':['quad1', 'quad2', 'quad3', 'quad4']})
		self.vel_net.add_cpds(theta_cpd2, acc_cpd, l_v_cpd, m_v_cpd, r_v_cpd)
		#
		# self.bp = {'vel_net': BeliefPropagation(self.vel_net),
		# 	 		'l_ori_net': BeliefPropagation(self.l_ori_net),
		# 			'r_ori_net': BeliefPropagation(self.r_ori_net)}
		self.ve = {'vel_net': VariableElimination(self.vel_net),
			 		'l_ori_net': VariableElimination(self.l_ori_net),
					'r_ori_net': VariableElimination(self.r_ori_net)}
		# self.ci = {'vel_net': CausalInference(self.vel_net),
		# 	 		'l_ori_net': CausalInference(self.l_ori_net),
		# 			'r_ori_net': CausalInference(self.r_ori_net)}
		# for net in self.bp.keys():
		# 	self.bp[net].calibrate()

	def __reduce__(self) -> str | tuple[any, ...]:
		return (self.__class__, (self.debug, self.theta_margin))

	def check_pgm(self, net):
		assert net.check_model() == True, f'ERROR: Inconsistent probability assignments detected in network "{net}"!'
		if self.debug:
			print(self.vel_net.get_cpds()[2])
			print(self.vel_net.get_cpds()[3])
			print(self.vel_net.get_cpds()[4])

	def marginalise(self, net, vars_to_marginalise, var_to_compute):
		idx = list(net.nodes()).index(var_to_compute)
		marginal = net.get_cpds()[idx].marginalize(vars_to_marginalise, inplace=False)
		if self.debug:
			print(marginal.get_values())

	def exact_inference(self, net, var, evidence):
		# var_max = self.bp[net].map_query(variables=[var], evidence=evidence, show_progress=False)[var]
		# var_max = self.ve[net].map_query(variables=[var], evidence=evidence, show_progress=False)[var]
		var_max = 'idle' if np.argmax(self.ve[net].query(variables=[var], evidence=evidence, elimination_order='greedy', joint=True, show_progress=False).values) else 'fire'
		# var_max = np.argmax(self.ve[net].query(variables=var, evidence=evidence, elimination_order='greedy', joint=True, show_progress=False).values)
		# var_max = np.argmax(self.ci[net].query(variables=[var], evidence=evidence, show_progress=False).values)
		# var_max = 'idle' if np.argmax(self.ci[net].query(variables=[var], evidence=evidence, show_progress=False).values) else 'fire'
		if self.debug:
			print(var_max)
		return var_max
	
	def encode_abstract_vel_net_states(self, vxd, rollout_data):
		# theta_evi, acc_evi = [], []
		ori = rollout_data.observations[:, 4]
		acc_evi = list(map(lambda i: 'pos_a' if ((vxd[i] > 0) and (rollout_data.observations[i, 2] < vxd[i])) else 'neg_a', range(rollout_data.observations.shape[0])))
		theta_evi = list(map(lambda i: 'quad1' if ((ori[i] > 0) and (ori[i] <= th.pi/2)) else 'quad2' if ((ori[i] > 0) and (ori[i] > th.pi/2)) else 'quad3' if ((ori[i] < 0) and (th.abs(ori[i]) < th.pi/2)) else 'quad4', range(rollout_data.observations.shape[0])))
		# for i in range(rollout_data.observations.shape[0]):
			# theta_evi.append('ltPB2' if th.abs(rollout_data.observations[i, 4] < th.pi/2) else 'gtPB2')
			# theta_evi.append('quad1' if ((ori[i] > 0) and (ori[i] <= th.pi/2)) else 'quad2' if ((ori[i] > 0) and (ori[i] > th.pi/2)) else 'quad3' if ((ori[i] < 0) and (th.abs(ori[i]) < th.pi/2)) else 'quad4')
			# acc_evi.append('pos_a' if ((vxd[i] > 0) and (rollout_data.observations[i, 2] < vxd[i])) else 'neg_a')
		return [theta_evi, acc_evi]
	
	def encode_abstract_ori_net_states(self, delta_theta, rollout_data):
		# theta_evi, x_evi, y_evi = [], [], []
		theta_evi = list(map(lambda i: 'safe' if th.abs(delta_theta[i]) < self.theta_margin else 'rDanger' if delta_theta[i] > self.theta_margin else 'lDanger', range(rollout_data.observations.shape[0])))
		x_evi = list(map(lambda i: 'safe' if th.abs(rollout_data.observations[i, 0]) < 0.5 else 'rDanger' if rollout_data.observations[i, 0] > 0.5 else 'lDanger', range(rollout_data.observations.shape[0])))
		y_evi = list(map(lambda i:'safe' if rollout_data.observations[i, 1] > 1.0 else 'caution' if rollout_data.observations[i, 1] > 0.5 else 'danger', range(rollout_data.observations.shape[0])))
		# for i in range(rollout_data.observations.shape[0]):
		# 	theta_evi.append('safe' if th.abs(delta_theta[i]) < self.theta_margin else 'rDanger' if delta_theta[i] > self.theta_margin else 'lDanger')
		# 	x_evi.append('safe' if th.abs(rollout_data.observations[i, 0]) < 0.5 else 'rDanger' if rollout_data.observations[i, 0] > 0.5 else 'lDanger')
		# 	y_evi.append('safe' if rollout_data.observations[i, 1] > 1.0 else 'caution' if rollout_data.observations[i, 1] > 0.5 else 'danger')
		return [theta_evi, x_evi, y_evi]
	
	def encode_abstract_states(self, rollout_data):
		desired_theta = th.atan2(rollout_data.observations[:, 1], rollout_data.observations[:, 0])
		vxd = rollout_data.observations[:, 3] / th.tan(desired_theta)
		delta_theta = desired_theta - rollout_data.observations[:, 4]
		[theta_evi, acc_evi] = self.encode_abstract_vel_net_states(vxd, rollout_data)
		[theta_evi2, x_evi, y_evi] = self.encode_abstract_ori_net_states(delta_theta, rollout_data)
		return [theta_evi, acc_evi, theta_evi2, x_evi, y_evi]
	
	def compute_intuition_diffs(self, rollout_data):
		actions = rollout_data.actions
		desired_theta = th.atan2(rollout_data.observations[:, 1], rollout_data.observations[:, 0])
		vxd = rollout_data.observations[:, 3] / th.tan(desired_theta)
		delta_theta = desired_theta - rollout_data.observations[:, 4]
		[theta_evi, acc_evi] = self.encode_abstract_vel_net_states(vxd, rollout_data)
		[theta_evi2, x_evi, y_evi] = self.encode_abstract_ori_net_states(delta_theta, rollout_data)
		# [theta_evi, acc_evi, theta_evi2, x_evi, y_evi] = self.encode_abstract_states(rollout_data=rollout_data)
		lat_diff, main_diff = th.zeros(rollout_data.observations.shape[0]), th.zeros(rollout_data.observations.shape[0])
		l_diff, r_diff = th.zeros(rollout_data.observations.shape[0]), th.zeros(rollout_data.observations.shape[0])
		lat_diff = th.tensor(list(map(lambda i: 1.0 if (((self.exact_inference('vel_net', 'l', {'theta': theta_evi[i], 'acc': acc_evi[i]}) == 'fire') and (actions[i, 1] > -0.5)) or ((self.exact_inference('vel_net', 'r', {'theta': theta_evi[i], 'acc': acc_evi[i]}) == 'fire') and (actions[i, 1] < 0.5))) else 0.0, range(rollout_data.observations.shape[0]))))
		main_diff = th.tensor(list(map(lambda i: 4.0 if ((self.exact_inference('vel_net', 'm', {'theta': theta_evi[i], 'acc': acc_evi[i]}) == 'fire') and (actions[i, 0] < 0.5)) else 0.0, range(rollout_data.observations.shape[0]))))
		l_diff = th.tensor(list(map(lambda i: 1.0 if (self.exact_inference('l_ori_net', 'l', {'theta': theta_evi2[i], 'py': y_evi[i], 'px': x_evi[i]}) and actions[i, 1] > -0.5) == 'fire' else 0.0, range(rollout_data.observations.shape[0]))))
		r_diff = th.tensor(list(map(lambda i: 1.0 if (self.exact_inference('r_ori_net', 'r', {'theta': theta_evi2[i], 'py': y_evi[i], 'px': x_evi[i]}) and actions[i, 1] < 0.5) == 'fire' else 0.0, range(rollout_data.observations.shape[0]))))
		# for i in range(rollout_data.observations.shape[0]):
			# evi = {'theta': theta_evi[i], 'acc': acc_evi[i]}
			# lat_diff[i] = 1.0 if (((self.exact_inference('vel_net', 'l', evi) == 'fire') and (actions[i, 1] > -0.5))
			# 					or ((self.exact_inference('vel_net', 'r', evi) == 'fire') and (actions[i, 1] < 0.5))) else 0.0
			# main_diff[i] = 4.0 if ((self.exact_inference('vel_net', 'm', evi) == 'fire') and (actions[i, 0] < 0.5)) else 0.0
			#
			# evi = {'theta': theta_evi2[i], 'py': y_evi[i], 'px': x_evi[i]}
			# l_diff[i] = 1.0 if (self.exact_inference('l_ori_net', 'l', evi) and actions[i, 1] > -0.5) == 'fire' else 0.0
			# r_diff[i] = 1.0 if (self.exact_inference('r_ori_net', 'r', evi) and actions[i, 1] < 0.5) == 'fire' else 0.0
		return [lat_diff, main_diff, l_diff, r_diff]
	
	def forward(self, rollout_data):
		[lat_diff, main_diff, l_diff, r_diff] = self.compute_intuition_diffs(rollout_data)
		# convert to hinge loss
		actions = rollout_data.actions
		lat_diff_max = th.maximum((1 - (lat_diff * th.abs(actions[:, 1].cpu()))), th.zeros(rollout_data.observations.shape[0]))
		main_diff_max = th.maximum((1 - (main_diff * th.abs(actions[:, 0].cpu()))), th.zeros(rollout_data.observations.shape[0]))
		# l_diff_max = th.maximum((1 - (l_diff * th.abs(actions[:, 1].cpu()))), th.zeros(rollout_data.observations.shape[0]))
		# r_diff_max = th.maximum((1 - (r_diff * th.abs(actions[:, 1].cpu()))), th.zeros(rollout_data.observations.shape[0]))
		#
		intuition_loss = 1.0 * (lat_diff_max.sum() + main_diff_max.sum())# + l_diff_max.sum() + r_diff_max.sum()
		self.save_for_backward(lat_diff, main_diff)#, l_diff, r_diff, actions)
		return intuition_loss
	
	# def forward(self, rollout_data):
	# 	actions = rollout_data.actions
	# 	desired_theta = th.atan2(rollout_data.observations[:, 1], rollout_data.observations[:, 0])
	# 	vxd = rollout_data.observations[:, 3] / th.tan(desired_theta)
	# 	delta_theta = desired_theta - rollout_data.observations[:, 4]
	# 	[theta_evi, acc_evi] = self.encode_abstract_vel_net_states(vxd, rollout_data)
	# 	[theta_evi2, x_evi, y_evi] = self.encode_abstract_ori_net_states(delta_theta, rollout_data)
	# 	# [theta_evi, acc_evi, theta_evi2, x_evi, y_evi] = self.encode_abstract_states(rollout_data=rollout_data)
	# 	lat_diff, main_diff = th.zeros(rollout_data.observations.shape[0]), th.zeros(rollout_data.observations.shape[0])
	# 	l_diff, r_diff = th.zeros(rollout_data.observations.shape[0]), th.zeros(rollout_data.observations.shape[0])
	# 	lat_diff = th.tensor(list(map(lambda i: 1.0 if (((self.exact_inference('vel_net', 'l', {'theta': theta_evi[i], 'acc': acc_evi[i]}) == 'fire') and (actions[i, 1] > -0.5)) or ((self.exact_inference('vel_net', 'r', {'theta': theta_evi[i], 'acc': acc_evi[i]}) == 'fire') and (actions[i, 1] < 0.5))) else 0.0, range(rollout_data.observations.shape[0]))))
	# 	main_diff = th.tensor(list(map(lambda i: 4.0 if ((self.exact_inference('vel_net', 'm', {'theta': theta_evi[i], 'acc': acc_evi[i]}) == 'fire') and (actions[i, 0] < 0.5)) else 0.0, range(rollout_data.observations.shape[0]))))
	# 	l_diff = th.tensor(list(map(lambda i: 1.0 if (self.exact_inference('l_ori_net', 'l', {'theta': theta_evi2[i], 'py': y_evi[i], 'px': x_evi[i]}) and actions[i, 1] > -0.5) == 'fire' else 0.0, range(rollout_data.observations.shape[0]))))
	# 	r_diff = th.tensor(list(map(lambda i: 1.0 if (self.exact_inference('r_ori_net', 'r', {'theta': theta_evi2[i], 'py': y_evi[i], 'px': x_evi[i]}) and actions[i, 1] < 0.5) == 'fire' else 0.0, range(rollout_data.observations.shape[0]))))
	# 	lat_diff_max = th.maximum((1 - (lat_diff * th.abs(actions[:, 1].cpu()))), th.zeros(rollout_data.observations.shape[0]))
	# 	main_diff_max = th.maximum((1 - (main_diff * th.abs(actions[:, 0].cpu()))), th.zeros(rollout_data.observations.shape[0]))
	# 	l_diff_max = th.maximum((1 - (l_diff * th.abs(actions[:, 1].cpu()))), th.zeros(rollout_data.observations.shape[0]))
	# 	r_diff_max = th.maximum((1 - (r_diff * th.abs(actions[:, 1].cpu()))), th.zeros(rollout_data.observations.shape[0]))
	# 	#
	# 	intuition_loss = lat_diff_max.sum() + main_diff_max.sum() + l_diff_max.sum() + r_diff_max.sum()
	# 	self.save_for_backward(lat_diff, main_diff, l_diff, r_diff, actions)
	# 	return intuition_loss
	
	def backward(self, grad_output):
		print("here!")
		# lat_diff, main_diff, l_diff, r_diff, actions = self.saved_tensors
		lat_diff, main_diff = self.saved_tensors
		return -grad_output * (lat_diff + main_diff) * 0.15# + l_diff + r_diff)
		# return -grad_output * (th.dot(lat_diff, actions[:, 1]) + th.dot(main_diff, actions[:, 0]) + th.dot(l_diff, actions[:, 1]) + th.dot(r_diff, actions[:, 1]))


if __name__ == "__main__":
	net = LunarLander(debug=True)
	# print(l_ori_cpd)
	# net.check_pgm(net.l_ori_net)
	# net.check_pgm(net.r_ori_net)
	net.check_pgm(net.vel_net)
	# l_red = net.l_ori_net.get_cpds()[3].reduce([('theta')])
	# net.marginalise(net.l_ori_net, ['theta', 'py', 'px'], 'l')
	net.marginalise(net.vel_net, ['theta', 'acc'], 'l')
	net.marginalise(net.vel_net, ['theta', 'acc'], 'm')
	net.marginalise(net.vel_net, ['theta', 'acc'], 'r')
	# bp_l = BeliefPropagation(net.l_ori_net)
	# bp_r = BeliefPropagation(net.r_ori_net)
	# l_max = bp_l.map_query(variables=['l'], evidence={'py':'caution', 'px':'lDanger', 'theta':'rDanger'})
	# print(l_max)
	# r_max = bp_r.map_query(variables=['r'], evidence={'py':'caution', 'px':'lDanger', 'theta':'rDanger'})
	# print(r_max)
	evi = {'theta': 'quad1', 'acc': 'pos_a'}
	t1 = time.time()
	net.exact_inference('vel_net', ['l', 'm'], evi)
	net.exact_inference('vel_net', ['m'], evi)
	net.exact_inference('vel_net', ['r'], evi)
	# net.approx_inference('vel_net', 'l', evi)
	# net.approx_inference('vel_net', 'm', evi)
	# net.approx_inference('vel_net', 'r', evi)
	t2 = time.time()
	print(t2-t1)
	# for ls in net.l_ori_net._node['l'].states:
		# print(net.l_ori_net.get_state_probability({'theta': 'safe', 'py':'caution', 'px':'lDanger', 'l':ls}))
