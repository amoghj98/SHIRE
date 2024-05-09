#!/usr/bin/python

import sys
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import BeliefPropagation


# lander = BayesianNetwork([('theta', 'theta_ctrl'), ('px', 'theta_ctrl'), ('py', 'theta_ctrl'), ('theta_ctrl', 'l'), ('theta_ctrl', 'r'), ('theta_ctrl', 'm')])

class LunarLander():
	def __init__(self, debug=False):
		self.debug = debug
		self.l_ori_net = BayesianNetwork()
		self.l_ori_net.add_nodes_from(['theta', 'py', 'px', 'l'])
		self.l_ori_net.add_edges_from([('theta', 'l'), ('px', 'l'), ('py', 'l')])
		#
		self.r_ori_net = BayesianNetwork()
		self.r_ori_net.add_nodes_from(['theta', 'py', 'px', 'r'])
		self.r_ori_net.add_edges_from([('theta', 'r'), ('px', 'r'), ('py', 'r')])
		# net.r_ori_net = BayesianNetwork([('theta', 'r'), ('px', 'r'), ('py', 'r')])
		# m_net = BayesianNetwork([('theta', 'm'), ('px', 'm'), ('py', 'm')])
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
		self.bp = {'vel_net': BeliefPropagation(self.vel_net),
			 		'l_ori_net': BeliefPropagation(self.l_ori_net),
					'r_ori_net': BeliefPropagation(self.r_ori_net)}

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
		self.bp[net].calibrate()
		var_max = self.bp[net].map_query(variables=[var], evidence=evidence)[var]
		if self.debug:
			print(var_max)
		return var_max


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
	net.exact_inference('vel_net', 'l', evi)
	net.exact_inference('vel_net', 'm', evi)
	net.exact_inference('vel_net', 'r', evi)
	# for ls in net.l_ori_net._node['l'].states:
		# print(net.l_ori_net.get_state_probability({'theta': 'safe', 'py':'caution', 'px':'lDanger', 'l':ls}))
