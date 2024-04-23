#!/usr/bin/python

import sys
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD


# lander = BayesianNetwork([('theta', 'theta_ctrl'), ('px', 'theta_ctrl'), ('py', 'theta_ctrl'), ('theta_ctrl', 'l'), ('theta_ctrl', 'r'), ('theta_ctrl', 'm')])
l_ori_net = BayesianNetwork()
l_ori_net.add_nodes_from(['theta', 'py', 'px', 'l'])
l_ori_net.add_edges_from([('theta', 'l'), ('px', 'l'), ('py', 'l')])
#
r_net = BayesianNetwork()
r_net.add_nodes_from(['theta', 'py', 'px', 'r'])
r_net.add_edges_from([('theta', 'r'), ('px', 'r'), ('py', 'r')])
# r_net = BayesianNetwork([('theta', 'r'), ('px', 'r'), ('py', 'r')])
# m_net = BayesianNetwork([('theta', 'm'), ('px', 'm'), ('py', 'm')])
theta_cpd = TabularCPD('theta', 3, [[19/48], [5/24], [19/48]],	# uniform for now
                       state_names={'theta': ['lDanger', 'safe', 'rDanger']})
px_cpd = TabularCPD('px', 3, [[1/3], [1/3], [1/3]],	# uniform for now
                       state_names={'px': ['lDanger', 'safe', 'rDanger']})
py_cpd = TabularCPD('py', 3, [[1/3], [1/3], [1/3]],	# uniform for now
                       state_names={'py': ['safe', 'caution', 'danger']})
l_ori_cpd = TabularCPD('l', 2, [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.5, 0.3, 0.85, 0.4, 0.15, 0.7, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.7, 0.15, 0.6, 0.85, 0.3, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                   evidence=['theta', 'px', 'py'], evidence_card=[3, 3, 3],
                   state_names={'l': ['fire', 'idle'],
								'theta': ['lDanger', 'safe', 'rDanger'],
								'py': ['safe', 'caution', 'danger'],
                                'px': ['lDanger', 'safe', 'rDanger']})
r_cpd = TabularCPD('r', 2, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.7, 0.15, 0.6, 0.85, 0.3, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.5, 0.3, 0.85, 0.4, 0.15, 0.7, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                   evidence=['theta', 'py', 'px'], evidence_card=[3, 3, 3],
                   state_names={'r': ['fire', 'idle'],
								'theta': ['lDanger', 'safe', 'rDanger'],
								'py': ['safe', 'caution', 'danger'],
                                'px': ['lDanger', 'safe', 'rDanger']})
l_ori_net.add_cpds(theta_cpd, px_cpd, py_cpd, l_ori_cpd)
r_net.add_cpds(theta_cpd, px_cpd, py_cpd, r_cpd)


if __name__ == "__main__":
    # print(l_ori_cpd)
    assert l_ori_net.check_model() == True, "ERROR: Inconsistent probability assignments detected in network 'l_ori_net'!"
    l_marginal = l_ori_net.get_cpds()[3].marginalize(['theta', 'py', 'px'], inplace=False)
    l_red = l_ori_net.get_cpds()[3].reduce([('theta')])
    print(l_marginal.get_values())
    # for ls in l_ori_net._node['l'].states:
    	# print(l_ori_net.get_state_probability({'theta': 'safe', 'py':'caution', 'px':'lDanger', 'l':ls}))
