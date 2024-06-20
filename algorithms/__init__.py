#!/usr/bin/python

from .intuitivePPO import *
from .intuitiveDQN import *
from .intuitiveTD3 import *
from .intuitiveSAC import *
from .intuitiveTRPO import *

from .buffers import *

__algorithms__ = {'ppo': intuitivePPO, 'dqn': intuitiveDQN, 'td3': intuitiveTD3, 'sac': intuitiveSAC, 'trpo': intuitiveTRPO}