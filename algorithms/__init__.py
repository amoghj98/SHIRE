#!/usr/bin/python

from .intuitivePPO import *
from .intuitiveDQN import *
from .intuitiveTD3 import *
from .intuitiveSAC import *

__algorithms__ = {'ppo': intuitivePPO, 'dqn': intuitiveDQN, 'td3': intuitiveTD3, 'sac': intuitiveSAC}