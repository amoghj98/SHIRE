#!/usr/bin/python

from .CartPole import *
from .LunarLander import *
from .MountainCar import *
from .Pendulum import *
from .Acrobot import *
from .InvertedPendulum import *
from .Swimmer import *
from .Ant import *
from .Taxi import *

__envs__ = {'CartPole-v1': CartPole, 'LunarLander-v2': LunarLander, 'MountainCar-v0': MountainCar, 'MountainCarContinuous-v0': MountainCar, 'Pendulum-v1': Pendulum, 'Acrobot-v1': Acrobot, 'InvertedPendulum-v4': InvertedPendulum, 'Swimmer-v4': Swimmer, 'Ant-v4': Ant, 'Taxi-v3': Taxi}