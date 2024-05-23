#!/usr/bin/python

from .CartPole import *
from .LunarLander import *
from .MountainCar import *
from .Pendulum import *

__envs__ = {'CartPole-v1': CartPole, 'LunarLander-v2': LunarLander, 'MountainCar-v0': MountainCar, 'Pendulum-v1': Pendulum}