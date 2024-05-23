#!/bin/bash

source $HOME/anaconda3/etc/profile.d/conda.sh

cd $HOME/intuitiveRL/
conda activate intuitiveRL
# python intuitiveTrain.py -a ppo --env LunarLander-v2 --continuous -bsz 4 -e 10 -lr 3e-4 1e-4 8e-5 -lrs 100000 500000 1000000 -cr 0.5 0.4 0.3 0.2 -crs 50000 100000 250000 500000 --totalSteps 500000 -ef 10000 --nCores 4
# python intuitiveTrain.py -a sac --env LunarLander-v2 --continuous -bfsz 50000 -ls 5000 --tau 0.995 --gamma 0.999 -bsz 8 -e 200 -lr 5e-3 -lrs 1000000 -cr 0.5 0.4 0.3 0.2 -crs 50000 100000 250000 500000 --totalSteps 1000000 -ef 10000 --nCores 4
# python intuitiveTrain.py -a ppo --env LunarLander-v2 --continuous -bsz 4 -e 10 -lr 2e-4 8e-5 3e-5 -lrs 50000 100000 500000 -cr 0.5 0.4 0.3 0.2 -crs 50000 100000 250000 500000 --totalSteps 500000 -ef 5000 --nCores 4 --intuitiveEncouragement -icf 20
python intuitiveTrain.py -a ppo --env CartPole-v1 -ns 512 -lr 1e-4 -lrs 100000 -cr 0.5 -crs 100000 --totalSteps 100000 -ef 1024 --nCores 1
# python intuitiveTrain.py -a ppo --env CartPole-v1 -ns 512 -lr 1e-4 -lrs 100000 -cr 0.5 -crs 100000 --totalSteps 100000 -ef 1024 --nCores 1 --intuitiveEncouragement -icf 20
# python intuitiveTrain.py -a dqn --env MountainCar-v0 -ns 512 -lr 1e-1 5e-2 -lrs 100000 1000000 -cr 0.5 -crs 100000 --totalSteps 1000000 -ef 5000 --nCores 4
# python intuitiveTrain.py -a sac --env MountainCarContinuous-v0 -ns 1024 -bfsz 100000 -ls 2000 -bsz 64 --tau 0.995 --gamma 0.999 -lr 1e-1 -lrs 1000000 -cr 0.5 -crs 100000 --totalSteps 1000000 -ef 5000
# python intuitiveTrain.py --env MountainCar-v0 -ns 512 -lr 1e-3 -lrs 1000000 -cr 0.5 -crs 100000 --totalSteps 1000000 -ef 5000 --nCores 4 --intuitiveEncouragement -icf 10
# python intuitiveTrain.py -a sac --env Pendulum-v1 -bsz 4 -e 10 -lr 1e-4 5e-5 1e-5 -lrs 100000 500000 1000000 -cr 0.5 0.4 0.3 0.2 -crs 50000 80000 125000 500000 --totalSteps 500000 -ef 10000 --nCores 4
# python intuitiveTrain.py -a ppo --env Pendulum-v1 -bsz 4 -e 10 -lr 1e-4 5e-5 1e-5 -lrs 100000 500000 1000000 -cr 0.5 0.4 0.3 0.2 -crs 50000 100000 250000 500000 --totalSteps 500000 -ef 10000 --nCores 4 --intuitiveEncouragement -icf 100