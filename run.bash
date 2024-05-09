#!/bin/bash

source $HOME/anaconda3/etc/profile.d/conda.sh

cd $HOME/intuitiveRL/
conda activate intuitiveRL
python intuitiveTrain.py --continuous -bsz 4 -e 10 -lr 2e-4 8e-5 3e-5 -lrs 50000 100000 500000 -cr 0.5 0.4 0.3 0.2 -crs 50000 100000 250000 500000 --totalSteps 500000 -ef 10000 --nCores 4 --intuitiveEncouragement -icf 20
# python intuitiveTrain.py --env CartPole-v1 -lr 1e-4 5e-5 -lrs 50000 100000 -cr 0.4 -crs 100000 --totalSteps 100000 -ef 1000 --nCores 2