#!/bin/bash

source $HOME/anaconda3/etc/profile.d/conda.sh

cd $HOME/intuitiveRL/
conda activate intuitiveRL
python intuitiveTrain.py -lr 3e-4 1e-4 8e-5 -lrs 100000 500000 1000000 -cr 0.5 0.4 0.3 0.2 -crs 40000 100000 500000 1000000 --totalSteps 1000000 -ef 2500 --nCores 4