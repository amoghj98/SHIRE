#!/bin/bash

source $HOME/anaconda3/etc/profile.d/conda.sh

# modify this portion to match your paths
uname=$(hostname)
if [[ $uname == *"cbric-gpu"* ]]; then
    cd /local/a/joshi157/SHIRE/
else
    cd $HOME/SHIRE/
fi

if [[ ! $(conda env list) == *"intuitiveRL"* ]]; then
	echo "[FATAL] No conda environment named 'intuitiveRL' found. Please install the environment using the provided yml files"
	exit -1
fi

conda activate intuitiveRL
##
# python intuitiveTrain.py -a ppo --env LunarLander-v2 --continuous -bsz 4 -e 10 -lr 3e-4 1e-4 8e-5 -lrs 100000 500000 1000000 -cr 0.5 0.4 0.3 0.2 -crs 50000 100000 250000 500000 --totalSteps 500000 -ef 10000 --nCores 4
# python intuitiveTrain.py -a ppo --env LunarLander-v2 --continuous -bsz 4 -e 10 -lr 2e-4 8e-5 3e-5 -lrs 50000 100000 500000 -cr 0.5 0.4 0.3 0.2 -crs 50000 100000 250000 500000 --beta 0.2 0.0 --beta-schedule 50000 500000 --totalSteps 500000 -ef 5000 -ssr 200.0 --nCores 4 --intuitiveEncouragement -icf 20 -ist 100
##
python intuitiveTrain.py -a ppo --env LunarLander-v2 --continuous -bsz 4 -e 10 -lr 1e-5 5e-4 1e-4 3e-5 -lrs 20000 80000 120000 500000 -cr 0.5 0.4 0.3 0.2 -crs 20000 100000 150000 200000 --beta 0.2 0.0 --beta-schedule 20000 200000 --totalSteps 200000 -ef 5000 -ssr 200.0 --nCores 4 --intuitiveEncouragement -icf 20 -ist 100
##
# python intuitiveTrain.py -a ppo --env CartPole-v1 -ns 512 -lr 1e-4 -lrs 100000 -cr 0.5 -crs 100000 --totalSteps 100000 -ef 1024 --nCores 1 --intuitiveEncouragement -icf 20 -ist 500--set-seeds --seed-file-dir 091024123208 --intuitiveEncouragement -icf 1e-6 -ist 195.0
##
# python intuitiveTrain.py -a ppo --env MountainCar-v0 -ns 512 -bsz 4 -lr 1e-3 -lrs 1000000 -cr 0.5 -crs 100000 --totalSteps 1000000 -ef 5000 --nCores 4 --intuitiveEncouragement -icf 10 -ist -150
# python intuitiveTrain.py -a ppo --env MountainCar-v0 --continuous -ns 512 -bsz 4 -e 10 -lr 1e-4 5e-4 -lrs 50000 300000 -cr 0.2 -crs 100000 --beta 0.3 0.0 --beta-schedule 30000 300000 --totalSteps 300000 -ef 5000 -ssr 90.0 --nCores 4 # --intuitiveEncouragement -icf 0.05 -ist 70
##
# python intuitiveTrain.py -a ppo --env Swimmer-v4 -bsz 4 -e 10 -lr 5e-4 5e-5 1e-5 -lrs 100000 600000 1000000 -cr 0.3 -crs 1000000  --totalSteps 5000000 -ef 5000 --beta 0.01 0.0 --beta-schedule 600000 5000000 --nCores 4 -ssr 100.0 --intuitiveEncouragement -icf 0.5 -ist 150.0
##
# python intuitiveTrain.py -a ppo --env Taxi-v3 -bsz 16 -e 10 -ns 1024 -lr 5e-4 3e-4 1e-4 5e-5 -lrs 500000 1500000 2500000 4000000 -cr 0.3 -crs 4000000 --totalSteps 4000000 -ef 5000 --beta 0.06 0.0 --beta-schedule 100000 4000000 --lambd 0.98 --gamma 0.99 --nCores 4 -ssr 8.0 --intuitiveEncouragement -icf 3.0 -ist 8.0
#