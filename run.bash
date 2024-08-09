#!/bin/bash


usage() {
  echo "usage: $0 [-h] [-c CARD_NUMBER]" 1>&2;
  exit 1;
}

CARD=0

while getopts "hc:" opts; do
	case "${opts}" in
		h)	usage;;
		c)	CARD=$OPTARG
			if [[ $CARD < 0 ]]
			then
    			echo "$0: error: Must specify GPU card to run experient on"
				usage
				exit 1;
			fi;;
		*)	usage;;
	esac
done

if [[ $CARD < 0 ]]
then
    echo "$0: error: Must specify GPU card to run experient on"
	usage
	exit 1
fi


source $HOME/anaconda3/etc/profile.d/conda.sh

uname=$(hostname)
if [[ $uname == *"cbric-gpu"* ]]; then
    cd /local/a/joshi157/intuitiveRL/
else
    cd $HOME/intuitiveRL/
fi

conda activate intuitiveRL
# python intuitiveTrain.py -a ppo --env LunarLander-v2 --continuous -bsz 4 -e 10 -lr 3e-4 1e-4 8e-5 -lrs 100000 500000 1000000 -cr 0.5 0.4 0.3 0.2 -crs 50000 100000 250000 500000 --totalSteps 500000 -ef 10000 --nCores 4
# python intuitiveTrain.py -a ppo --env LunarLander-v2 --continuous -bsz 4 -e 10 -lr 2e-4 8e-5 3e-5 -lrs 50000 100000 500000 -cr 0.5 0.4 0.3 0.2 -crs 50000 100000 250000 500000 --beta 0.2 0.0 --beta-schedule 50000 500000 --totalSteps 500000 -ef 5000 -ssr 200.0 --nCores 4 --intuitiveEncouragement -icf 20 -ist 100
##
# python intuitiveTrain.py -a ppo --env LunarLander-v2 --continuous -bsz 4 -e 10 -lr 1e-5 5e-4 1e-4 3e-5 -lrs 20000 80000 120000 500000 -cr 0.5 0.4 0.3 0.2 -crs 20000 100000 150000 200000 --beta 0.2 0.0 --beta-schedule 20000 200000 --totalSteps 200000 -ef 5000 -ssr 200.0 --nCores 4 --intuitiveEncouragement -icf 20 -ist 100
##
# python intuitiveTrain.py -a sac --env LunarLander-v2 --continuous -bfsz 50000 -ls 5000 --tau 0.995 --gamma 0.999 -bsz 8 -e 200 -lr 5e-3 -lrs 1000000 -cr 0.5 0.4 0.3 0.2 -crs 50000 100000 250000 500000 --totalSteps 1000000 -ef 10000 --nCores 4
# python intuitiveTrain.py -a ppo --env CartPole-v1 -ns 512 -lr 1e-4 -lrs 100000 -cr 0.5 -crs 100000 --totalSteps 100000 -ef 1024 --nCores 1 -ist 500
# python intuitiveTrain.py -a ppo --env CartPole-v1 -ns 512 -lr 1e-4 -lrs 100000 -cr 0.5 -crs 100000 --totalSteps 100000 -ef 1024 --nCores 1 --intuitiveEncouragement -icf 20 -ist 500
# python intuitiveTrain.py -a dqn --env MountainCar-v0 -ns 512 -lr 1e-1 5e-2 -lrs 100000 1000000 -cr 0.5 -crs 100000 --totalSteps 1000000 -ef 5000 --nCores 4
# python intuitiveTrain.py -a sac --env MountainCarContinuous-v0 -ns 1024 -bfsz 100000 -ls 2000 -bsz 64 --tau 0.995 --gamma 0.999 -lr 1e-1 -lrs 1000000 -cr 0.5 -crs 100000 --totalSteps 1000000 -ef 5000 --intuitiveEncouragement -icf 20
# python intuitiveTrain.py --env MountainCar-v0 -ns 512 -bsz 4 -lr 1e-3 -lrs 1000000 -cr 0.5 -crs 100000 --totalSteps 1000000 -ef 5000 --nCores 4 --intuitiveEncouragement -icf 10 -ist -150
# python intuitiveTrain.py -a ppo --env MountainCar-v0 --continuous -ns 512 -bsz 4 -e 10 -lr 1e-4 5e-4 -lrs 50000 300000 -cr 0.2 -crs 100000 --beta 0.3 0.0 --beta-schedule 30000 300000 --totalSteps 300000 -ef 5000 -ssr 90.0 --nCores 4 # --intuitiveEncouragement -icf 0.05 -ist 70
# python intuitiveTrain.py -a sac --env Pendulum-v1 -bsz 4 -e 10 -lr 1e-4 5e-5 1e-5 -lrs 100000 500000 1000000 -cr 0.5 0.4 0.3 0.2 -crs 50000 80000 125000 500000 --totalSteps 500000 -ef 10000 --nCores 4
# python intuitiveTrain.py -a td3 --env Pendulum-v1 -bsz 64 -e 10 -lr 1e-4 -lrs 50000000 -cr 0.5 0.4 0.3 0.2 -crs 500000 800000 1250000 5000000 --totalSteps 50000000 -ef 10000 --nCores 4
# python intuitiveTrain.py -a ppo --env Pendulum-v1 -bsz 8 -e 10 -lr 1e-4 5e-5 1e-5 5e-6 -lrs 100000 180000 250000 1000000 -cr 0.5 0.4 0.3 0.2 -crs 50000 100000 250000 500000 --totalSteps 500000 -ef 5000 --nCores 4 --intuitiveEncouragement -icf 100 -ist -200.0
# python intuitiveTrain.py -a ppo --env InvertedPendulum-v4 -bsz 4 -e 10 -lr 1e-4 -lrs 5000000 -cr 0.5 0.4 0.3 0.2 -crs 500000 800000 1250000 5000000 --totalSteps 1000000 -ef 5000 --nCores 4 --intuitiveEncouragement -icf 1 -ist 1000.0
# python intuitiveTrain.py -a ppo --env Acrobot-v1 -bsz 4 -n s 2000 -e 80 -lr 3e-3 -lrs 1000000 -cr 0.2 -crs 500000 --lambd 0.97 --totalSteps 5000000 -ef 20000 --nCores 4
# python intuitiveTrain.py -a ppo --env Acrobot-v1 -bsz 8 -ns 1000 -e 10 -lr 3e-3 -lrs 1000000 -cr 0.2 -crs 500000 --lambd 0.97 --totalSteps 5000000 -ef 10000 --nCores 4
# python intuitiveTrain.py -a ppo --env Acrobot-v1 -bsz 8 -ns 1000 -e 10 -lr 3e-3 -lrs 1000000 -cr 0.2 -crs 500000 --lambd 0.97 --totalSteps 5000000 -ef 10000 --nCores 4 --intuitiveEncouragement -icf 2
# python intuitiveTrain.py -a ppo --env Swimmer-v4 -bsz 4 -e 10 -lr 5e-4 5e-5 1e-5 -lrs 100000 600000 1000000 -cr 0.3 -crs 1000000  --totalSteps 5000000 -ef 5000 --beta 0.01 0.0 --beta-schedule 600000 5000000 --nCores 4 -ssr 100.0 --intuitiveEncouragement -icf 0.5 -ist 150.0
# CUDA_VISIBLE_DEVICES=$CARD python intuitiveTrain.py -a ppo --env Ant-v4 -bsz 8 -e 20 -ns 2048 -lr 1e-4 5e-5 1e-5 5e-6 1e-6 -lrs 50000 500000 2500000 10000000 40000000 -cr 0.3 -crs 10000000  --totalSteps 80000000 -ef 20000 --beta 0.005 0.0 --beta-schedule 500000 10000000 --lambd 0.98 --gamma 0.995 --nCores 8 -ssr 6000.0 --set-seeds --seed-file-dir 071524105033 --intuitiveEncouragement -icf 300 -ist 6000.0
CUDA_VISIBLE_DEVICES=$CARD python intuitiveTrain.py -a ppo --env Taxi-v3 -bsz 8 -e 10 -ns 1024 -lr 1e-3 5e-4 3e-4 -lrs 500000 2000000 4000000 -cr 0.3 -crs 4000000 --totalSteps 4000000 -ef 5000 --beta 0.01 0.005 0.0 --beta-schedule 1000000 1500000 4000000 --lambd 0.98 --gamma 0.99 --nCores 4 -ssr 8.0 --set-seeds --seed-file-dir 080624151058 --intuitiveEncouragement -icf 10 -ist 8.0
#
# python intuitiveTrain.py --env MountainCar-v0 --mode test -nep 2 --loadPath ./best_model/052424120338
# python intuitiveTrain.py --env Swimmer-v4 --mode test -nep 2 --loadPath ./best_model/061724173618
# python intuitiveTrain.py --env Ant-v4 --mode test -nep 2 --loadPath ./best_model/070924162944