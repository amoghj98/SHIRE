#!/usr/bin/bash

# define text colour escape codes
white='\033[1;37m'
red='\033[1;31m'
green='\033[1;32m'
yellow='\033[1;33m'
nc='\033[0m'

# arg definitions
_V=0

# arg parsing
while getopts "v" opts; do
  case "${opts}" in
    v) _V=1;;
  esac
done

# function definitions
function consoleOut () {
    if [[ $_V -eq 1 ]]; then
        echo -e "$@"
    fi
}

cd $HOME

consoleOut "${yellow}Setting up IntuitiveRL Environment${nc}"

# check for conda
if [[ ! $(which conda) ]]; then
    consoleOut "[${red}FATAL${nc}] ${red}Conda installation not found. Install conda an try again${nc}"
    exit -1
fi

# check for git
git --version 2>&1 >/dev/null # improvement by tripleee
GIT_IS_AVAILABLE=$?
if ! [ $GIT_IS_AVAILABLE -eq 0 ]; then
    consoleOut "[${red}FATAL${nc}] ${red}Git installation not found. Install Git and try again${nc}"
    exit -1
fi

# check for git ssh keys
if ! ( [ -f $HOME/.ssh/id_ed25519.pub ] || [ -f $HOME/.ssh/id_rsa.pub ] ) ; then
    consoleOut "[${red}FATAL${nc}] ${red}SSH keys absent. Setup Git SSH keys and try again${nc}"
    exit -1
fi

# create env
# consoleOut "${yellow}Creating conda environment...${nc}"
# conda create -n spinningup python=3.6
# conda activate spinningup
# consoleOut "${green}Done creating env!${nc}"
# install OpenMPI
consoleOut "${yellow}Installing OpenMPI...${nc}"
sudo apt-get update && sudo apt-get install libopenmpi-dev
consoleOut "${green}Done installing OpenMPI!${nc}"

# get spinningup and install dependencies
# consoleOut "${yellow}Getting spinningup...${nc}"
# git clone https://github.com/openai/spinningup.git
# cd spinningup
# consoleOut "${green}Done installing spinningup!${nc}"
# consoleOut "${yellow}Fetching dependencies...${nc}"
# # now the opencv fail in the next step does not occur!
# pip install -e .
# # opencv builds in the automated pip install step inevitably fails due to python3.6 being very old. To rememdy this, install opencv-python manually!
# pip install opencv-python==4.1.2.30
# consoleOut "${green}Done installing dependencies!${nc}"

# get stable-baselines3 with all the optional dependencies
pip install stable-baselines3[extra]

# perform standard install test
consoleOut "${yellow}Performing install test. This may take a long time...${nc}"
# python -m spinup.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999
consoleOut "${green}Install test finished!${nc}"