#! /bin/bash
# set -e
############ HOW TO RUN ####################
# run this script WITHOUT sudo and give    # 
# your git acc and login data, otherwise   #
# the repository can't be setup for you!   #
############################################

############### SCRIPT PARAMETERS

current_root=$(pwd)
echo $current_root
VENV_NAME=beam11_venv

############### CONFIGURE LAPTOP
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get -y install python3.11 python3.11-pip python3.11-venv

############### CONFIGURE PYTHON ENVIRONMENT
echo '-------------> Configuring python environemnt' 
echo 'Creating venv and activating it' 
python3.10 -m venv $VENV_NAME
source $VENV_NAME/bin/activate

echo 'installing dependencies'
pip install --upgrade pip
python -m pip install -r requirements.txt

echo 'DONE'