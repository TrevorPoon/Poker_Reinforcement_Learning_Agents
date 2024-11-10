#!/bin/bash

# Source the Conda setup script to enable Conda commands in non-interactive shells
source /afs/inf.ed.ac.uk/user/s26/s2652867/miniconda3/etc/profile.d/conda.sh

# Activate the 'poker' environment
conda activate /afs/inf.ed.ac.uk/user/s26/s2652867/miniconda3/envs/poker

# Run the Python script
python /afs/inf.ed.ac.uk/user/s26/s2652867/own_project/Poker_Reinforcement_Learning_Agents/training.py
