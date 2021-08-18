#!/bin/bash
#SBATCH -A research
#SBATCH -n 30
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=1G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_kiba_trial_2.txt

echo "starting"
PYTHONUNBUFFERED=1
python3 kiba_trial.py

echo "finished"

