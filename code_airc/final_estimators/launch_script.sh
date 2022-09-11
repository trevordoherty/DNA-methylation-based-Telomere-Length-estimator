#!/bin/sh

#SBATCH --job-name=rf_2_adj_tl_unadj_betas
#SBATCH --gres=gpu:0
#SBATCH --mem=350000
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=ADAPT-03
#SBATCH --partition=MEDIUM-G1

. /home/ICTDOMAIN/d18129068/feature_selection/feature_selection_venv/bin/activate

python3 test_sets_evaluation.py
