#!/bin/bash
#SBATCH --job-name=graphdta-davis-s1-f0
#SBATCH --account=Project_2002923
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12000
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=FAIL

module load pytorch

# dataset (0: davis, 1: kiba)
# model(0: GINConvNet, 1: GATNet, 2: GAT_GCN, 3: GCNNet)
# cuda:#
# foldnum
# setting
python3 training_cv.py 0 0 0 $1 S3
