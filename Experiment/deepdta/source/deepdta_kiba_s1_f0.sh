#!/bin/bash
#SBATCH --job-name=deepdta-davis
#SBATCH --account=Project_2002923
#SBATCH --partition=gpu
#SBATCH --time=9:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=FAIL

module load tensorflow

python run_cv.py --num_windows 32 \
                          --seq_window_lengths 4 8 12 \
                          --smi_window_lengths 4 6 8 \
                          --batch_size 256 \
                          --num_epoch 100 \
			  --learning_rates 0.001 \
                          --log_dir 'logs/' \
			  --metricsearch \
			  --save_freq 5 \
			  --cv_filename 'splits_kiba_RS_2688385916.csv' \
                          --max_seq_len 1000 \
                          --max_smi_len 100 \
                          --dataset_path 'data/kiba/' \
			  --fourfield_setting 'S1' \
			  --cv_fold 0
