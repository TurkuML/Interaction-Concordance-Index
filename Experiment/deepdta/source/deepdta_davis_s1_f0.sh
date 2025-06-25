#!/bin/bash
#SBATCH --job-name=deepdta-davis
#SBATCH --account=Project_2002923
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
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
			  --cv_filename 'splits_davis_RS_2688385916.csv' \
                          --max_seq_len 1200 \
                          --max_smi_len 85 \
                          --dataset_path 'data/davis/' \
			  --is_log \
			  --fourfield_setting 'S1' \
			  --cv_fold 0
