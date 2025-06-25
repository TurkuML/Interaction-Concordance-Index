Below is a basic SBATCH script for CSC that can be used as a template.

Steps to get started:

1. `ssh username@puhti.csc.fi` (or your preferred method)
2. `git clone` this repository to the desired location, for example `/scratch/project_2002923/yourusername/feedforward` (or your preferred method)
3. create a subdirectory e.g. `mkdir launchscripts` in the root of this repository
4. save the below launchscript as e.g. `feedforward_davis_s1.sh` in the `launchscripts` directory
5. `sbatch feedforward_davis_s1.sh 0` to run the script for the first fold
6. remember to adjust the `time` parameter and `mem-per-cpu` parameter if you're using very large datasets or very large hyperparameter grids

```bash
#!/bin/bash
#SBATCH --job-name=feedforward
#SBATCH --account=Project_2002923
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=FAIL

# this code was tested on tensorflow 2.12
# however, it's fairly standard code and should run on any tensorflow 2 subversion
# specify the version as below if the code won't run otherwise
#module load tensorflow/2.12
module load tensorflow

cd ..
python feedforward_regression.py --num_layers 2 3 \
                        --neurons_in_layers 4096,4096,2048 128,128,64 \
                        --dropout_ratio 0.1 0.25 \
                        --batch_size 64 256 \
                        --epochs 200 \
                        --save_freq 5 \
                        --learning_rate 0.005 0.001 \
                        --splits_file 'splits_davis_RS_2688385916.csv' \
                        --dataset 'davis' \
                        --setting 'S1' \
                        --cv_fold $1
```

`for i in {0..8}; do sbatch feedforward_davis_s1.sh ${i}; done` starts a job for each fold in setting 1

Similar script files can be written for other settings. Particularly in setting 4 it may make more sense to save more frequently and go through fewer epochs.

Arguments:
 - `num_layers`: a space-separated list of integers that define the possible network depths to search through
 - `neurons_in_layers`: a space-separated list of comma-separated lists of integers. the first element in a comma-separated list corresponds to the number of neurons in the first layer etc. every list must be at least as long as the largest value in `num_layers` - lists that are too long are ok.
 - `dropout_ratio`: a space-separated list of floats between 0 and 1. dropout regularisation, i.e. what ratio of samples should be dropped out by dropout layers during training
 - `batch_size`: a space-separated list of integers - how many samples in each minibatch
 - `epochs`: how many epochs to train for in total
 - `save_freq`: by default the predictions of the model are saved each epoch. `save_freq` can be used to save the predictions every nth epoch
 - `learning_rate`: a space-separated list of learning rates to try
 - `splits_file`: a file with the information about how the data is divided into cross-validation folds in different settings. one is provided for both datasets in the repository
 - `dataset`: currently "davis" and "kiba" are supported
 - `setting`: S1: known drug, known target, S2: known drug, unknown target, S3: unknown drug, known target, S4: unknown drug, unknown target
 - `cv_fold`: which fold to train for in a particular grid search. the same grid search should be repeated for all folds. valid values: 0 through 8

For the space-separated arguments, every possible combination is tested during training.
