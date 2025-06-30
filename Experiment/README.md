This folder contains the files that were used to run the experiments in the paper. The functions related to the real-world data sets and creating the splits for the different off-training-set prediction problems are included in the file data.py. There are separate files and folders for the simulation, CGKronRLS and sklearn-style of learning algorithms and deep learning methods. The method provided by Sandor Szedmak is in file ltr_solver_multiview_013.py and its wrapper to the sklearn style in file ltr_wrapper.py. The performance measures on the real-world data sets were calculated by the file performance.py. The function cindex in library RLScore was modified to be suitable for calculating the normalized groupwise C-indices, and is thus given in file cindex_measure.py.

# To repeat the experiment
1. Make sure all the necessary files are in the same folder.
2. Run setup.py in order to be able to use the function InteractionConcordanceIndex in the way how it is used in the experiment files.

## Dependencies
The experimental study was run by using Python 3.8.8 and visualized by using R version 4.4.1. The following libraries are needed for repeating the study according to the following instructions. Other libraries are needed as well, if the predictions are wanted to be calculated.
### Python
- NumPy (1.20.3)
- pandas (2.0.3)
- RLScore (0.8)
- statistics
- multiprocessing
### R
- stringi (1.8.4)
- ggplot2 (3.5.1)
- reshape2 (1.4.4)
- dplyr (1.1.4)
- patchwork (1.2.0)

## Experiment with the real-world data sets
1. Download the predictions from https://seafile.utu.fi/d/894492d8fe1c44dd9dec/.
2. Run performance.py to obtain the performance measure values for the real-world data sets.
3. Visualize the results in R by running the file figures_real.R.
## Simulation study
1. Run simulation.py to repeat the whole simulation study.
2. Use the file figures_simulation.R to summarise and visualize the results.
