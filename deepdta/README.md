# Heavily modified version of [DeepDTA](https://github.com/hkmztrk/DeepDTA/)

*  Compatible with Tensorflow 2
*  Most code is re-written
*  Performance improvements
*  Added functionality for coldstart settings described in [Toward more realistic drug-target interaction predictions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4364066/)
*  Added cross-validation functionality
*  Some processed data saved to file instead of generating them on every run
*  Able to save predictions on every *n*th epoch
*  Does not rely on the [vulnerable](https://docs.python.org/3/library/pickle.html) pickle format

## Dependencies

*  [RLScore](https://github.com/aatapa/RLScore)
	*  RLScore is used for a fast implementation of C-index score
*  Python >= 3.8
*  Tensorflow 2.x
*  Keras 2.x

## Running

TODO: Add scripts to get started immediately

There are example scripts for running in `metricsearch` mode (which basically means saving predictions every nth epoch). These example scripts can be used as a base for other scenarios.

TODO: Add code for automatically processing the saved predictions

Possible hyperparameters:
```bash
# number of convolution windows, identical on both sides of the network
--num_windows
# protein sequence convolution window lengths
--seq_window_lengths
# drug SMILES convolution window lengths
--smi_window_lengths
--learning_rates
```

TODO: Document arguments.py

Davis and KiBA datasets are included in the repository. Example data split files used when running crossvalidation are also included. When creating new split files, note that shell scripts need to be edited to include the files' names. The data is split into train, validation and test sets.

