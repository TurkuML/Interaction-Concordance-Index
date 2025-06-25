# IC-index
This repository contains the code, configurations, and resources required to run the experiments described in the paper: Interaction Concordance Index: Performance Evaluation for Interaction Prediction Methods. #LISÄÄ TIETO MISTÄ LÖYTYY.
Interaction concordance index is a novel measure of prediction 
performance both for interaction prediction models and for machine learning algorithms used for inferring such models.

This repository provides a Cython implementation for calculating the IC-index in the file IC_index.pyx as a function named InteractionConcordanceIndex. The implementation utilizes the count_swapped function from the file swapped.pyx. Thus both these files need to be compiled by the setup.py file before being able to use them. A small example for seeing how the InteractionConcordanceIndex works is given in file example.py. In addition all the necessary files for repeating the experiments detailed in the paper are given in the folder Experiment.
