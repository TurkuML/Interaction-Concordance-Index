# IC-index
This repository contains the code of the generalization performance evaluation measure called Interaction Concordance Index, which is presented in the paper *Lisää paperin tiedot.*.
[//]: # (, configurations, and resources required to run the experiments described in the paper: Interaction Concordance Index: Performance Evaluation for Interaction Prediction Methods.
Interaction concordance index)   
It is a novel measure of prediction 
performance both for interaction prediction models and for machine learning algorithms used for inferring such models. The data required by the method are pairwise data indices of the two domains, labels and predictions.

The code is provided as a Cython implementation for calculating the IC-index in the file IC_index.pyx as a function named InteractionConcordanceIndex. The implementation utilizes the count_swapped function from the file swapped.pyx. Thus both of these files need to be compiled by the setup.py file before being able to use the functions. A small example that demonstrates how to use the InteractionConcordanceIndex is given in file example.py. [//]: # (In addition all the necessary files for repeating the experiments detailed in the paper are given in the separate folder called Experiment.)  

In the example, a simple pairwise data set of 5 elements in one domain and 7 in another is generated. The generated data are complete, i.e. the labels are generated for every possible pair, but it does not have to be. The predictions are given to the measure as a matrix, where each column corresponds to the predictions of one method.
*TO DO: modify the code so that the default is that there are predictions only for one method given similarly to the labels.*
