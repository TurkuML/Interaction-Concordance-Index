# IC-index
This repository contains the code of the generalization performance evaluation measure called Interaction Concordance Index, which is presented in the paper *Lisää paperin tiedot.*.
It is a novel measure of prediction 
performance both for interaction prediction models and for machine learning algorithms used for inferring such models. The data required by the method are pairwise data indices of the two domains, labels and predictions.

## Files
<table align = "center">
    <tr>
        <th> File name </th>
        <th> File purpose </th>
    </tr>
    <tr>
        <td> IC_index.pyx </td>
        <td> Cython implementation of the method as a function named *InteractionConcordanceIndex*. </td>
    </tr>
    <tr>
        <td> swapped.pyx </td>
        <td> Contains the Cython implementation of *count_swapped* function needed as part of *InteractionConcordanceIndex*. </td>
    </tr>
    <tr>
        <td> setup.py </td>
        <td> File to Cythonize the Cython files. </td>
    </tr>
    <tr>
        <td> example.py </td>
        <td> A toy example of using the *InteractionConcordanceIndex*. </td>
    </tr>
</table>

## Example
In the example, a simple pairwise data set of 5 elements in one domain and 7 in another is generated. The generated data are complete, i.e. the labels are generated for every possible pair, but it does not have to be. The predictions are given to the measure as a matrix, where each column corresponds to the predictions of one method.
*TO DO: modify the code so that the default is that there are predictions only for one method given similarly to the labels.*

### Steps needed to test the method by the example
- Run setup.py.
- Run example.py.
- Notice that the value of the IC-index is 0.5 in the given example.