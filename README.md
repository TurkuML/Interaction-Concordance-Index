# IC-index
This repository contains an implementation of interaction concordance index, presented in [1].

## Files
<table align = "center">
    <tr>
        <th> File name </th>
        <th> File purpose </th>
    </tr>
    <tr>
        <td> IC_index.pyx </td>
        <td> Cython implementation of the method as a function named InteractionConcordanceIndex. </td>
    </tr>
    <tr>
        <td> swapped.pyx </td>
        <td> Contains the Cython implementation of count_swapped function needed as part of InteractionConcordanceIndex. </td>
    </tr>
    <tr>
        <td> setup.py </td>
        <td> File to compile the cython files. </td>
    </tr>
    <tr>
        <td> example.py </td>
        <td> A toy example of using the InteractionConcordanceIndex. </td>
    </tr>
</table>

## Example
In the example, a simple pairwise data set of 5 elements in one domain and 7 in another is generated. The generated data are complete, i.e. the labels are generated for every possible pair, but it does not have to be. The predictions are given to the measure as a matrix, where each column corresponds to the predictions of one method.
*TO DO: modify the code so that the default is that there are predictions only for one method given similarly to the labels.*

### Installation:
- Create a new environment for e.g. Anaconda
- Run `pip install git+https://github.com/TurkuML/Interaction-Concordance-Index`.

## References:
  [1] Pahikkala, T., Numminen, R., Movahedi, P., Karmitsa, N., & Airola, A. (2025). Interaction Concordance Index: Performance Evaluation for Interaction Prediction Methods. arXiv preprint arXiv:2510.14419.
