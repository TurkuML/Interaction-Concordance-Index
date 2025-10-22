# IC-index
This repository contains an implementation of interaction concordance index, presented in [1].

## Files
<table align = "center">
    <tr>
        <th> File name </th>
        <th> File purpose </th>
    </tr>
    <tr>
        <td> ic_index.pyx </td>
        <td> Cython implementation of the method as a function named ic_index. </td>
    </tr>
    <tr>
        <td> swapped.pxi </td>
        <td> Cython implementation of a helper module required by ic_index. </td>
    </tr>
    <tr>
        <td> example.py </td>
        <td> A toy example of using the InteractionConcordanceIndex. </td>
    </tr>
</table>

## Installation:
- Create a new environment for e.g. Anaconda
- Run `pip install git+https://github.com/TurkuML/Interaction-Concordance-Index`.

## Usage:
- import ic_index

### Example
- See the code in example.py

## References:
  [1] Pahikkala, T., Numminen, R., Movahedi, P., Karmitsa, N., & Airola, A. (2025). Interaction Concordance Index: Performance Evaluation for Interaction Prediction Methods. arXiv preprint arXiv:2510.14419.
