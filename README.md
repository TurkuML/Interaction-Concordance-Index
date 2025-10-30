# IC-index

An implementation of interaction concordance index as presented in [\[1\]](#ref1).

## Installation:

Run  
`pip install ic_index`  
to install from PyPI, or  
`pip install git+https://github.com/TurkuML/Interaction-Concordance-Index`  
to install directly from github.

### Example code for testing in Python interpreter
```
import numpy as np
import ic_index

# Generate random labels
example_Y = np.random.rand(35)
# Generate row indices of the pairs
example_rows = np.repeat(range(5), 7)
# Generate column indices of the pairs
example_cols = np.array(list(range(7))*5)

# Calculate IC-index with random predictions, result in [0, 1]:
print(ic_index.ic_index(example_rows, example_cols, example_Y, np.random.rand(35)))

# Calculate IC-index with correct labels, result 1:
print(ic_index.ic_index(example_rows, example_cols, example_Y, example_Y))

# Calculate IC-index with reversed labels, result 0:
print(ic_index.ic_index(example_rows, example_cols, example_Y, -example_Y))

# Calculate IC-index with constant labels, result 0.5:
print(ic_index.ic_index(example_rows, example_cols, example_Y, np.ones((35))))
```

## References:

\[1\] Pahikkala, T., Numminen, R., Movahedi, P., Karmitsa, N., & Airola, A. (2025). [Interaction Concordance Index: Performance Evaluation for Interaction Prediction Methods](https://arxiv.org/abs/2510.14419). arXiv preprint arXiv:2510.14419. <a name="ref1"></a>


