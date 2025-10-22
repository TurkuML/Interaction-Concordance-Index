# IC-index
This repository contains an implementation of interaction concordance index, presented in [1].

## Installation:
- Create a new environment for e.g. Anaconda
- Run `pip install git+https://github.com/TurkuML/Interaction-Concordance-Index`.

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

# Calculate IC-index with reversed labels, result -1:
print(ic_index.ic_index(example_rows, example_cols, example_Y, -example_Y))

# Calculate IC-index with constant labels, result 0.5:
print(ic_index.ic_index(example_rows, example_cols, example_Y, np.ones((35))))
```

## References:
  [1] Pahikkala, T., Numminen, R., Movahedi, P., Karmitsa, N., & Airola, A. (2025). Interaction Concordance Index: Performance Evaluation for Interaction Prediction Methods. arXiv preprint arXiv:2510.14419.
