import numpy as np
from IC_index import InteractionConcordanceIndex

# Generate row indices of the pairs.
example_rows = np.repeat(range(5), 7)
# Generate column indices of the pairs.
example_cols = np.array(list(range(7))*5)
# Generate random labels within range [0.0, 1.0).
example_Y = np.random.rand(35)
# Generate 4 columns of random predictions within range [0.0, 1.0) to demonstrate the use with predictions made by several "hypotheses"
# and a column of constant predictions to see that it really produces the expected random IC-index. 
example_P = np.hstack((np.random.rand(35, 4), np.ones((35,1))))

# Calculate the IC-indices at once for all the different "hypotheses".
IC_indices = InteractionConcordanceIndex(example_rows, example_cols, example_Y, example_P)
print(IC_indices)