import numpy as np
from IC_index import InteractionConcordanceIndex

# Generate row indices of the pairs.
example_rows = np.repeat(range(5), 7)
# Generate column indices of the pairs.
example_cols = np.array(list(range(7))*5)
# Generate random labels within range [0.0, 1.0).
example_Y = np.random.rand(35)
# Generate constant predictions. Note that the function InteractionConcordanceIndex requires the predictions is a two-dimensional shape. 
# If there is more than one column, the method calculates the IC-indices separately for each column and thus returns as many values as 
# there are columns in the prediction matrix.
example_P = np.ones((35,1))

# Calculate the IC-indices at once for all the different "hypotheses".
IC_indices = InteractionConcordanceIndex(example_rows, example_cols, example_Y, example_P)
print(IC_indices)