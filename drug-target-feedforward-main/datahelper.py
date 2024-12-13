import numpy as np

# helper function so the code isn't full of magic index numbers
def separate_indices(indices):
    Y_inds = indices[:,0]
    drug_inds = indices[:,1]
    target_inds = indices[:,2]
    return drug_inds, target_inds, Y_inds

def load_cv_indices(setting, fold, splitfile_path):
    fpath = splitfile_path

    rawcsv = np.loadtxt(fpath, delimiter=',', dtype=str)

    # delete first row (column labels)
    rawcsv = np.delete(rawcsv, 0, axis=0)

    training_indices = []
    validation_indices = []
    test_indices = []

    for row in rawcsv:
        if row[2] == 'training':
            if row[3] == setting:
                if row[1] == fold:
                    training_indices.append(row[4:].astype(int))

        if row[2] == 'validation':
            if row[3] == setting:
                if row[1] == fold:
                    validation_indices.append(row[4:].astype(int))

        if row[2] == 'test':
            if row[1] == fold:
                test_indices.append(row[4:].astype(int))
    
    random_seed = rawcsv[0][0]

    return np.array(training_indices), np.array(validation_indices), np.array(test_indices), random_seed

