import numpy as np

# helper function so the code isn't full of magic index numbers
def separate_indices(indices):
    Y_inds = indices[:,0]
    drug_inds = indices[:,1]
    target_inds = indices[:,2]
    return drug_inds, target_inds, Y_inds

def load_cv_indices(FLAGS):
    fpath = FLAGS.dataset_path + FLAGS.cv_filename
    setting = FLAGS.fourfield_setting
    fold = str(FLAGS.cv_fold)


    rawcsv = np.loadtxt(fpath, delimiter=',', dtype=str)

    # delete first row (column labels)
    # column labels: roundID,subset,setting,index,drugs,targets
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

    # assuming just one random seed per split file
    FLAGS.random_seed = rawcsv[0][0]

    return np.array(training_indices), np.array(validation_indices), np.array(test_indices)

def load_rawtext_data(FLAGS):
    print("Loading data from",FLAGS.dataset_path)
    XD = np.loadtxt(FLAGS.dataset_path+"XD.txt")
    XT = np.loadtxt(FLAGS.dataset_path+"XT.txt")
    Y = np.loadtxt(FLAGS.dataset_path+"Y.txt")
    
    # convert davis data from Kd to pKd
    if FLAGS.is_log:
        Y = -(np.log10(Y/1e9))

    # dump NaN entries and return as 1D array
    # we don't need NaNs here because using ready-made split indices
    Y = Y.ravel()
    Y = Y[np.logical_not(np.isnan(Y))]

    return XD, XT, Y

