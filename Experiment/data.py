import numpy as np
import pandas as pd
import itertools as it

"""
Function to load an incomplete data set introduced by Metz et al. (2011).
Returns the data matrices and lists of drug and target indices for the known pairs.
"""
def load_metz():
    Y = np.loadtxt("../Data sets/known_drug-target_interaction_affinities_pKi__Metz_et_al.2011.txt")
    XD = np.loadtxt("../Data sets/drug-drug_similarities_2D__Metz_et_al.2011.txt")
    XT = np.loadtxt("../Data sets/target-target_similarities_WS_normalized__Metz_et_al.2011.txt")
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

"""
Function to load a complete data set introduced by Davis et al. (2011).
Returns the data matrices and lists of drug and target indices for the known pairs.
The matrix of drug similarities is multiplied by 100 in order to obtain the same
range as in the corresponding matrix of Metz data.
The returned continuous labels are natural logarithm of the Kd values so that the
range is again similar to the range of continuous labels in Metz data.
"""
def load_davis():
    Y = np.loadtxt("../Data sets/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt")
    XD = np.loadtxt("../Data sets/drug-drug_similarities_2D__Davis_et_al.2011.txt")
    XT = np.loadtxt("../Data sets/target-target_similarities_WS_normalized__Davis_et_al.2011.txt")
    XD = 100*XD
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    Y = -1*np.log10(Y/1e9)

    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

"""
Function to load an incomplete data set introduced by Merget et al. (2017) and
updated by Cichonska et al (2018).
Returns the data matrices and lists of drug and target indices for the known pairs.
The matrices of drug similarities and target similaritied are multiplied by 100 in
order to obtain the same range as in the corresponding matrices of Metz data.
"""
def load_merget():
    Y = np.loadtxt("../Data sets/Merget_DTIs_2967com_226kin.txt")
    XD = np.loadtxt("../Data sets/Kd_Tanimoto-shortestpath.txt")
    XT = np.loadtxt("../Data sets/Kp_GS-ATP_L5_Sp4.0_Sc4.0.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

"""
Function to load an incomplete data set introduced by
Returns the data matrices and lists of drug and target indices for the known pairs.
The matrices of drug similarities and target similaritied are multiplied by 100 in
order to obtain the same range as in the corresponding matrices of Metz data.
Files kiba_binding_affinity_v2.txt and kiba_drug_sim.txt are slightly modifief in R
because their last columns were such that all values were "NA" and the numbers of
rows and columns did not match.
"""
def load_kiba():
    Y = np.loadtxt("../Data sets/kiba_binding_affinity_v2.txt")
    XD = np.loadtxt("../Data sets/kiba_drug_sim.txt")
    XT = np.loadtxt("../Data sets/kiba_target_sim.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

def load_GPCR():
    Y = np.loadtxt("../Data sets/gpcr_admat_dgc.txt")
    XD = np.loadtxt("../Data sets/gpcr_simmat_dc.txt")
    XT = np.loadtxt("../Data sets/gpcr_simmat_dg.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

def load_IC():
    Y = np.loadtxt("../Data sets/ic_admat_dgc.txt")
    XD = np.loadtxt("../Data sets/ic_simmat_dc.txt")
    XT = np.loadtxt("../Data sets/ic_simmat_dg.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

def load_E():
    Y = np.loadtxt("../Data sets/e_admat_dgc.txt")
    XD = np.loadtxt("../Data sets/e_simmat_dc.txt")
    XT = np.loadtxt("../Data sets/e_simmat_dg.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')


"""
Function to split training data for IDIT training and validation by
dividing the drugs and targets as equally to the parts as possible,
but adding a pair primarily to training data and secondarily to validation
data if its drug or target is not yet included in training or validation
drugs and targets.
"""
def split_both_in(training_inds, drug_inds, target_inds):

    inds1 = []
    inds2 = []
    D_1 = set()
    T_1 = set()
    D_2 = set()
    T_2 = set()

    for j in training_inds:
        drug_j = drug_inds[j]
        target_j = target_inds[j]
        # Add pair to subset 1 if its drug or its target is not yet included in the corresponding sets.
        if drug_j not in D_1 or target_j not in T_1:
            inds1.append(j)
            D_1.add(drug_j)
            T_1.add(target_j)
        # Add pair to subset 2 if its drug or its target is not yet included in the corresponding sets.
        elif drug_j not in D_2 or target_j not in T_2:
            inds2.append(j)
            D_2.add(drug_j)
            T_2.add(target_j)
        # Add pair to the smallest set if both its drug and its target are already included in the corresponding sets.
        else:
            n_1 = len(inds1)
            n_2 = len(inds2)
            if n_1 <= n_2:
                inds1.append(j)
                D_1.add(drug_j)
                T_1.add(target_j)
            else:
                inds2.append(j)
                D_2.add(drug_j)
                T_2.add(target_j)

    return inds1, inds2

"""
Split the training data into IDOT/ODIT training and validation sets by
splitting the drugs or targets into two subsets and defining the first
part to be the set of training drugs or targets and the second part to be
the set of validation drugs or targets.
"""
def split_one_in_one_off(training_inds, drug_target_inds):
    drugs_targets = list(set(drug_target_inds[training_inds]))
    n = len(drugs_targets)
    drugs_targets_train = drugs_targets[:int(n/2)]
    inds1 = []
    inds2 = []
    for j in training_inds:
        if drug_target_inds[j] in drugs_targets_train:
            inds1.append(j)
        else:
            inds2.append(j)

    return inds1, inds2

"""
Split the training data into ODOT training and validation sets by
splitting the drugs and targets into two subsets and defining the
training data to be such pairs whose drug and target are both
included in the sets of training drugs and training targets.
"""
def split_both_off(training_inds, drug_inds, target_inds):
    drugs = list(set(drug_inds[training_inds]))
    targets = list(set(target_inds[training_inds]))
    n_D = len(drugs)
    n_T = len(targets)
    D_train = drugs[:int(n_D/2)]
    T_train = targets[:int(n_T/2)]
    inds1 = []
    inds2 = []
    for j in training_inds:
        drug_j = drug_inds[j]
        target_j = target_inds[j]
        if drug_j in D_train and target_j in T_train:
            inds1.append(j)
        elif drug_j not in D_train and target_j not in T_train:
            inds2.append(j)

    return inds1, inds2


"""
Function to create the splits for the four settings so that there is one common test set for every setting.
IDIT: Both the drugs and targets in the test/validation pairs are also included in the training data.
IDOT: The drugs in the test/validation pairs are included in the training data, but the targets are new.
ODIT: The targets in the test/validation pairs are included in the training data, but the drugs are new.
ODOT: Neither the drugs nor the targets in the test/validation pairs are included in the training data.
"""
def splits(drug_inds, target_inds, split_percentage, random_seed):
    n_sample = len(drug_inds)
    drugs = list(set(drug_inds))
    n_drugs = len(drugs)
    targets = list(set(target_inds))
    n_targets = len(targets)

    np.random.seed(random_seed)
    # Shuffle the unique drugs and split them so that a wanted percentage of them are considered as area IDIT drugs,
    # rest of them are split in halves into test and validation drugs.
    np.random.shuffle(drugs)
    index_drugs = int(np.ceil(n_drugs*(1-split_percentage)))
    drugs_test = drugs[index_drugs:]

    # Shuffle the unique targets and split them so that a wanted percentage of them are considered as area IDIT targets,
    # rest of them are split in halves into test and validation targets.
    np.random.shuffle(targets)
    index_targets = int(np.ceil(n_targets*split_percentage))
    targets_test = targets[:index_targets]

    # Create the test and validation sets of indices for every setting.
    test_inds = []
    IDIT_inds = []
    IDOT_inds = []
    ODIT_inds = []
    ODOT_inds = []
    for i in range(n_sample):
        drug_i = drug_inds[i]
        target_i = target_inds[i]
        if drug_i in drugs_test:
            if target_i in targets_test:
                test_inds.append(i)
            else:
                IDIT_inds.append(i)
                IDOT_inds.append(i)
        else:
            if target_i in targets_test:
                IDIT_inds.append(i)
                ODIT_inds.append(i)
            else:
                ODOT_inds.append(i)

    np.random.shuffle(IDIT_inds)
    np.random.shuffle(IDOT_inds)
    np.random.shuffle(ODIT_inds)
    np.random.shuffle(ODOT_inds)
    IDIT_training, IDIT_validation = split_both_in(IDIT_inds, drug_inds, target_inds)
    IDOT_training, IDOT_validation = split_one_in_one_off(IDOT_inds, target_inds)
    ODIT_training, ODIT_validation = split_one_in_one_off(ODIT_inds, drug_inds)
    ODOT_training, ODOT_validation = split_both_off(ODOT_inds, drug_inds, target_inds)

    IDIT_split = [IDIT_training, test_inds, IDIT_validation]
    IDOT_split = [IDOT_training, test_inds, IDOT_validation]
    ODIT_split = [ODIT_training, test_inds, ODIT_validation]
    ODOT_split = [ODOT_training, test_inds, ODOT_validation]
    splits = [IDIT_split, IDOT_split, ODIT_split, ODOT_split]

    df_indices = pd.concat([pd.DataFrame({'random_seed':random_seed, 'subset':'test', 'setting':'all', 'index':test_inds, 'ID_d':drug_inds[test_inds], 'ID_t':target_inds[test_inds]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'training', 'setting':'IDIT', 'index':IDIT_training, 'ID_d':drug_inds[IDIT_training], 'ID_t':target_inds[IDIT_training]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'validation', 'setting':'IDIT', 'index':IDIT_validation, 'ID_d':drug_inds[IDIT_validation], 'ID_t':target_inds[IDIT_validation]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'training', 'setting':'IDOT', 'index':IDOT_training, 'ID_d':drug_inds[IDOT_training], 'ID_t':target_inds[IDOT_training]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'validation', 'setting':'IDOT', 'index':IDOT_validation, 'ID_d':drug_inds[IDOT_validation], 'ID_t':target_inds[IDOT_validation]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'training', 'setting':'ODIT', 'index':ODIT_training, 'ID_d':drug_inds[ODIT_training], 'ID_t':target_inds[ODIT_training]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'validation', 'setting':'ODIT', 'index':ODIT_validation, 'ID_d':drug_inds[ODIT_validation], 'ID_t':target_inds[ODIT_validation]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'training', 'setting':'ODOT', 'index':ODOT_training, 'ID_d':drug_inds[ODOT_training], 'ID_t':target_inds[ODOT_training]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'validation', 'setting':'ODOT', 'index':ODOT_validation, 'ID_d':drug_inds[ODOT_validation], 'ID_t':target_inds[ODOT_validation]})],
        axis = 0, sort = False)
    return df_indices, splits

"""
Function to create the splits for the four settings so that there is one common test set for every setting.
IDIT: Both the drugs and targets in the test pairs are also included in the training data.
IDOT: The drugs in the test pairs are included in the training data, but the targets are new.
ODIT: The targets in the test pairs are included in the training data, but the drugs are new.
ODOT: Neither the drugs nor the targets in the test pairs are included in the training data.
"""
def train_test_splits(drug_inds, target_inds, split_percentage, random_seed):
    n_sample = len(drug_inds)
    drugs = list(set(drug_inds))
    n_drugs = len(drugs)
    targets = list(set(target_inds))
    n_targets = len(targets)

    np.random.seed(random_seed)
    # Shuffle the unique drugs and split them so that a wanted percentage of them are considered as area IDIT drugs,
    # rest of them are split in halves into test and validation drugs.
    np.random.shuffle(drugs)
    index_drugs = int(np.ceil(n_drugs*(split_percentage)))
    drugs_test = drugs[index_drugs:]

    # Shuffle the unique targets and split them so that a wanted percentage of them are considered as area IDIT targets,
    # rest of them are split in halves into test and validation targets.
    np.random.shuffle(targets)
    index_targets = int(np.ceil(n_targets*split_percentage))
    targets_test = targets[:index_targets]

    # Create the test and training sets of indices for every setting.
    test_inds = []
    IDIT_inds = []
    IDOT_inds = []
    ODIT_inds = []
    ODOT_inds = []
    for i in range(n_sample):
        drug_i = drug_inds[i]
        target_i = target_inds[i]
        if drug_i in drugs_test:
            if target_i in targets_test:
                test_inds.append(i)
            else:
                IDIT_inds.append(i)
                IDOT_inds.append(i)
        else:
            if target_i in targets_test:
                IDIT_inds.append(i)
                ODIT_inds.append(i)
            else:
                ODOT_inds.append(i)

    """
    Uncomment the following part if the training set needs to be further split into 
    a smaller training and a validation sets. Uncomment also the lines for having 
    all of the indices in a data frame. 
    """
    # np.random.shuffle(IDIT_inds)
    # np.random.shuffle(IDOT_inds)
    # np.random.shuffle(ODIT_inds)
    # np.random.shuffle(ODOT_inds)
    # IDIT_training, IDIT_validation = split_both_in(IDIT_inds, drug_inds, target_inds)
    # IDOT_training, IDOT_validation = split_one_in_one_off(IDOT_inds, target_inds)
    # ODIT_training, ODIT_validation = split_one_in_one_off(ODIT_inds, drug_inds)
    # ODOT_training, ODOT_validation = split_both_off(ODOT_inds, drug_inds, target_inds)

    IDIT_split = [IDIT_inds, test_inds]
    IDOT_split = [IDOT_inds, test_inds]
    ODIT_split = [ODIT_inds, test_inds]
    ODOT_split = [ODOT_inds, test_inds]
    splits = [IDIT_split, IDOT_split, ODIT_split, ODOT_split]

    df_indices = pd.concat([pd.DataFrame({'random_seed':random_seed, 'subset':'test', 'setting':'all', 'index':test_inds, 'ID_d':drug_inds[test_inds], 'ID_t':target_inds[test_inds]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'training', 'setting':'IDIT', 'index':IDIT_inds, 'ID_d':drug_inds[IDIT_inds], 'ID_t':target_inds[IDIT_inds]}),
        # pd.DataFrame({'random_seed':random_seed, 'subset':'validation', 'setting':'IDIT', 'index':IDIT_validation, 'ID_d':drug_inds[IDIT_validation], 'ID_t':target_inds[IDIT_validation]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'training', 'setting':'IDOT', 'index':IDOT_inds, 'ID_d':drug_inds[IDOT_inds], 'ID_t':target_inds[IDOT_inds]}),
        # pd.DataFrame({'random_seed':random_seed, 'subset':'validation', 'setting':'IDOT', 'index':IDOT_validation, 'ID_d':drug_inds[IDOT_validation], 'ID_t':target_inds[IDOT_validation]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'training', 'setting':'ODIT', 'index':ODIT_inds, 'ID_d':drug_inds[ODIT_inds], 'ID_t':target_inds[ODIT_inds]}),
        # pd.DataFrame({'random_seed':random_seed, 'subset':'validation', 'setting':'ODIT', 'index':ODIT_validation, 'ID_d':drug_inds[ODIT_validation], 'ID_t':target_inds[ODIT_validation]}),
        pd.DataFrame({'random_seed':random_seed, 'subset':'training', 'setting':'ODOT', 'index':ODOT_inds, 'ID_d':drug_inds[ODOT_inds], 'ID_t':target_inds[ODOT_inds]}),
        # pd.DataFrame({'random_seed':random_seed, 'subset':'validation', 'setting':'ODOT', 'index':ODOT_validation, 'ID_d':drug_inds[ODOT_validation], 'ID_t':target_inds[ODOT_validation]})
        ],
        axis = 0, sort = False)
    return df_indices, splits


"""
Function to split data into 9 folds by randomly splitting drugs and targets into three subsets.
"""
def cv_folds(drug_inds, target_inds, random_seed, split_percentage = 1.0/3):
    n_sample = len(drug_inds)
    drugs = list(set(drug_inds))
    n_drugs = len(drugs)
    targets = list(set(target_inds))
    n_targets = len(targets)

    np.random.seed(random_seed)
    # Shuffle the unique drugs and split them into three equally big subsets.
    np.random.shuffle(drugs)
    index_drugs_1 = int(np.ceil(n_drugs*split_percentage))
    drugs_1 = drugs[:index_drugs_1]
    index_drugs_2 = index_drugs_1 + int(np.ceil((n_drugs-index_drugs_1)*0.5))
    drugs_2 = drugs[index_drugs_1:index_drugs_2]
    drugs_3 = drugs[index_drugs_2:]

    # Shuffle the unique targets and split them into three equally big subsets.
    np.random.shuffle(targets)
    index_targets_1 = int(np.ceil(n_targets*split_percentage))
    targets_1 = targets[:index_targets_1]
    index_targets_2 = index_targets_1 + int(np.ceil((n_targets-index_targets_1)*0.5))
    targets_2 = targets[index_targets_1:index_targets_2]
    targets_3 = targets[index_targets_2:]

    fold1_inds = []
    fold2_inds = []
    fold3_inds = []
    fold4_inds = []
    fold5_inds = []
    fold6_inds = []
    fold7_inds = []
    fold8_inds = []
    fold9_inds = []
    # Create folds for cross-validation by assinging a fold for each data point. 
    for i in range(n_sample):
        if drug_inds[i] in drugs_1:
            if target_inds[i] in targets_1:
                fold1_inds.append(i)
            elif target_inds[i] in targets_2:
                fold2_inds.append(i)
            elif target_inds[i] in targets_3:
                fold3_inds.append(i)
        elif drug_inds[i] in drugs_2:
            if target_inds[i] in targets_1:
                fold4_inds.append(i)
            elif target_inds[i] in targets_2:
                fold5_inds.append(i)
            elif target_inds[i] in targets_3:
                fold6_inds.append(i)
        elif drug_inds[i] in drugs_3:
            if target_inds[i] in targets_1:
                fold7_inds.append(i)
            elif target_inds[i] in targets_2:
                fold8_inds.append(i)
            elif target_inds[i] in targets_3:
                fold9_inds.append(i)

    drugs = [drugs_1, drugs_2, drugs_3]
    targets = [targets_1, targets_2, targets_3]
    folds = [fold1_inds, fold2_inds, fold3_inds, fold4_inds, fold5_inds, fold6_inds, fold7_inds, fold8_inds, fold9_inds]
    return folds#, drugs, targets


"""
Function to create data splits for a test fold of cross-validation.
"""
def cv_splits(drug_inds, target_inds, random_seed):
    folds = cv_folds(drug_inds, target_inds, random_seed)
    n_folds = len(folds)
    splits = []
    cv_dfs = []
    for i in range(n_folds):
        test_fold_inds = folds[i]
        other_folds = folds[:i] + folds[i+1:]

        D_test_fold = set(drug_inds[test_fold_inds])
        T_test_fold = set(target_inds[test_fold_inds])

        IDIT_inds = []
        IDOT_inds = []
        ODIT_inds = []
        ODOT_inds = []
        for j in range(n_folds-1):
            temp_inds = other_folds[j]
            D_temp = set(drug_inds[temp_inds])
            T_temp = set(target_inds[temp_inds])

            if D_test_fold.isdisjoint(D_temp):
                if T_test_fold.isdisjoint(T_temp):
                    ODOT_inds += temp_inds
                else:
                    IDIT_inds += temp_inds
                    ODIT_inds += temp_inds
            else:
                IDOT_inds += temp_inds
                if T_test_fold.isdisjoint(T_temp):
                    IDIT_inds += temp_inds
        
        np.random.seed(random_seed)
        np.random.shuffle(IDIT_inds)
        np.random.shuffle(IDOT_inds)
        np.random.shuffle(ODIT_inds)
        np.random.shuffle(ODOT_inds)
        IDIT_training, IDIT_validation = split_both_in(IDIT_inds, drug_inds, target_inds)
        IDOT_training, IDOT_validation = split_one_in_one_off(IDOT_inds, target_inds)
        ODIT_training, ODIT_validation = split_one_in_one_off(ODIT_inds, drug_inds)
        ODOT_training, ODOT_validation = split_both_off(ODOT_inds, drug_inds, target_inds)

        IDIT_split = [IDIT_training, test_fold_inds, IDIT_validation, str(i)]
        IDOT_split = [IDOT_training, test_fold_inds, IDOT_validation, str(i)]
        ODIT_split = [ODIT_training, test_fold_inds, ODIT_validation, str(i)]
        ODOT_split = [ODOT_training, test_fold_inds, ODOT_validation, str(i)]
        splits.append([IDIT_split, IDOT_split, ODIT_split, ODOT_split])

        df_i = pd.concat([pd.DataFrame({'random_seed':random_seed, 'roundID':i, 'subset':'test', 'setting':'all', 'index':test_fold_inds, 'ID_d':drug_inds[test_fold_inds], 'ID_t':target_inds[test_fold_inds]}),
            pd.DataFrame({'random_seed':random_seed, 'roundID':i, 'subset':'training', 'setting':'IDIT', 'index':IDIT_training, 'ID_d':drug_inds[IDIT_training], 'ID_t':target_inds[IDIT_training]}),
            pd.DataFrame({'random_seed':random_seed, 'roundID':i, 'subset':'validation', 'setting':'IDIT', 'index':IDIT_validation, 'ID_d':drug_inds[IDIT_validation], 'ID_t':target_inds[IDIT_validation]}),
            pd.DataFrame({'random_seed':random_seed, 'roundID':i, 'subset':'training', 'setting':'IDOT', 'index':IDOT_training, 'ID_d':drug_inds[IDOT_training], 'ID_t':target_inds[IDOT_training]}),
            pd.DataFrame({'random_seed':random_seed, 'roundID':i, 'subset':'validation', 'setting':'IDOT', 'index':IDOT_validation, 'ID_d':drug_inds[IDOT_validation], 'ID_t':target_inds[IDOT_validation]}),
            pd.DataFrame({'random_seed':random_seed, 'roundID':i, 'subset':'training', 'setting':'ODIT', 'index':ODIT_training, 'ID_d':drug_inds[ODIT_training], 'ID_t':target_inds[ODIT_training]}),
            pd.DataFrame({'random_seed':random_seed, 'roundID':i, 'subset':'validation', 'setting':'ODIT', 'index':ODIT_validation, 'ID_d':drug_inds[ODIT_validation], 'ID_t':target_inds[ODIT_validation]}),
            pd.DataFrame({'random_seed':random_seed, 'roundID':i, 'subset':'training', 'setting':'ODOT', 'index':ODOT_training, 'ID_d':drug_inds[ODOT_training], 'ID_t':target_inds[ODOT_training]}),
            pd.DataFrame({'random_seed':random_seed, 'roundID':i, 'subset':'validation', 'setting':'ODOT', 'index':ODOT_validation, 'ID_d':drug_inds[ODOT_validation], 'ID_t':target_inds[ODOT_validation]})],
            axis = 0, sort = False)
        cv_dfs.append(df_i)
    cv_dfs = pd.concat(cv_dfs, axis = 0)

    return cv_dfs, splits

"""
This part is for checking that the functions work as expected and saving the splits as csv-files.
One file per data set. 
"""
if __name__ == "__main__":
    # Select a seed or multiple seeds for controlling the randomness in creating the splits.
    random_seeds = [2688385916]
    # Choose the percentage of drugs and targets that are defined to be in area IDIT and the percentage of area IDIT pairs that will be used as training data.
    split_percentage = 1.0/3.0
    datasets = ["davis", "metz", "kiba", "merget", "GPCR", "IC", "E"]
    for random_seed in random_seeds:
        for ds in datasets:
            # Load the data set in the wanted form.
            XD, XT, Y, drug_inds, target_inds = eval('load_'+ds+'()')

            if ds == "kernelFilling":
                # Create a single split of one common test set and training+validation sets for settings IDIT-ODOT.
                df_indices, splits = splits(drug_inds, target_inds, split_percentage, random_seed)
            else:
                # Create splits for 9 fold cv.
                # For each test fold, there are training+validation sets for settings IDIT-ODOT.
                df_indices, splits = cv_splits(drug_inds, target_inds, random_seed)
            
            # Save the information of the splits as a csv-file.
            df_indices.to_csv('splits_'+ds+'_RS_'+str(random_seed)+'_smallerIDIT.csv', index = False)