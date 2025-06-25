import numpy as np
from numpy.random import SeedSequence
import pandas as pd
from sklearn_predictions import concatenate_features
import time
import data
from ltr_wrapper import ltr_cls
from IC_index import InteractionConcordanceIndex
from rlscore.measure import cindex
from cindex_measure import cindex_modified
from performance import group_performance_normalized
import multiprocessing as mp

def generate_data(randomSeed, n_dim1 = 100, n_dim2 = 100, known_fraction = 0.25,\
                  p_flipped = 1/12, row_imbalance = 0.2, col_imbalance = 0.2):
    np.random.seed(randomSeed)

    X1 = np.identity(n_dim1, dtype='float64')
    X2 = np.identity(n_dim2, dtype='float64')

    known_pairs = np.random.choice(a = [0,1], size = n_dim1*n_dim2, replace = True, p = [1-known_fraction, known_fraction])
    pair_indicator_matrix = np.reshape(known_pairs, (n_dim1, n_dim2))
    row_inds, col_inds = np.where(pair_indicator_matrix == 1)
    
    Y = (-1)**(row_inds < n_dim1*row_imbalance)*(-1)**(col_inds < n_dim2*col_imbalance)

    # "Coin flip" randomness
    flipped = np.random.choice(a = [-1,1], size = len(Y), replace = True, \
                                p = [p_flipped, 1-p_flipped])
    Y = Y*flipped
 
    return X1, X2, Y.astype('float64'), row_inds.astype(int), col_inds.astype(int)

def group_sum(y, train_group_ids, test_group_ids):
    g_sum = np.zeros(len(test_group_ids))
    for i in set(test_group_ids):
        y_subset = y[train_group_ids == i]
        if len(y_subset) > 0:
            g_sum[test_group_ids == i] = np.sum(y_subset)            
    return(g_sum.astype('float64'))

def simulation(rn_generator, n_drugs = 100, n_targets = 100, k_f = 0.25, \
               row_imbalance = 0.10, col_imbalance = 0.20, p_flip = 1/20, split_percentage = 0.5):
    time_start_data = time.time()
    random_seed = rn_generator.generate_state(1)[0]
    # Generate imbalanced XOR data
    X1, X2, Y, row_inds, col_inds = generate_data(random_seed, n_dim1 = n_drugs, n_dim2 = n_targets, \
                                                    known_fraction = k_f, p_flipped=p_flip, \
                                                    row_imbalance=row_imbalance, col_imbalance=col_imbalance)
    # Split the data into training, testing and validation sets for all four settings.
    df, splits = data.train_test_splits(row_inds, col_inds, split_percentage, random_seed)
    
    test_inds = splits[0][1]
    Y_test = Y[test_inds]
    row_test = row_inds[test_inds]
    col_test = col_inds[test_inds]
    X_test = concatenate_features(X1, X2, row_test, col_test)
    
    df_list = []
    for split_ind in range(4):
        # The splits for different settings are in the following order based on the
        # train_test_splits function. 
        setting = ["IDIT", "IDOT", "ODIT", "ODOT"][split_ind]
        training_inds = splits[split_ind][0]
        Y_training = Y[training_inds]
        row_training = row_inds[training_inds]
        col_training = col_inds[training_inds]

        # Global sum of labels.
        P_test = np.sum(Y_training == 1)
        df_list.append(pd.DataFrame({'random_seed':random_seed, 'ID_d':row_test,\
                                     'ID_t':col_test, 'setting': setting, 'Y':Y_test, \
                                        'P':P_test, 'model':"GS"}))

        # Rowwise sum of labels.
        P_test_R = group_sum(Y_training, row_training, row_test)
        if any(P_test_R != 0):
            df_list.append(pd.DataFrame({'random_seed':random_seed, 'ID_d':row_test,\
                                            'ID_t':col_test, 'setting': setting, 'Y':Y_test, \
                                            'P':P_test_R, 'model':"DS"}))
        
        # Columnwise sum of labels.
        P_test_C = group_sum(Y_training, col_training, col_test)
        if any(P_test_C != 0):
            df_list.append(pd.DataFrame({'random_seed':random_seed, 'ID_d':row_test,\
                                            'ID_t':col_test, 'setting': setting, 'Y':Y_test, \
                                            'P':P_test_C, 'model':"TS"}))
        
        # Sum of the rowwise and columnwise sums of labels.
        P_test_B = P_test_R + P_test_C
        if any(P_test_B != 0):
            df_list.append(pd.DataFrame({'random_seed':random_seed, 'ID_d':row_test,\
                                            'ID_t':col_test, 'setting': setting, 'Y':Y_test, \
                                            'P':P_test_B, 'model':"SS"}))

        # Product of the rowwise and columnwise sums of labels.
        P_test_P = P_test_R*P_test_C
        if any(P_test_P != 0):
            df_list.append(pd.DataFrame({'random_seed':random_seed, 'ID_d':row_test,\
                                            'ID_t':col_test, 'setting': setting, 'Y':Y_test, \
                                            'P':P_test_P, 'model':"PS"}))

        # LTR predictions.
        X_train = concatenate_features(X1, X2, row_training, col_training)
        model = ltr_cls(order = 2, rank = 1).fit(X_train, Y_training)
        P_test_PR = model.predict(X_test)
        df_list.append(pd.DataFrame({'random_seed':random_seed, 'ID_d':row_test,\
                                     'ID_t':col_test, 'setting': setting, 'Y':Y_test, \
                                        'P':P_test_PR, 'model':"PR"}))


    # All models in one data frame and reshape the data frame.
    df_all = pd.concat(df_list)
    df_all = df_all.pivot_table(index = ['ID_d','ID_t', 'Y'], values = 'P', \
                            columns = ['setting', 'model']).reset_index()
    df_all.columns.values[0] = 'ID_d'
    df_all.columns.values[1] = 'ID_t'
    df_all.columns.values[2] = 'Y'
    df_all.columns = df_all.columns.to_flat_index()
    
    C_indices = []
    C_d_indices = []
    C_t_indices = []
    accuracies = []
    
    for m in range(3, len(df_all.columns)):
        
        # Calculate global C-index.
        C_indices.append(cindex(Y_test, df_all.iloc[:,m].values))
        # Calculate averaged drugwise C-index.
        C_d_indices.append(group_performance_normalized(cindex_modified, Y_test, \
                                                                df_all.iloc[:,m].values, row_test))
        # Calculate averaged targetwise C-index.
        C_t_indices.append(group_performance_normalized(cindex_modified, Y_test, \
                                                                df_all.iloc[:,m].values, col_test))
        # Calculate the accuracy.
        accuracies.append(np.mean(np.heaviside(Y_test*df_all.iloc[:,m].values, 1/2)))
    
    # Calculate IC-indices for all models at once. 
    IC_indices = InteractionConcordanceIndex(row_test, col_test, \
                                            Y_test, df_all.iloc[:,3:].to_numpy())
    
    performance = pd.DataFrame({'random_seed':random_seed, 'model':df_all.columns[3:], 'IC_index': IC_indices, \
                                'accuracy':accuracies, 'C_index':C_indices, \
                                'C_d_index':C_d_indices, 'C_t_index':C_t_indices})
    print("Predictions and performances calculated with every method", \
            "in time", time.time()-time_start_data, "with random seed", random_seed)
    return(performance)


if __name__ == "__main__":
    # Generate the random seeds to be used for generating different data sets for the simulation study.
    base_seed = 12345
    repetitions = 10**5
    ss = SeedSequence(base_seed)
    generators = ss.spawn(repetitions)

    # Define the setup for the simulation. 
    n_drugs = 200
    n_targets = 200
    k_f = 0.25
    row_imbalance = 0.10
    col_imbalance = 0.20
    p_flip = 1/20
    ds = "XOR_imbalance_"+str(row_imbalance)+"_"+str(col_imbalance)
    performances = []
    # Use 50 % of the data as test set.
    split_percetage = 0.5
    
    # Compute different cases (models & settings & folds) at the same time.
    pool = mp.Pool(processes = 8)
    output = pool.map(simulation, list(generators))
    pool.close()
    pool.join()
    df = pd.concat(output, ignore_index=False, axis = 0)
    performances.append(df)
        
    # Save the test performances for all data sets, random seeds and validation performance measures in one file.
    pd.concat(performances, ignore_index = True).to_csv('performances_'+ds+'.csv', index = False)