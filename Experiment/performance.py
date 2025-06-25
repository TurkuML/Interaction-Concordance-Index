import numpy as np
import pandas as pd
import itertools as it
import multiprocessing as mp
import time
from A_index import cython_assignmentIndex
from rlscore.measure import cindex
from cindex_measure import cindex_modified
from statistics import mode

"""
Functions to calculate drug or targetwise performance measures. 
All elements in a group are taken as a subset for which the performance is calculated. 
Returns average performance over the groups. Normalized version takes into account
the numbers of elements in the groups. 
"""
def group_performance(measure, y, y_predicted, group_ids):
    performances = []
    for i in set(group_ids):
        y_subset = y[group_ids == i]
        y_predicted_subset = y_predicted[group_ids == i]
        if len(set(y_subset)) > 1:
            performances.append(measure(y_subset, y_predicted_subset))
    performances_average = np.mean(performances)
    return(performances_average)

def group_performance_normalized(measure, y, y_predicted, group_ids):
    performances = []
    for i in set(group_ids):
        y_subset = y[group_ids == i]
        y_predicted_subset = y_predicted[group_ids == i]
        if len(set(y_subset)) > 1:
            performances.append(measure(y_subset, y_predicted_subset))
    performances = np.array(performances)   
    
    performances_average = np.sum(performances[:, 0])/np.sum(performances[:, 1])
    return(1- performances_average)

"""
Function to calculate assignment index, concordance index, 
and drug and targetwise concordance indices.

Input: A data frame where the first 6 columns are ID_d, ID_t, fold, Y, P_BCF and P_BLF.
Total number of columns depends of the number of different models whose predictions are
gathered in the data frame as columns. There are also the predictions for different settings.

Output: The function returns the lists of A-indices, C-indices, R2s and modified R2s for
all the models including also the oracle model.
"""
def calculate_foldwise_A_C_indices(df):
    folds = set(df['fold'])
    """
    Foldwise measures: C- and A-indices.
    """
    C_index_list = []
    C_d_index_list = []
    C_t_index_list = []
    A_index_list = []
    for fold_id in folds:
        df_fold = df.loc[df['fold'] == fold_id,:]
        drug_inds_fold = df_fold.ID_d.values.astype('int32')
        target_inds_fold = df_fold.ID_t.values.astype('int32')
        Y_fold = df_fold.Y.values
        P_fold = df_fold.iloc[:,4:]
        C_indices_fold = []
        C_d_indices_fold = []
        C_t_indices_fold = []

        for m in range(P_fold.shape[1]):
            print(P_fold.columns[m])
            # Calculate global C-index.
            C_indices_fold.append(cindex(Y_fold, P_fold.iloc[:,m].values))
            # Calculate averaged drugwise C-index.
            C_d_indices_fold.append(group_performance_normalized(cindex_modified, Y_fold, \
                                                                    P_fold.iloc[:,m].values, drug_inds_fold))
            # Calculate averaged targetwise C-index.
            C_t_indices_fold.append(group_performance_normalized(cindex_modified, Y_fold, \
                                                                    P_fold.iloc[:,m].values, target_inds_fold))

        A_indices_fold = cython_assignmentIndex(drug_inds_fold, target_inds_fold, \
                                                Y_fold.astype(float), P_fold.to_numpy())
        
        C_index_list.append(C_indices_fold)
        C_d_index_list.append(C_d_indices_fold)
        C_t_index_list.append(C_t_indices_fold)
        A_index_list.append(A_indices_fold)
    
    # Averages of the foldwise C- and A-indices.
    C_indices = pd.DataFrame(np.vstack(C_index_list)).mean()
    C_d_indices = pd.DataFrame(np.vstack(C_d_index_list)).mean()
    C_t_indices = pd.DataFrame(np.vstack(C_t_index_list)).mean()
    A_indices = pd.DataFrame(np.vstack(A_index_list)).mean()

    
    return A_indices, C_indices, C_d_indices, C_t_indices


if __name__ == "__main__":
    # List the data sets for which the performances are calculated.
    data_sets = ["davis", "metz", "kiba", "merget", "GPCR", "IC", "E"]

    df_ds = []
    for ds in data_sets:

        print("Calculation of performance measures started for data", ds)
        df_list = []
        # List the algorithms for which the performances are calculated.
        # Now the algorithms that are used in sklearn style are listed separately,
        # because they were run like that, but if they were run as they are now in the 
        # file sklearn_predictions.py, replace those names by "sklearnStyle".
        for m in ["KRLS", "kNN", "ltr", "RF", "XGBoost", "DDTA", "FF", "GT"]: 
            try:
                df_list.append(pd.read_csv('Puhti predictions/predictions_'+m+'_'+ds+'.csv'))
            except:
                continue 

        df = pd.concat(df_list)
        
        # There are different precisions for Y in FF and GT results. 
        # Hence, use the most common precision for all methods.
        df_grouped = df.groupby(['ID_d', 'ID_t']).agg({'Y': lambda x: mode(x)}).reset_index()
        # Join the groupwise modes to the original data frame based on the IDs. 
        df = df.merge(df_grouped, on=['ID_d', 'ID_t'], suffixes=('', '_mode'))
        # Replace the column Y with the values where the groupwise mode is used for all methods.
        df['Y'] = df['Y_mode']
        df.drop(columns=['Y_mode'], inplace=True)
        
        # Spread the predictions of different models from one column to several columns. 
        df_all = df.pivot_table(index = ['ID_d','ID_t', 'fold', 'Y'], values = 'P', \
                                columns = ['setting', 'model', 'perf_measure']).reset_index()
        
        print(df_all)
        
        time_start = time.time()
        
        A_indices, C_indices, C_d_indices, C_t_indices = calculate_foldwise_A_C_indices(df_all)
        
        df_ds.append(pd.DataFrame({'data':ds, 'model':df_all.columns[4:].to_flat_index(), \
                                   'A_index': A_indices, 'C_index':C_indices, \
                                   'C_d_index':C_d_indices, 'C_t_index':C_t_indices})) 
  
        print("Calculations finished in time", time.time()-time_start)
    
    # Save all the results in a .csv file. 
    pd.concat(df_ds, ignore_index = True).to_csv('performances_20122024.csv', index = False)
