import numpy as np
import pandas as pd
import time
from IC_index import InteractionConcordanceIndex
from rlscore.measure import cindex
from cindex_measure import cindex_modified
from statistics import mode

"""
Function to calculate drug or targetwise performance measures. 
All elements in a group are taken as a subset for which the performance is calculated. 
Returns average performance over the groups. Normalized version takes into account
the numbers of elements in the groups. 

Input: performance measure function, arrays of labels, predictions and IDs that determine the group.

Output: prediction performance within the group measured by the given performance measure.
"""
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
Function to calculate interaction concordance index, concordance index, 
and drug and targetwise concordance indices.

Input: A data frame where the first 4 columns are ID_d, ID_t, fold and Y.
Total number of columns depends of the number of different hypotheses whose predictions are
gathered in the data frame as columns. These contain also the predictions for the different settings.

Output: The function returns the lists of IC-indices and all variations of C-indices.
"""
def calculate_foldwise_IC_C_indices(df):
    folds = set(df['fold'])

    # Initialize lists for collecting the foldwise performances.
    C_index_list = []
    C_d_index_list = []
    C_t_index_list = []
    IC_index_list = []

    # Go through the folds.
    for fold_id in folds:
        # Take a subset of data containing only rows related to the current fold.
        df_fold = df.loc[df['fold'] == fold_id,:]
        drug_inds_fold = df_fold.ID_d.values.astype('int32')
        target_inds_fold = df_fold.ID_t.values.astype('int32')
        Y_fold = df_fold.Y.values
        P_fold = df_fold.iloc[:,4:]

        # Initialize lists for collecting the C-index based performance measures for the different hypotheses.
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

        # Calculate IC-indices for all hypotheses at once. 
        IC_indices_fold = InteractionConcordanceIndex(drug_inds_fold, target_inds_fold, \
                                                Y_fold.astype(float), P_fold.to_numpy())
        
        # Add the performance measure values related to this fold to the lists where all foldwise values are collected.
        C_index_list.append(C_indices_fold)
        C_d_index_list.append(C_d_indices_fold)
        C_t_index_list.append(C_t_indices_fold)
        IC_index_list.append(IC_indices_fold)
    
    # Calculate the averages of the foldwise C- and IC-indices.
    C_indices = pd.DataFrame(np.vstack(C_index_list)).mean()
    C_d_indices = pd.DataFrame(np.vstack(C_d_index_list)).mean()
    C_t_indices = pd.DataFrame(np.vstack(C_t_index_list)).mean()
    IC_indices = pd.DataFrame(np.vstack(IC_index_list)).mean()

    
    return IC_indices, C_indices, C_d_indices, C_t_indices


if __name__ == "__main__":
    # List the data sets for which the performances are calculated.
    data_sets = ["davis", "metz", "kiba", "merget", "GPCR", "IC", "E"]

    df_ds = []
    for ds in data_sets:

        print("Calculation of performance measures started for data", ds)
        df_list = []
        # List the algorithms for which the performances are calculated.
        for m in ["KRLS", "kNN", "ltr", "RF", "XGBoost", "DDTA", "FF", "GT"]: 
            try:
                # Modify the path if the predictions are not in the same folder as this file.
                df_list.append(pd.read_csv('predictions_'+m+'_'+ds+'.csv'))
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
                                columns = ['setting', 'model']).reset_index()
        
        print(df_all)
        
        time_start = time.time()
        
        IC_indices, C_indices, C_d_indices, C_t_indices = calculate_foldwise_IC_C_indices(df_all)
        
        df_ds.append(pd.DataFrame({'data':ds, 'model':df_all.columns[4:].to_flat_index(), \
                                   'IC_index': IC_indices, 'C_index':C_indices, \
                                   'C_d_index':C_d_indices, 'C_t_index':C_t_indices})) 
  
        print("Calculations finished in time", time.time()-time_start)
    
    # Save all the results in a .csv file. 
    pd.concat(df_ds, ignore_index = True).to_csv('performances.csv', index = False)
