from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from ltr_wrapper import ltr_cls

from numpy.random import SeedSequence
import data
import itertools as it
import pandas as pd
import multiprocessing as mp
from rlscore.measure import sqerror
import numpy as np
from sklearn.model_selection import ParameterGrid

"""
Function to concatenate two feature matrices. 
"""
def concatenate_features(X1, X2, inds1, inds2):
    features_X1 = X1[inds1,:]
    features_X2 = X2[inds2,:]
    X = np.hstack((features_X1, features_X2))
    return X

"""
Function to be executed during parallel computations. 
"""
def predictions(params):
    Y = params[0]
    drug_inds = params[1]
    target_inds = params[2]
    training_inds = params[3][0]
    test_inds = params[3][1]
    validation_inds = params[3][2]
    fold_id = params[3][3]
    model = params[4][0]
    parameters = params[4][1]
    hyperparams = params[4][2]
    XD = params[5]
    XT = params[6]

    train_drug_inds = drug_inds[training_inds]
    train_target_inds = target_inds[training_inds]
    Y_train = Y[training_inds]

    test_drug_inds = drug_inds[test_inds]
    test_target_inds = target_inds[test_inds]
    Y_test = Y[test_inds]

    validation_drug_inds = drug_inds[validation_inds]
    validation_target_inds = target_inds[validation_inds]
    Y_validation = Y[validation_inds]

    # The setting is not explicitly given so it needs to be determined.
    if set(test_drug_inds).isdisjoint(set(train_drug_inds)):
        if set(test_target_inds).isdisjoint(set(train_target_inds)):
            setting = "ODOT"
        else:
            setting = "ODIT"
    else:
        if set(test_target_inds).isdisjoint(set(train_target_inds)):
            setting = "IDOT"
        else:
            setting = "IDIT"

    """
    The part that is specifically for algorithms that are used in sklearn style starts here.
    """
    # Create such feature representation that it is suitable for algorithms in the library sklearn. 
    X_train = concatenate_features(XD, XT, train_drug_inds, train_target_inds)
    X_test = concatenate_features(XD, XT, test_drug_inds, test_target_inds)
    X_validation = concatenate_features(XD, XT, validation_drug_inds, validation_target_inds)
    
    # Initialize the variables for keeping track of the best model.
    MSE_perf_best = np.inf
    MSE_P_test = []
    MSE_hp_best = 0

    # Create a regressor object with parameter values that will not be optimized.
    regressor = model(**parameters)
    for hp_dict in hyperparams:
        regressor.set_params(**hp_dict)
        # Fit the model on training data. 
        model_hp = regressor.fit(X_train, Y_train)
        P_validation = model_hp.predict(X_validation).astype('float64')
        P_test = model_hp.predict(X_test)
        
        perf_validation = sqerror(Y_validation, P_validation)
        if perf_validation < MSE_perf_best:
            MSE_perf_best = perf_validation
            MSE_hp_best = hp_dict
            MSE_P_test = P_test

    # Save the test data predictions with the best hyperparameters for the performance measures. 
    df_predictions_list = []
    df_predictions_list.append(pd.DataFrame({'ID_d':test_drug_inds, 'ID_t':test_target_inds, 'Y':Y_test, \
    'P':MSE_P_test, 'setting':setting, 'fold':fold_id, 'model':str(model)+str(parameters), \
        'hyperparameter':str(MSE_hp_best)}))
    
    """
    The part that is specifically for algorithms that are used in sklearn style ends here.
    """

    predictions_test = pd.concat(df_predictions_list, ignore_index = True)
    return(predictions_test)

if __name__ == "__main__":
    base_seed = 12345
    repetitions = 10**0
    ss = SeedSequence(base_seed)
    random_seeds = ss.generate_state(repetitions)
    datasets = ["davis", "metz", "kiba", "merget", "GPCR", "IC", "E"]
    split_percentage = 1.0/3

    for ds in datasets:
        df_list = []
        print(ds)
        XD, XT, Y, drug_inds, target_inds = eval('data.load_'+ds+'()')    
        n_D = XD.shape[0]
        n_T = XT.shape[0]

        for random_seed in random_seeds:
            df, splits = data.cv_splits(drug_inds, target_inds, random_seed)
            splits_foldwise = list(it.chain.from_iterable(splits))
            n_splits = len(splits_foldwise)
            # Previously the order of splits has been fold 0: IDIT, IDOT, ODIT, ODOT, fold 1: IDIT, IDOT, ODIT, ODOT etc.
            # Change it to IDIT: fold 0,..., fold 8, IDOT: fold 0,..., fold 8 etc.
            new_order = list(range(0,n_splits, 4))+list(range(1,n_splits, 4))+list(range(2,n_splits, 4))+list(range(3,n_splits, 4))
            splits_settingwise = [splits_foldwise[i] for i in new_order]
            
            """
            Determine the algorithms and their parameters here.
            Choose the values to be tested for the hyperparameter whose value will be optimized.
            Elements in the list of models are in form [learner object, dictionary of parameters whose values are given but not optimized, 
                                                        list of dictionaries containing all possible values in the hyperparameter space.]
            """
            models = [[XGBRegressor, \
                       {'objective':'reg:squarederror', 'random_state':random_seed}, \
                       ParameterGrid([{'n_estimators':[100, 125, 150]}])], \
                      [KNeighborsRegressor,
                       {}, 
                       ParameterGrid([{'n_neighbors':[5,10,30,50,75,100]}])], \
                      [RandomForestRegressor,
                       {'random_state':random_seed, 'warm_start':True}, 
                       ParameterGrid([{'n_estimators':[100,200,300]}])], \
                      [ltr_cls,
                       {'order':2, 'rank':30}, 
                       ParameterGrid([{'rank': [ 10,20,30,40,50,60,70,80]}])]]

            """
            The part that needs to be changed ends here.
            """
            parameters = it.product([Y], [drug_inds], [target_inds], splits_settingwise, \
                                    models, [XD], [XT])
        
            # Compute different cases (models & settings & folds) at the same time.
            pool = mp.Pool(processes = 4)
            output = pool.map(predictions, list(parameters))
            pool.close()
            pool.join()
            df = pd.concat(output, ignore_index=False, axis = 0)
            df['data_set'] = ds
            df['random_seed'] = random_seed
            df_list.append(df)
            
        # Save the predictions for the data set as csv-file. 
        pd.concat(df_list, ignore_index = True).to_csv('predictions_sklearnStyle_'+ds+'.csv', index = False)