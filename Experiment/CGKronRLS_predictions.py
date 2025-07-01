import pandas as pd
import itertools as it
import multiprocessing as mp

import numpy as np
from rlscore.predictor import KernelPairwisePredictor , LinearPairwisePredictor
from rlscore.utilities.pairwise_kernel_operator import PairwiseKernelOperator
from numpy.random import SeedSequence

from rlscore.kernel import GaussianKernel, LinearKernel
from rlscore.learner import CGKronRLS
from rlscore.measure import sqerror
import data

def pko_kronecker(K1, K2, rows1, cols1, rows2, cols2):
    pko = PairwiseKernelOperator([K1], [K2], [rows1], [cols1], [rows2], [cols2], weights=[1.0])
    return pko

def pko_linear(K1, K2, rows1, cols1, rows2, cols2):
    n, d = K1.shape#[0]
    m, k = K2.shape#[0]
    O1 = np.ones((n, d))
    O2 = np.ones((m, k))
    pko = PairwiseKernelOperator([K1, O1], [O2, K2], [rows1, rows1], [cols1, cols1], [rows2, rows2], [cols2, cols2], weights=[1.0, 1.0])
    return pko

# Transform K into a dense matrix, such that column and row indices are surjective
def K_to_dense(K, row_inds, col_inds):
    rows, rows_inverse = np.unique(row_inds, return_inverse=True)
    cols, cols_inverse = np.unique(col_inds, return_inverse=True)
    K = np.array(K[np.ix_(rows, cols)])
    row_indices = np.arange(len(rows))
    col_indices = np.arange(len(cols))
    row_inds = np.array(row_indices[rows_inverse])
    col_inds = np.array(col_indices[cols_inverse])
    return K, row_inds, col_inds

# Class of callback object for cg_kron_rls function.
class CallBack(object):

    def __init__(self, Y, pko, ID_d, ID_t, ESlag = 10): 
        self.Y = Y
        self.pko = pko
        self.iter = 1
        self.earlyStopLag = ESlag
        self.drug_inds = ID_d
        self.target_inds = ID_t

        self.MSE = True
        self.MSE_best_perf = np.inf
        self.MSE_best_learner = None
        self.MSE_best_iter = 0
        self.best_models = [self.MSE_best_learner, self.MSE_best_perf, self.MSE_best_iter]

    def callback(self, learner):
        P = self.pko.matvec(learner.A)
        predictor = learner.A
        
        if self.MSE:
            perf = sqerror(self.Y, P)
            # Save the model with the smallest mean squared error.
            if perf < self.MSE_best_perf:
                self.MSE_best_perf = perf
                self.MSE_best_iter = self.iter
                self.MSE_best_learner = predictor
                # Update also the list of the best models
                self.best_models = [self.MSE_best_learner, self.MSE_best_perf, self.MSE_best_iter]
            if self.iter > self.MSE_best_iter + self.earlyStopLag:
                # The best model with MSE is already found, no need to continue calculating these.
                self.MSE = False
        
        # Start the next iteration once the performance on the current iteration is measured with all performance measures.
        self.iter += 1
        # Raise an error if the early stop criteria is met.
        if (not self.MSE): 
            raise ValueError(self.best_models)
            
    def finished(self, learner):
        pass

def predictions(params):
    Y = params[0]
    drug_inds = params[1]
    target_inds = params[2]
    training_inds = params[3][0]
    test_inds = params[3][1]
    validation_inds = params[3][2]
    fold_id = params[3][3]
    model = params[4][0][0] # CGKronRLS object.
    parameters = params[4][0][1] # Parameter values that are pre-determined and not optimized. 
    hyperparams = params[4][1] # List of possible values for the hyperparameter to be optimized.
    XD = params[5]
    XT = params[6]
    
    # Split the indices of the drugs, targets and Y according to the training-test-validation split.
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
    The part that is specifically for CGKronRLS starts here. 
    """
    kernels = parameters[-1] # List of three elements: type of KD, type of KT and type of pko.
    # Basis kernels: Linear or Gaussian.
    drug_kernel_type = kernels[0]
    if drug_kernel_type == "linear":
        drug_kernel = LinearKernel(XD)
    elif drug_kernel_type == "gaussian":
        drug_kernel = GaussianKernel(XD, gamma=parameters[0]['gamma'])
    KD = drug_kernel.getKM(XD)

    target_kernel_type = kernels[1]
    if target_kernel_type == "linear":
        target_kernel = LinearKernel(XT)
    elif target_kernel_type == "gaussian":
        target_kernel = GaussianKernel(XT, gamma=parameters[0]['gamma'])
    KT = target_kernel.getKM(XT)

    # Create training kernels to be given to CGKronRLS.
    KD_train, rows_train, rows_train = K_to_dense(KD, train_drug_inds, train_drug_inds)
    KT_train, cols_train, cols_train = K_to_dense(KT, train_target_inds, train_target_inds)
    # Create the kernels for the validation set. 
    KD_validation, rows_validation1, rows_validation2 = K_to_dense(KD, validation_drug_inds, train_drug_inds)
    KT_validation, cols_validation1, cols_validation2 = K_to_dense(KT, validation_target_inds, train_target_inds)
    
    # Test kernels so that only training set is used for predicting the test set labels.
    KD_test, rows_test1, rows_test2 = K_to_dense(KD, test_drug_inds, train_drug_inds)
    KT_test, cols_test1, cols_test2 = K_to_dense(KT, test_target_inds, train_target_inds)
    
    # Create pkos.
    pko_function = kernels[2]
    pko_train = eval(pko_function+'(KD_train, KT_train, rows_train, cols_train, rows_train, cols_train)')
    pko_validation = eval(pko_function+'(KD_validation, KT_validation, rows_validation1, cols_validation1, \
        rows_validation2, cols_validation2)')
    pko_test = eval(pko_function+'(KD_test, KT_test, rows_test1, cols_test1, rows_test2, cols_test2)')

    kronRLS_performances = []
    kronRLS_MIs = []
    kronRLS_predictors = []

    # Hyperparameter optimization.
    for hp in hyperparams:
        cb_validation = CallBack(Y = Y_validation, pko = pko_validation,
            ID_d = validation_drug_inds, ID_t = validation_target_inds, ESlag = parameters[0]['ES_lag'])
        try:
            CGKronRLS(Y = Y_train, pko = pko_train, maxiter = parameters[0]['MI'], regparam = hp, callback = cb_validation)
        except ValueError as err:
            kronRLS_predictors.append([err.args[0][0]])
            kronRLS_performances.append([err.args[0][1]])
            kronRLS_MIs.append([err.args[0][2]])
        else:
            kronRLS_predictors.append([cb_validation.best_models[0]])
            kronRLS_performances.append([cb_validation.best_models[1]])
            kronRLS_MIs.append([cb_validation.best_models[1]])
    
    # Select the best hyperparameters for the performance measures. 
    df_predictions_list = []
    index_regparam_best = np.argmin([performances[0] for performances in kronRLS_performances])
    hp_best = hyperparams[index_regparam_best]
    P_test = pko_test.matvec(kronRLS_predictors[index_regparam_best][0])
    df_predictions_list.append(pd.DataFrame({'ID_d':test_drug_inds, 'ID_t':test_target_inds, 'Y':Y_test, \
    'P':P_test, 'setting':setting, 'fold':fold_id, 'model':str(model)+str(parameters), \
        'hyperparameter':str({'regparam':hp_best})})) 
        
    """
    The part that is specifically for CGKronRLS ends here. 
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

    """
    Determine the algorithms and their parameters here.
    Choose the values to be tested for the hyperparameter whose value will be optimized.
    """
    kernels = [["linear", "linear", "pko_linear"], ["linear", "linear", "pko_kronecker"], \
        ["gaussian", "gaussian", "pko_linear"], ["gaussian", "gaussian", "pko_kronecker"]]
    ES_lag = 50
    MI = 1000
    models = list(it.product([CGKronRLS], it.product([{'ES_lag':ES_lag, 'MI':MI, 'gamma':1e-5}], kernels)))
    # May be needed to think about these values more carefully. 
    hyperparams = [[2.0**(-10), 2.0**(-5), 2.0**(-4), 2.0**(-3), 2.0**(-2), 2.0**(-1), 2.0**(0), 2.0**(1), \
                    2.0**(2), 2.0**(3), 2.0**(4), 2.0**(5), 2.0**(10)]]
    """
    The part that needs to be changed ends here.
    """
    
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
            # Previously the order of splits is fold 0: IDIT, IDOT, ODIT, ODOT, fold 1: IDIT, IDOT, ODIT, ODOT etc.
            # Change it to IDIT: fold 0,..., fold 8, IDOT: fold 0,..., fold 8 etc.
            new_order = list(range(0,n_splits, 4))+list(range(1,n_splits, 4))+list(range(2,n_splits, 4))+list(range(3,n_splits, 4))
            splits_settingwise = [splits_foldwise[i] for i in new_order]
            
            parameters = it.product([Y], [drug_inds], [target_inds], splits_settingwise, \
                list(it.product(models, hyperparams)), [XD], [XT]) # , [perf_measures] poistettu lopusta.
            
            # Compute different settings at the same time.
            pool = mp.Pool(processes = 4)
            output = pool.map(predictions, list(parameters))
            pool.close()
            pool.join()
            df = pd.concat(output, ignore_index=False, axis = 0)
            df['data_set'] = ds
            df['random_seed'] = random_seed
            df_list.append(df)
            
        # Save the predictions for the data set as csv-file. 
        pd.concat(df_list, ignore_index = True).to_csv('predictions_KRLS_'+ds+'.csv', index = False)