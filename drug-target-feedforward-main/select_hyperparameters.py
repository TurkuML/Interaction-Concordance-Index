import numpy as np
import pandas as pd
import time
import sys

dataset = sys.argv[1]
date_created = sys.argv[2]
setting = sys.argv[3]
fold = sys.argv[4]

def get_best_testset(s, f, df):
    start_time = time.time()

    epochs = df.epoch.unique()
    learning_rates = df.lr.unique()
    batch_sizes = df.batch_size.unique()
    num_layers = df.num_layers.unique()
    neurons_in_layers = df.neurons_in_layers.unique()
    dropouts = df.dropout_ratio.unique()

    best_mse = 1e15
    i = 0
    for epoch in epochs:
        print(epoch)
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for n_layers in num_layers:
                    for n_in_layers in neurons_in_layers:
                        for dropout_ratio in dropouts:
                            tmp_df = df[
                                        (df['testset'] == 'valid') &
                                        (df['epoch'] == epoch) &
                                        (df['batch_size'] == batch_size) &
                                        (df['lr'] == learning_rate) &
                                        (df['num_layers'] == n_layers) &
                                        (df['neurons_in_layers'] == n_in_layers) &
                                        (df['dropout_ratio'] == dropout_ratio)
                                        ]
                            # TODO: add option to use other metric
                            mse = np.mean((tmp_df['Y'] - tmp_df['P'])**2)
                            if mse < best_mse:
                                best_mse = mse
                                best_epoch = epoch
                                best_batch_size = batch_size
                                best_lr = learning_rate
                                best_n_layers = n_layers
                                best_n_in_layers = n_in_layers
                                best_dropout = dropout_ratio
                                print("best MSE updated to:", mse)
                                print("best hyperparameters updated to:",
                                      "\nEpoch:\t\t\t", best_epoch,
                                      "\nBatch size:\t\t", best_batch_size,
                                      "\nLearning rate:\t\t", best_lr,
                                      "\nBest number of layers:\t", best_n_layers,
                                      "\nBest neurons in layer:\t", best_n_in_layers,
                                      "\nBest dropout ratio:\t", best_dropout,
                                        sep = "")

    best_testdf = df[ (df['testset'] == 'test') &
                     (df['epoch'] == best_epoch) &
                     (df['batch_size'] == best_batch_size) &
                     (df['lr'] == best_lr) &
                     (df['num_layers'] == best_n_layers) &
                     (df['neurons_in_layers'] == best_n_in_layers) &
                     (df['dropout_ratio'] == best_dropout)
                     ]

    fname = "predictions/" + dataset + "/" + date_created + "/MSE_S" + s + "F" + f + ".csv"
    best_testdf.to_csv(fname, index=False)
    print("Runtime [S"+s+"F"+f+"]: " + str(time.time()-start_time) + "s")


print("S"+setting+"F"+fold)
get_best_testset(setting, fold, pd.read_csv('predictions/' + dataset + '/' + date_created + '/S' + setting + 'F' + fold + '.csv'))
