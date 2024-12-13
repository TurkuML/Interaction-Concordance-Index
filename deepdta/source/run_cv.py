import numpy as np
import pandas as pd
import tensorflow as tf
import os

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from datahelper import *
from arguments import argparser, logging
import time
from datetime import datetime

# slow O(n^2) python implementation of C-index
#from emetrics import get_cindex as cindex
# O(n log(n)) cython implementation of C-index
from rlscore.measure.cindex_measure import cindex
from emetrics import mse
from A_index import cython_assignmentIndex


# tensorflow makes no guarantees about random seeds across different versions:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
# note also that there is evidence, that using a GPU (cuDNN) might make the results irreproducible:
# https://github.com/keras-team/keras/issues/2479#issuecomment-213987747
# as such, setting random seeds does not guarantee reproducibility.
# see https://github.com/NVIDIA/framework-reproducibility/blob/master/doc/d9m/tensorflow.md
#+if reproducible results are necessary
tf.random.set_seed(42)
np.random.seed(1)

figdir = "figures/"

def build_model(FLAGS, NUM_FILTERS, FILTER_LENGTH_SMILES, FILTER_LENGTH_PROTEIN, LEARNING_RATE):
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput) 
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH_SMILES,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH_SMILES,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH_SMILES,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH_PROTEIN,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH_PROTEIN,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH_PROTEIN,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1)

    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    # Regression task: output the actual value with no activation function
    predictions = Dense(1, kernel_initializer='normal')(FC2)

    # glue it all together
    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    optimizer = Adam(learning_rate=LEARNING_RATE, amsgrad=False)

    interactionModel.compile(optimizer=optimizer, loss='mean_squared_error')

    return interactionModel


    

def crossvalidation_experiment(FLAGS, perfmeasure, deepmethod):
    XD, XT, Y = load_rawtext_data(FLAGS)
    train_ids, val_ids, test_ids = load_cv_indices(FLAGS)

    print(len(train_ids), "training samples")
    print(len(val_ids), "validation samples")


    print(f"Choosing hyperparameters for fold {FLAGS.cv_fold} in setting {FLAGS.fourfield_setting}")
    best_lr, best_num_win, best_smi_len, best_seq_len, num_epochs, best_mse = select_hyperparameters(FLAGS, perfmeasure, deepmethod, XD, XT, Y, train_ids, val_ids)
    print("Chosen hyperparameters")
    print("----------------------")
    print("Number of convolution windows:",best_num_win)
    print("SMILES window length:",best_smi_len)
    print("Protein sequence window length:",best_seq_len)
    print("Stopped early on epoch:",num_epochs)
    print("Learning rate:",best_lr)


    logging("Hyperparameters num_win: %d, smi_len: %d, seq_len: %d, lr: %s gave the best mse(%f)" %
            (best_num_win, best_smi_len, best_seq_len, best_lr, best_mse), FLAGS)
    if num_epochs < FLAGS.num_epoch:
        logging("Training for the chosen hyperparameters was stopped early after %d epochs." %
                (num_epochs,), FLAGS)
    else:
        logging("The model was trained with the above hyperparameters for the full %d epochs." %
                (FLAGS.num_epoch,), FLAGS)

    print("Training the model on train+val sets and testing on test set.")
    print(len(train_ids)+len(val_ids), "train samples")
    print(len(test_ids), "test samples")
    Y, P = train_test(FLAGS, perfmeasure, deepmethod, XD, XT, Y, np.concatenate((train_ids, val_ids)), test_ids,
           best_num_win, best_smi_len, best_seq_len, best_lr, num_epochs)

    test_drug_ids, test_target_ids, test_Y_ids = separate_indices(test_ids)
    # save predictions
    df = pd.DataFrame()
    df['ID_d'] = test_drug_ids
    df['ID_t'] = test_target_ids
    df['Y'] = Y[test_Y_ids]
    df['P'] = P.flatten()
    df['setting'] = [ FLAGS.fourfield_setting for _ in range(df.shape[0]) ]
    df['fold'] = [ FLAGS.cv_fold for _ in range(df.shape[0]) ]
    df['model'] = [ "DeepDTA" for _ in range(df.shape[0]) ]
    df.to_csv(FLAGS.dataset_path + '/results/'+ 'RS' + FLAGS.random_seed + '_' + FLAGS.fourfield_setting + 'F' +  str(FLAGS.cv_fold) + '.csv', index=False)


def train_test(FLAGS, perfmeasure, deepmethod, XD, XT, Y, training_indices, test_indices, num_win, smi_len, seq_len, lr, num_epochs):
    train_drug_inds, train_target_inds, train_Y_inds = separate_indices(training_indices)
    test_drug_inds, test_target_inds, test_Y_inds = separate_indices(test_indices)

    logging("Training model on the combined training and validation set", FLAGS)

    model = deepmethod(FLAGS, num_win, smi_len, seq_len, lr)
    results = model.fit(([ XD[train_drug_inds], XT[train_target_inds] ]), Y[train_Y_inds],
                            batch_size=FLAGS.batch_size, epochs=num_epochs, shuffle=True) 
    Y_pred = model.predict((XD[test_drug_inds], XT[test_target_inds]))
    mse = perfmeasure(Y[test_Y_inds], Y_pred.flatten())
    ci = cindex(Y[test_Y_inds], Y_pred)

    print("Model trained on train+validation data with hyperparameters")
    print("----------------------------------------------------------")
    print("Number of convolution windows:",num_win)
    print("SMILES window length:",smi_len)
    print("Protein sequence window length:",seq_len)
    print("Learning rate:",lr)
    print("Test set C-index:",ci)
    print("Test set MSE:",mse)
    print("Epochs:",num_epochs)

    logging("Test set CI: %f" % (ci,), FLAGS)
    logging("Test set MSE: %f" % (mse,), FLAGS)

    model.save(FLAGS.dataset_path + 'saved_models/' + 'RS' + FLAGS.random_seed + '_' + FLAGS.fourfield_setting + 'F' + str(FLAGS.cv_fold))

    return Y, Y_pred

def select_hyperparameters(FLAGS, perfmeasure, deepmethod, XD, XT, Y, training_indices, validation_indices):
    
    train_drug_inds, train_target_inds, train_Y_inds = separate_indices(training_indices)
    val_drug_inds, val_target_inds, val_Y_inds = separate_indices(validation_indices)

    best_lr = 0
    best_num_win = 0
    best_smi_len = 0
    best_seq_len = 0
    num_epochs = 100
    
    best_mse = 1e20
    #best_ci = 0.5

    logging("Setting: %s" % (FLAGS.fourfield_setting,), FLAGS)
    logging("Searching hyperparameter space num_win: %s - smi_win_len: %s - seq_win_len: %s - lr: %s" %
            (", ".join(str(x) for x in FLAGS.num_windows), ", ".join(str(x) for x in FLAGS.smi_window_lengths),
             ", ".join(str(x) for x in FLAGS.seq_window_lengths), ", ".join(str(x) for x in FLAGS.learning_rates)), FLAGS)
    

    for lr in FLAGS.learning_rates:
        for num_win in FLAGS.num_windows:
            for smi_len in FLAGS.smi_window_lengths:
                for seq_len in FLAGS.seq_window_lengths:
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
                    model = deepmethod(FLAGS, num_win, smi_len, seq_len, lr)
                    results = model.fit(([ XD[train_drug_inds], XT[train_target_inds] ]), Y[train_Y_inds],
                                            batch_size=FLAGS.batch_size, epochs=FLAGS.num_epoch, 
                                            validation_data=( ([ XD[val_drug_inds], XT[val_target_inds] ]), Y[val_Y_inds]),
                                            shuffle=True, callbacks=[es] ) 

                    Y_pred = model.predict((XD[val_drug_inds], XT[val_target_inds]))
                    mse = perfmeasure(Y[val_Y_inds], Y_pred.flatten())
                    print("Trained with hyperparameters")
                    print("-----------------------------")
                    print("Number of convolution windows:",num_win)
                    print("SMILES window length:",smi_len)
                    print("Protein sequence window length:",seq_len)
                    print("Learning rate:",lr)
                    print("MSE:",mse)
                    if mse < best_mse:
                        best_mse = mse
                        print("Best mse updated to",best_mse)
                        best_lr = lr
                        best_num_win = num_win
                        best_smi_len = smi_len
                        best_seq_len = seq_len
                        # if stopped early, when?
                        num_epochs = len(results.history['loss'])

                        logging("MSE improved to %f with hyperparameters num_win: %d, smi_len: %d, seq_len: %d, lr: %s" %
                                (mse, num_win, smi_len, seq_len, lr), FLAGS)
                    print()


    return best_lr, best_num_win, best_smi_len, best_seq_len, num_epochs, best_mse

class SavePredictionsCallback(keras.callbacks.Callback):
    def __init__(self, val_sets, FLAGS, predictions_df, save_freq=1):
        self.FLAGS = FLAGS
        '''
        val_sets is a list of validation sets.
        one validation set must be a dictionary such that:
        { 
            'XD': drugs,
            'XT': targets,
            'Y': val_set_labels,
            'ID_d': val_set_drug_indices,
            'ID_t': val_set_target_indices,
            'ID_y': val_set_label_indices,
            'purpose': 'valid' OR 'test',
            'lr': learning_rate,
            'num_win': number of convolution windows,
            'smi_len': SMILES window length,
            'seq_len': sequence window length
        }
        '''
        self.val_sets = val_sets
        '''
        pass in a DataFrame of previously saved results
        pass an empty DataFrame if no results have been saved on previous epochs yet
        '''
        self.predictions_df = predictions_df
        '''
        pass in n > 1 if predictions should only be saved every nth epoch
        '''
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):

        if ((epoch+1) % self.save_freq) != 0:
            print("\nnot saving predictions for epoch",epoch+1,'due to save_freq\n')
            return

        for val_set in self.val_sets:
            drugs = val_set['XD'][val_set['ID_d']]
            targets = val_set['XT'][val_set['ID_t']]
            purpose = val_set['purpose']

            P = self.model.predict((drugs, targets))

            # save predictions to file
            df = pd.DataFrame()
            df['ID_d'] = val_set['ID_d']
            df['ID_t'] = val_set['ID_t']
            df['Y'] = val_set['Y'][val_set['ID_y']]
            df['P'] = P.flatten()
            df['testset'] = [ val_set['purpose'] for _ in range(df.shape[0]) ]
            df['setting'] = [ self.FLAGS.fourfield_setting for _ in range(df.shape[0]) ]
            df['fold'] = [ self.FLAGS.cv_fold for _ in range(df.shape[0]) ]
            df['model'] = [ "DeepDTA" for _ in range(df.shape[0]) ]
            df['lr'] = [ val_set['lr'] for _ in range(df.shape[0]) ]
            df['num_win'] = [ val_set['num_win'] for _ in range(df.shape[0]) ]
            df['smi_len'] = [ val_set['smi_len'] for _ in range(df.shape[0]) ]
            df['seq_len'] = [ val_set['seq_len'] for _ in range(df.shape[0]) ]
            df['epoch'] = [ epoch+1 for _ in range(df.shape[0]) ]
            df['random_seed'] = [ self.FLAGS.random_seed for _ in range(df.shape[0]) ]

            if self.predictions_df.shape[0] == 0:
                self.predictions_df = df
            else:
                self.predictions_df = pd.concat([self.predictions_df, df])

    def get_predictions(self):
        return self.predictions_df

def metricsearch_experiment(FLAGS, deepmethod):

    XD, XT, Y = load_rawtext_data(FLAGS)
    train_ids, val_ids, test_ids = load_cv_indices(FLAGS)

    train_drug_inds, train_target_inds, train_Y_inds = separate_indices(train_ids)
    val_drug_inds, val_target_inds, val_Y_inds = separate_indices(val_ids)
    test_drug_inds, test_target_inds, test_Y_inds = separate_indices(test_ids)

    validation_data=( ([ XD[val_drug_inds], XT[val_target_inds] ]), Y[val_Y_inds]),

    logging("Setting: %s" % (FLAGS.fourfield_setting,), FLAGS)
    logging("Saving predictions for both validation and test sets for the whole hyperparameter space: num_win: %s - smi_win_len: %s - seq_win_len: %s - lr: %s" %
            (", ".join(str(x) for x in FLAGS.num_windows), ", ".join(str(x) for x in FLAGS.smi_window_lengths),
             ", ".join(str(x) for x in FLAGS.seq_window_lengths), ", ".join(str(x) for x in FLAGS.learning_rates)), FLAGS)
    
    # all the predictions from all epochs for all hyperparameters are stored in this DataFrame
    predictions_df = pd.DataFrame()

    for lr in FLAGS.learning_rates:
        for num_win in FLAGS.num_windows:
            for smi_len in FLAGS.smi_window_lengths:
                for seq_len in FLAGS.seq_window_lengths:

                    # create validation and test sets for the callback that
                    #+saves predictions at the end of each epoch
                    val_dict = {
                            'XD': XD,
                            'XT': XT,
                            'Y': Y,
                            'ID_d': val_drug_inds,
                            'ID_t': val_target_inds,
                            'ID_y': val_Y_inds,
                            'purpose': 'valid',
                            'lr': lr,
                            'num_win': num_win,
                            'smi_len': smi_len,
                            'seq_len': seq_len
                            }
                    test_dict = {
                            'XD': XD,
                            'XT': XT,
                            'Y': Y,
                            'ID_d': test_drug_inds,
                            'ID_t': test_target_inds,
                            'ID_y': test_Y_inds,
                            'purpose': 'test',
                            'lr': lr,
                            'num_win': num_win,
                            'smi_len': smi_len,
                            'seq_len': seq_len
                            }

                    save_pred = SavePredictionsCallback([val_dict, test_dict], FLAGS, predictions_df, save_freq=FLAGS.save_freq)

                    model = deepmethod(FLAGS, num_win, smi_len, seq_len, lr)
                    results = model.fit(([ XD[train_drug_inds], XT[train_target_inds] ]), Y[train_Y_inds],
                                            batch_size=FLAGS.batch_size, epochs=FLAGS.num_epoch, 
                                            shuffle=True, callbacks=[save_pred]) 

                    predictions_df = save_pred.get_predictions()

    pred_filename = FLAGS.dataset_path + '/metricsearch_predictions/' + FLAGS.now + '/' + \
                          's' + FLAGS.fourfield_setting[1] + 'f' + str(FLAGS.cv_fold) + '.csv'

    predictions_df.to_csv(pred_filename, index=False)



def run_regression( FLAGS ): 
    perfmeasure = mse
    deepmethod = build_model
    
    if FLAGS.crossvalidation:
        print("Running in cross-validation mode")
        crossvalidation_experiment(FLAGS, perfmeasure, deepmethod)
    elif FLAGS.metricsearch:
        # optimising hyperparameters for each metric separately may waste GPU time:
        #+if each is run for e.g. 90 epochs, we'll end up with 270 epochs per hyper parameter combination

        # !!if train+valid set are not being combined!!
        #+especially in light of the fact that on CSC, GPU time is billed regardless of how much the GPU is used,
        #+it makes more sense to calculate and save all predictions for all epochs, for both validation and test sets,
        #+and calculate metrics on the predictions alone afterwards. that way there's no need
        #+to reserve a GPU while using CPU to calculate metrics

        # the way "metric search" is implemented here, then, is to simply run for maximum epochs and
        #+save predictions on both valid and test set. separate code is needed to later process the predictions

        # the obvious trade-off is that potentially a lot of data will need to be stored

        print("Running in metric search mode, which practically means saving predictions for all epochs")

        if not os.path.exists(FLAGS.dataset_path + '/metricsearch_predictions/' + FLAGS.now):
            os.makedirs(FLAGS.dataset_path + '/metricsearch_predictions/' + FLAGS.now)

        metricsearch_experiment(FLAGS, deepmethod)
        

    else:
        print("pass ...")
        exit()

if __name__=="__main__":

    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    # ugly magic numbers - these need to be changed if the drug or target representation changes
    FLAGS.charseqset_size = 25 
    FLAGS.charsmiset_size = 62

    # for saving some files
    # NOTE: current implementation overwrites results for one setting+fold (e.g. s1f0) combination that were generated on the same day
    # use a more granular FLAGS.now to easily save different results more frequently
    FLAGS.now = datetime.now().strftime('%Y%m%d')

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    run_regression( FLAGS )
