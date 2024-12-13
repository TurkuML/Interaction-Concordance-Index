import tensorflow as tf
from tensorflow import keras
from keras import layers
from rlscore.measure.cindex_measure import cindex
import os
from datetime import datetime
import pandas as pd
import sys

from arguments import parse_arguments
from datahelper import *

ARGS = parse_arguments(sys.argv)
print(vars(ARGS))

if ARGS.dataset == "davis":
    XD = np.loadtxt("drug-drug_similarities_2D__Davis_et_al.2011.txt", dtype=np.float32)
    XT = np.loadtxt("target-target_similarities_WS_normalized__Davis_et_al.2011.txt", dtype=np.float32)
    Y_noz = np.loadtxt("drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt", dtype=np.float32)
elif ARGS.dataset == "kiba":
    XD = np.loadtxt("kiba_drug_sim.txt", dtype=np.float32)
    XT = np.loadtxt("kiba_target_sim.txt", dtype=np.float32)
    Y_noz = np.loadtxt("kiba_binding_affinity_v2.txt", dtype=np.float32)
elif ARGS.dataset == "metz":
    XD = np.loadtxt("drug-drug_similarities_2D__Metz_et_al.2011.txt", dtype=np.float32)
    XT = np.loadtxt("target-target_similarities_WS_normalized__Metz_et_al.2011.txt", dtype=np.float32)
    Y_noz = np.loadtxt("known_drug-target_interaction_affinities_pKi__Metz_et_al.2011.txt", dtype=np.float32)
elif ARGS.dataset == "merget":
    XD = np.loadtxt("Kd_Tanimoto-shortestpath.txt", dtype=np.float32)
    XT = np.loadtxt("Kp_GS-ATP_L5_Sp4.0_Sc4.0.txt", dtype=np.float32)
    Y_noz = np.loadtxt("Merget_DTIs_2967com_226kin.txt", dtype=np.float32)
Y_noz = Y_noz.ravel()
# the structural information from the matrix is not needed,
#+because the same information is stored in the split files
Y_noz = Y_noz[np.logical_not(np.isnan(Y_noz))]

train_inds, valid_inds, test_inds, splits_random_seed = load_cv_indices(ARGS.setting, ARGS.cv_fold, ARGS.splits_file)
train_drug, train_target, train_Y = separate_indices(train_inds)
valid_drug, valid_target, valid_Y = separate_indices(valid_inds)
test_drug, test_target, test_Y = separate_indices(test_inds)

# using train set mean and standard deviation, to ensure that no info is leaked to validation/test data
# affinity mean and standard deviation are saved, to allow converting predictions back
train_affinities_mean = Y_noz[train_Y].mean()
train_affinities_std = Y_noz[train_Y].std()

# z-score
XD = (XD - XD[train_drug].mean()) / XD[train_drug].std()
XT = (XT - XT[train_target].mean()) / XT[train_target].std()
Y = (Y_noz - train_affinities_mean) / train_affinities_std

def build_and_compile_model(num_layers, neurons_shape, dropout, lr):
    # neurons_shape is a list with length >= num_layers
    XD_in = layers.Input(shape=XD.shape[0])
    XT_in = layers.Input(shape=XT.shape[0])
    FC = layers.concatenate([XD_in, XT_in], axis=1)

    for i in range(num_layers):
        FC = layers.Dense(neurons_shape[i], activation='relu')(FC)
        FC = layers.Dropout(dropout)(FC)
    out = layers.Dense(1)(FC)

    model = keras.models.Model(inputs=[XD_in, XT_in], outputs=[out])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    return model

class SavePredictionsCallback(keras.callbacks.Callback):
    def __init__(self, val_sets, random_seed, save_freq=1):
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
        }
        '''
        self.val_sets = val_sets
        '''
        pass in a DataFrame of previously saved results
        pass an empty DataFrame if no results have been saved on previous epochs yet
        '''
        self.predictions_df = pd.DataFrame()
        '''
        pass in n > 1 if predictions should only be saved every nth epoch
        '''
        self.save_freq = save_freq

        self.random_seed = random_seed

    def on_epoch_end(self, epoch, logs=None):
        if ((epoch+1) % self.save_freq) != 0:
            print("\nnot saving predictions for epoch",epoch+1,'due to save_freq\n')
            return

        for val_set in self.val_sets:
            drugs = val_set['XD'][val_set['ID_d']]
            targets = val_set['XT'][val_set['ID_t']]

            P = self.model.predict((drugs, targets))

            # save predictions to file
            df = pd.DataFrame()
            df['ID_d'] = val_set['ID_d']
            df['ID_t'] = val_set['ID_t']
            df['Y'] = val_set['Y'][val_set['ID_y']]
            df['P'] = (P*train_affinities_std + train_affinities_mean).flatten()
            df['testset'] = [ val_set['purpose'] for _ in range(df.shape[0]) ]
            df['setting'] = [ ARGS.setting for _ in range(df.shape[0]) ]
            df['fold'] = [ ARGS.cv_fold for _ in range(df.shape[0]) ]
            df['random_seed'] = [ self.random_seed for _ in range(df.shape[0]) ]
            df['model'] = [ "feedforward" for _ in range(df.shape[0]) ]
            df['lr'] = [ self.lr for _ in range(df.shape[0]) ]
            df['epoch'] = [ epoch+1 for _ in range(df.shape[0]) ]
            df['batch_size'] = [ self.batch_size for _ in range(df.shape[0]) ]
            df['num_layers'] = [ self.num_layers for _ in range(df.shape[0]) ]
            df['neurons_in_layers'] = [ " ".join(map(str,self.neurons_in_layers)) for _ in range(df.shape[0]) ]
            df['dropout_ratio'] = [ self.dropout_ratio for _ in range(df.shape[0]) ]

            if self.predictions_df.shape[0] == 0:
                self.predictions_df = df
            else:
                self.predictions_df = pd.concat([self.predictions_df, df])

    ''' set the hyperparameter values to be saved.
        takes a dictionary such that:
        {
            'num_layers': val set num_layers,
            'neurons_in_layers': list of numbers of neurons in layers (sequential),
            'dropout_ratio': how much dropout between layers (same for all)
            'lr': learning_rate,
            'batch_size': how many samples in one minibatch
        }
    '''
    def set_hyperparameters(self, hyperparamdict):
        self.num_layers = hyperparamdict['num_layers']
        self.neurons_in_layers = hyperparamdict['neurons_in_layers']
        self.dropout_ratio = hyperparamdict['dropout_ratio']
        self.lr = hyperparamdict['lr']
        self.batch_size = hyperparamdict['batch_size']

    def get_predictions(self):
        return self.predictions_df

# create validation and test sets for the callback that
#+saves predictions at the end of each epoch
val_dict = {
        'XD': XD,
        'XT': XT,
        'Y': Y_noz,
        'ID_d': valid_drug,
        'ID_t': valid_target,
        'ID_y': valid_Y,
        'purpose': 'valid',
}
test_dict = {
        'XD': XD,
        'XT': XT,
        'Y': Y_noz,
        'ID_d': test_drug,
        'ID_t': test_target,
        'ID_y': test_Y,
        'purpose': 'test',
}


save_pred = SavePredictionsCallback([val_dict, test_dict], random_seed=splits_random_seed, save_freq=ARGS.save_freq)

for epochs in ARGS.epochs:
    for batch_size in ARGS.batch_size:
        for learning_rate in ARGS.learning_rate:
            for num_layers in ARGS.num_layers:
                for neurons_in_layers in ARGS.neurons_in_layers:
                    for dropout_ratio in ARGS.dropout_ratio:
                        print("Training with hyperparameters:")
                        print("Epochs:\t\t",epochs)
                        print("Batch size:\t",batch_size)
                        print("Learning rate:",learning_rate)
                        print("Number of layers:",num_layers)
                        print("Neurons in layers:",neurons_in_layers)
                        print("Dropout ratio:",dropout_ratio)

                        hyperparamdict = {
                                'num_layers': num_layers,
                                'neurons_in_layers': neurons_in_layers,
                                'dropout_ratio': dropout_ratio,
                                'lr': learning_rate,
                                'batch_size': batch_size
                        }

                        save_pred.set_hyperparameters(hyperparamdict)

                        model = build_and_compile_model(num_layers=num_layers, neurons_shape=neurons_in_layers,
                                                        dropout=dropout_ratio, lr=learning_rate)

                        history = model.fit(([ XD[train_drug], XT[train_target] ]), Y[train_Y],
                                               batch_size=batch_size, epochs=epochs,
                                               shuffle=True, callbacks=[save_pred])

predictions_df = save_pred.get_predictions()

today = datetime.now().strftime('%Y%m%d')
pred_dir = 'predictions/' + ARGS.dataset + '/' + today + '/'
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

pred_filename =  ARGS.setting + 'F' + ARGS.cv_fold + '.csv'
predictions_df.to_csv(pred_dir+pred_filename, index=False)

