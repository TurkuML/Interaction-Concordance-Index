import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
from rlscore.measure.cindex_measure import cindex as rlscore_ci

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} {:.0f}%\tLoss: {:.6f}'.format(epoch,
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


datasets = [['davis','kiba'][int(sys.argv[1])]] 
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
print('cuda_name:', cuda_name)

foldnum = sys.argv[4]

setting = sys.argv[5]

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 2
NUM_EPOCHS = 1000

if len(sys.argv)>6:
    NUM_EPOCHS=int(sys.argv[6])

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train_' + setting + '_' + foldnum + '.pt'
    processed_data_file_valid = 'data/processed/' + dataset + '_valid_' + setting + '_'+ foldnum + '.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test_' + setting + '_'+ foldnum + '.pt'

    if (not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_valid)) or (not os.path.isfile(processed_data_file_test)):
        print('please run create_cv_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train_'+setting+'_'+foldnum)
        valid_data = TestbedDataset(root='data',dataset=dataset+'_valid_'+setting+'_'+foldnum)
        test_data = TestbedDataset(root='data', dataset=dataset+'_test_'+setting+'_'+foldnum)
        # first train on training set only
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        # use validation set for hyperparameter selection
        valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        # predict on test set with the chosen hyperparameters
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1e15
        best_epoch = -1
        val_progress = []

        # hyperparameter selection currently only chooses the number of epochs to train
        print("Selecting hyperparameters...")
        for epoch in range(1, NUM_EPOCHS+1):
            train(model, device, train_loader, optimizer, epoch)
            print('predicting for validation data')
            Y,P = predicting(model, device, valid_loader)
            val_loss = mse(Y,P)

            if val_loss<best_mse:
                best_mse = val_loss
                best_epoch = epoch
                print('val_loss improved to ',val_loss,' at epoch ',best_epoch, sep='')
                
                # validation result improved. predict on test set and save results
                Y_test, P_test = predicting(model, device, test_loader)

                # save the trained model
                model_file_name = 'model_' + model_st + '_' + dataset + '_' + setting + '_' + foldnum +  '.model'
                torch.save(model.state_dict(), model_file_name)
            else:
                print('epoch: ',epoch,' val_loss: ',val_loss,', no improvement since epoch ',best_epoch, sep='')

        print("Saving final predictions on test set...")
        # get the real affinities
        df = pd.read_csv("data/" + dataset + "_test_" + setting + '_' + foldnum + ".csv")
        # drop features
        df = df.drop('compound_iso_smiles', axis=1)
        df = df.drop('target_sequence', axis=1)
        # save predictions and other information
        df['P'] = P_test
        df['setting'] = [ setting for _ in range(df.shape[0]) ]
        df['fold'] = [ foldnum for _ in range(df.shape[0]) ]
        df['model'] = [model_st for _ in range(df.shape[0]) ]
        df.to_csv('data/' + dataset + '/results/' + setting + 'F' +  foldnum + '.csv', index=False)

