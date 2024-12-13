import pandas as pd
import numpy as np
import os, sys
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def load_cv_indices(fpath, setting):
    rawcsv = np.loadtxt(fpath, delimiter=',', dtype=str)

    # delete first row (column labels)
    # column labels: roundID,subset,setting,index,drugs,targets
    rawcsv = np.delete(rawcsv, 0, axis=0)

    training_indices = [[] for _ in range(9)]
    validation_indices = [[] for _ in range(9)]
    test_indices = [[] for _ in range(9)]

    for row in rawcsv:
        if row[2] == 'training':
            if row[3] == setting:
                training_indices[row[1].astype(int)].append(row[4].astype(int))

        if row[2] == 'validation':
            if row[3] == setting:
                validation_indices[row[1].astype(int)].append(row[4].astype(int))

        if row[2] == 'test':
            test_indices[row[1].astype(int)].append(row[4].astype(int))

    return training_indices, validation_indices, test_indices

# from DeepDTA data
#datasets = ['kiba','davis']
datasets = ['davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'

    splits_fname = "splits_davis_RS_2688385916.csv"
    setting = sys.argv[1]
    train_fold, valid_fold, test_fold = load_cv_indices(fpath + splits_fname, setting)

    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = np.loadtxt(fpath+"Y.txt")

    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = -np.log10(affinity/1e9)

    opts = ['train', 'valid', 'test']
    for i in range(9):
        for opt in opts:
            rows, cols = np.where(np.isnan(affinity)==False)  
            if opt=='train':
                rows,cols = rows[train_fold[i]], cols[train_fold[i]]
            elif opt=='valid':
                rows,cols = rows[valid_fold[i]], cols[valid_fold[i]]
            elif opt=='test':
                rows,cols = rows[test_fold[i]], cols[test_fold[i]]
            with open('data/' + dataset + '_' + opt + '_' + setting + '_' + str(i) + '.csv', 'w') as f:
                f.write('ID_d,compound_iso_smiles,ID_t,target_sequence,Y\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [rows[pair_ind]]
                    ls += [ drugs[rows[pair_ind]]  ]
                    ls += [cols[pair_ind]]
                    ls += [ prots[cols[pair_ind]]  ]
                    ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                    f.write(','.join(map(str,ls)) + '\n')       
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('valid_fold:', len(valid_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)),len(set(prots)))
    
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

compound_iso_smiles = [[] for _ in range(9)]
smile_graph = [{} for _ in range(9)]
for i in range(9):
    #for dt_name in ['kiba','davis']:
    for dt_name in ['davis']:
        opts = ['train', 'valid', 'test']
        for opt in opts:
            df = pd.read_csv('data/' + dt_name + '_' + opt + '_' + setting + '_'+ str(i) + '.csv')
            compound_iso_smiles[i] += list( df['compound_iso_smiles'] )
    compound_iso_smiles[i] = set(compound_iso_smiles[i])
    for smile in compound_iso_smiles[i]:
        g = smile_to_graph(smile)
        smile_graph[i][smile] = g

#datasets = ['davis','kiba']
datasets = ['davis']
# convert to PyTorch data format
for i in range(9):
    for dataset in datasets:
        processed_data_file_train = 'data/processed/' + dataset + '_train_' + setting + '_'+ str(i) + '.pt'
        processed_data_file_valid = 'data/processed/' + dataset + '_valid_' + setting + '_'+ str(i) + '.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_test_' + setting + '_'+ str(i) + '.pt'
        if ((not os.path.isfile(processed_data_file_train)) or
            (not os.path.isfile(processed_data_file_valid)) or
            (not os.path.isfile(processed_data_file_test))):

            df = pd.read_csv('data/' + dataset + '_train_' + setting + '_'+ str(i) + '.csv')
            train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['Y'])
            XT = [seq_cat(t) for t in train_prots]
            train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)

            df = pd.read_csv('data/' + dataset + '_valid_' + setting + '_'+ str(i) + '.csv')
            valid_drugs, valid_prots,  valid_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['Y'])
            XT = [seq_cat(t) for t in valid_prots]
            valid_drugs, valid_prots,  valid_Y = np.asarray(valid_drugs), np.asarray(XT), np.asarray(valid_Y)

            df = pd.read_csv('data/' + dataset + '_test_' + setting + '_'+str(i) + '.csv')
            test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['Y'])
            XT = [seq_cat(t) for t in test_prots]
            test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)

            # make data PyTorch Geometric ready
            print('preparing ', dataset + '_train_' + setting + '_'+ str(i) + '.pt in pytorch format!')
            train_data = TestbedDataset(root='data', dataset=dataset+'_train_'+setting+'_'+str(i), xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph[i])
            print('preparing ', dataset + '_valid_' + setting + '_'+ str(i) + '.pt in pytorch format!')
            valid_data = TestbedDataset(root='data', dataset=dataset+'_valid_'+setting+'_'+str(i), xd=valid_drugs, xt=valid_prots, y=valid_Y,smile_graph=smile_graph[i])
            print('preparing ', dataset + '_test_' + setting + '_'+ str(i) + '.pt in pytorch format!')
            test_data = TestbedDataset(root='data', dataset=dataset+'_test_'+setting+'_'+str(i), xd=test_drugs, xt=test_prots, y=test_Y,smile_graph=smile_graph[i])
            print(processed_data_file_train, ', ', processed_data_file_valid, ' and ', processed_data_file_test, ' have been created', sep='')
        else:
            print(processed_data_file_train, ', ', processed_data_file_valid,' and ', processed_data_file_test, ' are already created', sep='')        
