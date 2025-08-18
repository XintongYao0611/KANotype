import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch

from .utils import *

class MyDataSet(Dataset):
    """ 
    Preproces input matrix and labels.

    """
    def __init__(self, exp, label):
        self.exp = exp
        self.label = label
        self.len = len(label)
    def __getitem__(self,index):
        return self.exp[index],self.label[index]
    def __len__(self):
        return self.len


def balance_populations(data):
    ct_names = np.unique(data[:,-1])
    ct_counts = pd.value_counts(data[:,-1])
    max_val = min(ct_counts.max(),np.int32(2000000/len(ct_counts)))
    balanced_data = np.empty(shape=(1,data.shape[1]),dtype=np.float32)
    for ct in ct_names:
        tmp = data[data[:,-1] == ct]
        idx = np.random.choice(range(len(tmp)), max_val)
        tmp_X = tmp[idx]
        balanced_data = np.r_[balanced_data,tmp_X]
    return np.delete(balanced_data,0,axis=0)


def splitDataSet(adata,label_name='Celltype', tr_ratio= 0.7): 
    """ 
    Split data set into training set and test set.

    """
    label_encoder = LabelEncoder()
    el_data = pd.DataFrame(todense(adata),index=np.array(adata.obs_names).tolist(), columns=np.array(adata.var_names).tolist())
    el_data[label_name] = adata.obs[label_name].astype('str')
    #el_data = pd.read_table(data_path,sep=",",header=0,index_col=0)
    genes = el_data.columns.values[:-1]
    el_data = np.array(el_data)
    # el_data = np.delete(el_data,-1,axis=1)
    el_data[:,-1] = label_encoder.fit_transform(el_data[:,-1])
    inverse = label_encoder.inverse_transform(range(0,np.max(el_data[:,-1])+1))
    el_data = el_data.astype(np.float32)
    el_data = balance_populations(data = el_data)
    n_genes = len(el_data[1])-1

    if tr_ratio > 0:
        train_size = int(len(el_data) * tr_ratio)
        train_dataset, valid_dataset = torch.utils.data.random_split(el_data, [train_size,len(el_data)-train_size])
        exp_train = torch.from_numpy(np.array(train_dataset)[:,:n_genes].astype(np.float32))
        label_train = torch.from_numpy(np.array(train_dataset)[:,-1].astype(np.int64))
        exp_valid = torch.from_numpy(np.array(valid_dataset)[:,:n_genes].astype(np.float32))
        label_valid = torch.from_numpy(np.array(valid_dataset)[:,-1].astype(np.int64))
        return exp_train, label_train, exp_valid, label_valid, inverse, genes
    else:
        test_dataset = el_data
        exp_test = torch.from_numpy(np.array(test_dataset)[:,:n_genes].astype(np.float32))
        label_test = torch.from_numpy(np.array(test_dataset)[:,-1].astype(np.int64))
        return exp_test, label_test
    