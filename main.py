#!/usr/bin/env python3
import numpy as np
import pandas as pd

import torch as t
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import argparse as arp
import os
import re
import datetime

from typing import Dict, List

from scipy.spatial.distance import cdist
from QuickST.data import utils as ut
from QuickST.utils import get_crd



def weighted_expression(count : np.ndarray,
                        crd : np.ndarray,
                        sigma : float = 1.0,
                        ) -> np.ndarray :

          wmat = np.exp(- cdist(crd,crd) / sigma )
          denom = wmat.sum(axis = 1).reshape(-1,1)
          denom[ denom == 0.0 ] = np.nan
          wmat = np.divide(wmat,denom)
          wmat[np.isnan(wmat)] = 0.0
          
          return np.dot(wmat,count)

def accuracy_stats(pred, true):

    rounded_pred = pred.round(0).flatten()
    correct = np.sum(rounded_pred == true.flatten())
    stats = np.array([correct, true.shape[0]])

    return stats


class STdata(Dataset):
    def __init__(self,
                 data : np.ndarray,
                 labels : np.ndarray,
                 ):
        
        self.x = data
        self.y = labels.reshape(-1,1)
        
        self.S = data.shape[0]
        self.G = data.shape[1]
    
    def __len__(self,):
        return self.S
    
    def __getitem__(self,idx):
        sample = {'x': self.x[idx,:],
                  'y': self.y[idx,:],
                  }
        
        return sample

class Model(nn.Module):
    def __init__(self, input_dim : int):
        super(Model,self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim,1,
                                bias = True,
                                )
    
    def forward(self,x):
        
        y = t.sigmoid(self.linear(x))
        return y 
    


def fit(train_loader : STdata,
        val_loader : STdata,
        model : Model,
        epochs : int = 1000,
        eval_interval : int = 100,
        device : str = 'cpu',
        lambda1 : float = 0.0,
        ) -> Dict[np.ndarray,np.ndarray]:
        
        
        n_genes = model.input_dim    
        optim = t.optim.Adam(model.parameters(), lr = 0.01)    
        loss_function = nn.BCELoss(reduction = 'mean')
        best = dict(weights = None,
                    bias = None,
                    accuracy = 0,
                    loss = np.inf,
                    epoch = 0,
                    )
        
        
        # implement save upon best validation loss
        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                model.train()
                for tbatch in train_loader:
                    
                    optim.zero_grad()
                    
                    x_train, y_train = Variable(tbatch['x']), tbatch['y']
                    x_train, y_train = x_train.to(device), y_train.to(device)
                    train_pred = model(x_train)
                    
                    loss = loss_function(train_pred,y_train)
                    if lambda1 > 0:
                        l1_loss = t.norm(model.linear.weight.view(-1),p = 1)
                        loss += lambda1 * l1_loss / n_genes
                    
                    loss.backward()
                    epoch_loss += loss.item()
                    
                    optim.step()
                    
                
                print('\r',end = '')
                print(f"epoch {epoch} | loss {epoch_loss}", end = '')
                
                if (epoch % eval_interval == 0 and epoch >= eval_interval)\
                    or epoch == (epochs - 1):
                
                    model.eval()
                    with t.no_grad():
                        acc = np.zeros(2).astype(int)
                        vloss = 0.0
                        for vbatch in val_loader:
                            x_val, y_val = vbatch['x'], vbatch['y']
                            x_val, y_val = x_val.to(device), y_val.to(device)
                            val_pred = model(x_val)
                            vloss += loss_function(val_pred,y_val).item()
                            acc += accuracy_stats(val_pred.numpy(),y_val.numpy())
                            
                        if lambda1 > 0:
                            vloss += lambda1 * l1_loss / n_genes 
                            
                   
                        acc = acc[0] / acc[1]
                        print(f"\nEpoch : {epoch} | Accuracy : {acc} | Loss : {vloss}")
                        if vloss < best['loss']:
                            prm = list(model.parameters())
                            best['loss'] = vloss
                            best['weights'] = prm[0].detach().numpy().reshape(-1,1)
                            best['bias'] = prm[1].detach().numpy().reshape(-1,1)
                            best['epoch'] = epoch
                            best['accuracy'] = acc
                            
        except KeyboardInterrupt:
            print('\nEarly Interruption >> Saving Current Results')
            
        
        return best
    
    
def main(cnt_files : List[str],
         meta_files : List[str],
         colname : str = 'tumor',
         tumor_label : str = 'tumor',
         sigma : float = 1.0,
         pval : float = 0.2,
         batch_size : int = 512,
         epochs : int = 1000,
         device : str = 'cpu',
         lambda1 : float = 0.0,
         ) -> pd.DataFrame:
    
    read_file = lambda file: pd.read_csv(file,
                                         sep = '\t',
                                         header = 0,
                                         index_col = 0 )
    
    cntL = list()
    meta_joint = pd.Series([],name = colname)
    for sample in range(len(cnt_files)):
        # count data
        print(f'loading count file {cnt_files[sample]}')
        print(f'loading meta file {meta_files[sample]}')
        tmp_cnt = read_file(cnt_files[sample])
        tmp_crd = get_crd(tmp_cnt.index)
        
        # normalize by library size
        denom = tmp_cnt.values.sum(axis = 1).reshape(-1,1)
        tmp_cnt.iloc[:,:] = np.divide(tmp_cnt.values,
                                      denom,
                                      where = (denom != 0.0)) 
        
        # weight expression
        tmp_cnt.iloc[:,:] = weighted_expression(tmp_cnt.values,
                                                tmp_crd,
                                                sigma = sigma)
        
        # intersect
        tmp_meta = read_file(meta_files[sample])
        inter = tmp_meta.index.intersection(tmp_cnt.index)
        tmp_cnt = tmp_cnt.loc[inter,:]
        tmp_meta = tmp_meta.loc[inter,colname]
        ishot = (tmp_meta.values.flatten() == tumor_label)
        tmp_meta.iloc[ishot] = 1.0
        tmp_meta.iloc[ishot == False] = 0.0
        
        meta_joint = pd.concat([meta_joint,tmp_meta])
        cntL.append(tmp_cnt)
        
    cnt_joint = ut.join_samples(cntL)
    genes = cnt_joint['joint_matrix'].columns 
    
    all_idx = np.arange(cnt_joint['joint_matrix'].shape[0]).astype(int)
    np.random.shuffle(all_idx)
    n_train = np.floor(all_idx.shape[0]*(1.0-pval)).astype(int)
    
    train_idx = all_idx[0:n_train]
    val_idx = all_idx[n_train::]
    
    tdata = cnt_joint['joint_matrix'].iloc[train_idx,:]
    tdata = t.tensor(tdata.values.astype(np.float32))
    tlabels = t.tensor(meta_joint[train_idx].values.astype(np.float32))
    
    vdata = cnt_joint['joint_matrix'].iloc[val_idx,:]
    vdata = t.tensor(vdata.values.astype(np.float32))
    vlabels = t.tensor(meta_joint[val_idx].values.astype(np.float32))
    
    del cnt_joint
    
    train_loader = DataLoader(STdata(data = tdata, 
                                     labels = tlabels),
                       batch_size = batch_size,
                       )
    
    val_loader = DataLoader(STdata(data = vdata, 
                                   labels = vlabels),
                       batch_size = val_idx.shape[0],
                       )
    
    model = Model(genes.shape[0])
    model = model.to(device)
    
    res = fit(train_loader = train_loader,
              val_loader = val_loader,
              model = model,
              epochs = epochs,
              device = device,
              lambda1 = lambda1,
              )
    
    print(' '.join([f"Using parameters from epoch {res['epoch']}",
                    f"with accuracy {res['accuracy']} and",
                    f"loss {res['loss']}"]))

    format_res = np.vstack((res['bias'],res['weights']))
    format_index = pd.Index(['bias']).append(genes)
    
    
    res = pd.DataFrame(format_res,
                       index = format_index,
                       columns = ['coef'])

    return res
              
              
if __name__ == "__main__":
    
    timestamp = re.sub(':| |-|','', str(datetime.datetime.now()))
    
    prs = arp.ArgumentParser()
    
    prs.add_argument('-c','--count_files',
                     required = True,
                     nargs = '+',
                     help = ''.join([]),
                     )
    
    
    prs.add_argument('-m','--meta_files',
                     required = True,
                     nargs = '+',
                     help = ''.join([]),
                     )
    
    
    prs.add_argument('-o','--output_directory',
                     default = None,
                     help = ''.join([]),
                     )
    
    
    prs.add_argument('-e','--epochs',
                     type = int,
                     default = 1000,
                     help = ''.join([]),
                     )
    
    prs.add_argument('-s','--sigma',
                     type = float,
                     default = 1.0,
                     help = ''.join([]),
                     )
    
    
    prs.add_argument('-n','--colname',
                     type = str,
                     default = 'tumor',           
                     help = ''.join([]),
                     )
    
    
    prs.add_argument('-tl','--tumor_label',
                     default = 'tumor',
                     type = str,
                     help = ''.join([]),
                     )
    
    prs.add_argument('-pv','--p_validation',
                     type = float,
                     default = 0.2,
                     help = ''.join([]),
                     )
    
    prs.add_argument('-bs','--batch_size',
                     type = int,
                     default = 512,
                     help = ''.join([]),
                     )
    
    
    prs.add_argument('-l1','--l1_regularization',
                     type = float,
                     default = 0.0,
                     help = ''.join([]),
                     )
    
    
    prs.add_argument('-g','--gpu',
                     default = False,
                     action = 'store_true',
                     help = ''.join([]),
                     )
    
    
    
    args = prs.parse_args()
    
    if args.gpu:
        device = ('cuda' if t.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    
    print(f"Will be using {device}")
    
    input_args = dict(cnt_files = args.count_files,
                      meta_files = args.meta_files,
                      colname = args.colname,
                      tumor_label = args.tumor_label,
                      pval = args.p_validation,
                      epochs = args.epochs,
                      batch_size = args.batch_size,
                      sigma = args.sigma,
                      lambda1 = args.l1_regularization,
                      )

    if not isinstance(args.count_files,list):
        args.count_files = [args.count_files]
        args.meta_files = [args.meta_files]
    
    args.count_files.sort()
    args.meta_files.sort()
    
    check_alignment = ut.control_lists(list1 = args.count_files,
                                       list2 = args.meta_files)
    if check_alignment:
        print('Input data seems to be matched')
    else:
        print('WARNING: Input data does not seem to be matched')
        
    
    res  = main(**input_args)
    
    if not args.output_directory:
        odir = os.path.dirname(args.cnt_files[0])
    else:
        odir = args.output_directory
    
    if not os.path.exists(odir):
        os.mkdir(odir)
    
    opth = os.path.join(odir,'.'.join(['tumorpred',timestamp,'tsv']))
    res.to_csv(opth,
               sep = '\t',
               header = True,
               index = True)
    
    sigma_file = os.path.join(odir,'.'.join(['sigma',timestamp,'txt']))
    with open(sigma_file,'w+') as fopen:
        fopen.write(str(args.sigma))
        