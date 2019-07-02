#!/usr/bin/env python3

import pandas as pd
import numpy as np

from QuickST.utils import get_crd
from main import weighted_expression

import matplotlib.pyplot as plt
import argparse as arp
import os
import sys
import re

from typing import Dict

def plot(axdict : Dict[str,plt.Axes],
         labels : Dict[str,np.ndarray],
         crd : np.ndarray,
         ):
   
    cmap = plt.cm.RdYlGn_r
    prm = dict(edgecolor = 'black',
               s = 200,
               cmap = cmap,
               vmin = 0.0,
               vmax = 1.0
               )
    
    axdict['pred'].scatter(x = crd[:,0],
                           y = crd[:,1],
                           c = labels['pred'].reshape(-1,),
                           **prm)
    
    if 'true' in axdict:
        axdict['true'].scatter(x = crd[:,0],
                               y = crd[:,1],
                               c = labels['true'].reshape(-1,),
                               **prm)
        
    for aa in axdict.values():
        aa.set_xticks([])
        aa.set_yticks([])
        for spine in aa.spines.values():
            spine.set_visible(False)
    
    
    return None


def main(cnt,
         crd,
         intercept,
         coef,
         opth,
         labels,
         ):
    
    
    denom = cnt.sum(axis = 1).reshape(-1,1)
    cnt = np.divide(cnt,denom, where = (denom != 0.0))
    cnt = weighted_expression(cnt,crd)
    
    prod = np.dot(cnt, coef) + intercept
    denom = np.exp(-prod) + 1.0
    labels.update({'pred':np.divide(1.0,denom)})
    
    
    figsize = (15,10)
    fig, ax = plt.subplots(nrows = 1,
                           ncols = 2,
                           figsize = figsize,
                           constrained_layout = True)
    ax = ax.flatten()
    
    axdict = dict(pred = ax[0],
                  true = ax[1])
    
    plot(axdict = axdict,
         labels = labels,
         crd = crd,
         )
    
    fig.savefig(opth)
    

if __name__ == "__main__" :
    
    prs = arp.ArgumentParser()
        
    
    prs.add_argument('-c','--count_file',
                     required = True,
                     help = ''.join([]),
                     )
    
    prs.add_argument('-m','--meta_file',
                     required = False,
                     default = None,
                     type = str,
                     help = ''.join([]),
                     )

    prs.add_argument('-b','--beta_file',
                     required = True,
                     help = ''.join([]),
                     )
    
    prs.add_argument('-o','--output_directory',
                     default = None,
                     required = False,
                     help = ''.join([]),
                     )
    
    prs.add_argument('-s','--sigma',
                     default = "1",
                     type = str,
                     required = False,
                     help = ''.join([]),
                     )


    args = prs.parse_args()
    
    if not args.output_directory:
        odir = os.path.dirname(args.count_file)
    else:
        odir = args.output_directory
    
    if not os.path.exists(odir):
        os.mkdir(odir)
    
    filename = re.sub('.tsv','.png',os.path.basename(args.count_file))
    opth = os.path.join(odir,filename)
    
    read_file = lambda file: pd.read_csv(file,
                                         sep = '\t',
                                         header = 0,
                                         index_col = 0 )
    
    cnt = read_file(args.count_file)
    beta = read_file(args.beta_file)
    
    intergenes = cnt.columns.intersection(beta.index)
    cnt = cnt.loc[:,intergenes]
    
    labels = {}
    
    if args.meta_file:
        meta = read_file(args.meta_file)
        interspot = cnt.index.intersection(meta.index)
        meta = meta.loc[interspot,:]
        cnt = cnt.loc[interspot,:]
    
        true_labels = meta.loc[:,'tumor'].values
        true_labels[true_labels == "tumor"] = 1.0
        true_labels[true_labels != 1.0] = 0.0
        true_labels = true_labels.astype(np.float).reshape(-1,1)
        labels.update({'true':true_labels})
        
    if args.sigma.isnumeric():
        sigma = float(args.sigma)
    elif os.path.exists(args.sigma):
         with open(args.sigma,'r+') as fopen:
             sigma = float(fopen.readlines()[0].strip('\n'))
    else:
        print(f"sigma path does not exist. EXIT.")
        sys.exit(-1)
    
    crd = get_crd(cnt.index)
    intercept = np.float(beta.loc['bias',:].values)
    coef = beta.loc[intergenes,:].values
    
    main(cnt = cnt.values,
         labels = labels,
         crd = crd,
         intercept = intercept,
         coef = coef,
         opth = opth,
         )
    