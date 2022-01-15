import numpy as np
from utils import *
from partition import *

def warm_up_hist(feat, dims):
    nd = len(feat.shape)
    nt = np.prod(np.array(feat.shape))
    idq = np.arange(nt, dtype='int')
    X = np.zeros((nt, nd + 1)).astype(int)
    for d in range(dims):
        X[:, d + 1] = idq % np.array(feat.shape)[d]
        idq = (idq / np.array(feat.shape)[d]).astype(int)
    X[:, 0] = feat.transpose().reshape(nt)
    return X

def infer_sparse_hist(feat, dims, low, high, sparse):
    sumv = 0
    if sparse:
        for r in np.where(np.all((feat[:,dims] >= low) & (feat[:,dims] <= high), axis=1))[0]:
            sumv += feat[r][-1]
    else:
        for r in np.where(np.all((feat[:,dims+1] >= low) & (feat[:, dims+1] <= high), axis = 1))[0]:
            sumv += feat[r, 0]
    return sumv

def build_sparse_hist(feat, nss, sparse):
    unq = np.unique(feat, axis=0, return_counts=True)
    feat = []
    if sparse:
        for i in range(len(unq[0])):
            feat += [np.hstack([unq[0][i], unq[1][i]])]
    else:
        feat = np.zeros(np.array(nss).astype(int)).astype(int)
        if len(nss)==1:
            for i in range(len(unq[0])):
                feat[unq[0][i][0]] += unq[1][i]
        elif len(nss)==2:
            for i in range(len(unq[0])):
                feat[unq[0][i][0], unq[0][i][1]] += unq[1][i]
        elif len(nss)==3:
            for i in range(len(unq[0])):
                feat[unq[0][i][0], unq[0][i][1], unq[0][i][2]] += unq[1][i]
    return np.array(feat)


