import numpy as np
import os
import sys
import gzip
from tqdm import tqdm,trange
from partition import *
from iris_io import *
from utils import *

from datetime import date, timedelta


def gen_test(ncd, nd, X_in, nss, normlen, nt, nm, neb, maxd):
    X = np.zeros((1, nt, nd + 1)).astype(int)
    Xs = np.zeros((1, ncd + 2, maxd + 1))
    idq = np.arange(nt, dtype='int')
    X[0, :, 0] += X_in[:]
    for d in range(len(nss)):
        X[0, :, d + 1] = idq % nss[d]
        idq = idq / nss[d]
    Xs[0, :ncd, :nd+1] = shrink(X[0, :, :], nm, nt)
    lens = len(nss)
    Xs[0, :, 1:lens + 1] = np.round((Xs[0, :, 1:lens + 1]) * (neb-1) / ([nn - 1 for nn in nss]))
    sf = normlen/float(sum(Xs[0, :, 0]))
    Xs[0, :ncd, 0] = np.round(np.multiply(Xs[0, :ncd, 0], sf)).astype(int)
    Xs[0, :ncd, 0] *= 1.0 / sum(Xs[0, :ncd, 0])

    Xs[:, :, 1:] = np.maximum(Xs[:, :, 1:], -1) + 1
    return Xs