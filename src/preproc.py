import numpy as np
import sys
from tqdm import tqdm,trange
from partition import *
from iris_io import *
from utils import *
from datetime import date, timedelta

def prep_train(nl, lend, nt, nb, neb, nlen, dic, istest, usecpp, icache, bcache, ccache):
    X = np.zeros((nl, nt, 1)).astype(int)
    NSS, TYPE, PL, PH = [], [], [], []
    for i in tqdm(range(nl), desc='Preparing examples: '):
        if istest == 2:  # test
            id = int(i / lend)
            nqss = [int(i % lend)]
            if (id >= len(dic['DVcol'])):
                print("Err_TooManyTests")
                sys.exit()
        elif istest == 0: # pre-train
            id = int(i % len(dic['DVcol']))
            nqss = [nn for nn in range(lend)]
        elif istest == 1: # validate during pre-train
            id = int(i % len(dic['DVcol']))
            nqss = [int(i / len(dic['DVcol'])) % lend]

        #dvcol 0
        dvcol = dic['DVcol'][id][0] if len(dic['DVcol'][id]) > 0 else []

        if (len(dic['DimC'][id]) > 0):
            ds = np.array(dic['DimC'][id])
            nss = rebucket(np.array(dic['Cnts'][id]), ds, nt, nb, neb, 'Point')
        elif (len(dic['DimR'][id]) > 0):
            ds = np.array(dic['DimR'][id])
            nss = rebucket(np.array(dic['Cnts'][id]), ds, nt, nb, neb, 'Range', [np.where(ds == dvcol)[0]])
        else:
            print('Err_Type: ' + dic['Table'][id])
            sys.exit()

        nsssort, dimsort = map(list, zip(*sorted(zip(nss, ds), reverse=True)))
        NSS += [nsssort]
        CNTs = [dic['Cnts'][id][d] for d in dimsort]
        KEYs = [dic['Keys'][id][d] for d in dimsort]
        TYPES = [dic['Types'][id][d] for d in dimsort]

        ckey = dic['Table'][id] + ';' + str(nsssort) + ';' + str(dimsort)
        KEYs, CNTs = bkt_shrink(ccache, ckey, KEYs, CNTs, TYPES, nsssort, usecpp, [dimsort.index(dvc) for dvc in dvcol])
        for d in range(len(CNTs)):
            for k in range(1, len(CNTs[d])):
                CNTs[d][k] += CNTs[d][k - 1]
        KEYs = [parse_keys(KEYs[id], TYPES[id]) for id in range(len(KEYs))]


        pl, ph = [], []
        for nqs in nqss:
            pred_low = dic['Pred_low'][id][nqs] if len(dic['Pred_low'][id]) > nqs else []
            pred_high = dic['Pred_high'][id][nqs] if len(dic['Pred_high'][id]) > nqs else []

            if len(pred_low) > 0:
                pred_low_r, pred_high_r = [], []
                for d in range(len(nss)):
                    pred_low_r += [pred_low[int(np.where(ds == dimsort[d])[0])]]
                    pred_high_r += [pred_high[int(np.where(ds == dimsort[d])[0])]]
                for d in range(len(nss)):
                    if (TYPES[d] == 'R'):
                        pred_low_r[d] = float(pred_low_r[d])
                        pred_high_r[d] = float(pred_high_r[d])
                    elif (TYPES[d] == 'D'):
                        dl, dh = parse_date(pred_low_r[d]), parse_date(pred_high_r[d])
                        pred_low_r[d] = (date(dl[0], dl[1], dl[2]) - date(1900, 1, 1)).days
                        pred_high_r[d] = (date(dh[0], dh[1], dh[2]) - date(1900, 1, 1)).days
                    else:  # 'C' do nothing
                        pass
                pred_low, pred_high, _ = readp(pred_low_r, pred_high_r, KEYs, nsssort)
            pl += [pred_low]
            ph += [pred_high]
        PL += [pl]
        PH += [ph]

        Fnms = dic['Fnms'][id].copy()

        slen = 10000
        if istest == 0:
            if np.random.randint(2) == 0:
                nlenc = np.random.randint(1, int(nlen/slen+1))*slen
            else:
                nlenc = nlen
            if np.random.randint(4)==0:
                np.random.shuffle(Fnms)
            else:
                Fnms = np.roll(Fnms, np.random.randint(len(Fnms)))
        else:
            nlenc = nlen
            #np.random.seed(0)
            np.random.shuffle(Fnms)

        Vo = readr(icache, bcache, Fnms[0:int(nlenc/slen)], TYPES, nsssort, np.array(dimsort), CNTs)
        V = parse_raw(Vo, KEYs, nss)

        for d in range(1, len(nss)):
            #V[:, 0] = V[:, 0] + V[:, d] * prod(nsssort, d)
            V[:, 0] = V[:, 0] + V[:, d] * prod(nss, d)
        cnt = np.bincount(V[:, 0])
        key = np.nonzero(cnt)
        for k in key:
            X[i, k, 0] += cnt[k]
    return X, NSS, PL, PH

def gen_train(X_in, nss, nm, nt, nbat, neb, normlen, nl, nd, maxd, istest, PL=[], PH=[], P=[], type='ub'):
    cnt_X = np.zeros((nbat)).astype(float)

    ncd = nt-(int)(nt*(1-nm))
    Tns, FIDS = [], []
    X = np.zeros((nbat, nt + 2, nd + 1))
    X[:, -2:, 0] = 1
    # Stage 1: base data structure & predicate
    for i in range(nbat):
        idq = np.arange(nt, dtype='int')
        fid = i if istest == 2 else np.random.randint(nl)
        X[i, :nt, 0] += X_in[fid, :nt, 0]
        for d in range(len(nss[fid])):
            X[i, :-2, d + 1] = (idq % nss[fid][d]).astype(int)
            idq = idq / nss[fid][d]
        Xc = X[i, :-2, :].copy()
        FIDS += [fid]

        if (istest == 0):  # data augmentation for training
            for d in range(len(nss[fid])):
                usezoom = np.random.randint(0, 4)
                usepan = np.random.randint(0, 2)
                useflip = np.random.randint(0, 2)
                if usezoom <= 1:
                   width = min(nss[fid][d] - 1, max(1, int(nss[fid][d] * (np.random.randint(1, 5) / 5.0))))
                   left = np.random.randint(nss[fid][d] - width)
                   ratio = float(width) / nss[fid][d]
                   if usezoom == 0:  # zoom out
                       X[i, :-2, d + 1] = (left + X[i, :-2, d + 1] * ratio).astype(int)
                   else:  # zoom in
                       X[i, :-2, d + 1] = ((X[i, :-2, d + 1] - left) / ratio).astype(int) % nss[fid][d]
                if usepan:
                    X[i, :-2, d + 1] = (X[i, :-2, d + 1] + np.random.randint(0, nss[fid][d])) % nss[fid][d]
                if useflip:
                    X[i, :-2, d + 1] = -X[i, :-2, d + 1] + nss[fid][d] - 1
            # if np.random.randint(0, 4) == 0: #partial resample
            #    sfids = np.arange(nt)
            #    np.random.shuffle(sfids)
            #    spid = np.random.randint(nt * 0.2, nt * .8)
            #    suma = sum(X[i, sfids[:spid], 0])
            #    sumb = sum(X[i, sfids[spid:], 0])
            #    if (suma != 0):
            #        X[i, sfids[:spid], 0] = np.round(np.multiply(X[i, sfids[:spid], 0], float(sumb + suma) / suma)).astype(int)
            #        X[i, sfids[spid:], 0] = 0

            #sort
            dic={}
            for t in range(nt):
                if(X[i, t, 0]):
                    key = sum(X[i, t, 1:] * primes[:maxd])
                    if key in dic:
                        dic[key][1] += X[i, t, 0]
                    else:
                        dic[key] = [X[i, t, 1:], X[i, t, 0]]
            X[i, :-2, :] = Xc
            keys, values = np.array([dic[v][0] for v in dic.keys()]).astype(int), np.array([dic[v][1] for v in dic.keys()])
            for d in range(1, len(nss[fid])):
                keys[:, 0] = keys[:, 0] + keys[:, d] * prod(nss[fid], d)
            X[i, keys[:, 0], 0] = values

    for i in range(nbat):
        if istest == 0 and np.random.randint(2) == 0:
            aid = np.random.choice(np.where(np.all(np.array([nss[fid] for fid in FIDS]) == nss[FIDS[i]], axis=1))[0])
            Xa = X[aid, :-2, 0]
            if np.random.randint(2) == 0:  # +
                X[i, :-2, 0] += Xa
            else:  # -
                XX = np.maximum(X[i, :-2, 0] - Xa, 0)
                if (not np.all(XX == 0)):
                    X[i, :-2, 0] = XX
        sumx = sum(X[i, :nt, 0])
        sf = float(normlen) / sumx
        X[i, :nt, 0] = np.round(np.multiply(X[i, :nt, 0], sf)).astype(int)

    # Compress
    Xs = np.zeros((nbat, ncd + 2, maxd + 1))
    for i in range(nbat):
       Xs[i, :-2, :] = shrink(X[i, :-2, :], nm, nt)
    # Xs[:, :, :nd+1] = X

    # State 3: compute GT
    for i in range(nbat):
        Xs[i, -2:, 0] = 1.0
        sumx = sum(Xs[i, :-2, 0])
        if sumx == 0:
            Xs[i, np.random.randint(0, ncd), 0] = 1
        fid = FIDS[i]

        qid = np.random.randint(len(PL[fid]))
        Xs[i, -2, 1:len(PL[fid][qid]) + 1] = PL[fid][qid]
        Xs[i, -1, 1:len(PH[fid][qid]) + 1] = PH[fid][qid]
        if istest==0 and np.random.randint(2)==0:
            ql, qh = [], []
            q1, q2 = np.random.randint(nt), np.random.randint(nt)
            for d in range(len(nss[fid])):
                ql += [int(q1 % nss[fid][d])]
                q1 = int(q1 / nss[fid][d])
                qh += [int(q2 % nss[fid][d])]
                q2 = int(q2 / nss[fid][d])
            Xs[i, -2, 1:] = np.minimum(ql, qh)
            Xs[i, -1, 1:] = np.maximum(ql, qh)

        if type=='lb':
            idx = np.where(np.all((Xs[i, :-2, 1:] > Xs[i, -2, 1:]) & (Xs[i, :-2, 1:] < Xs[i, -1, 1:]), axis=1))[0]
        else:
            idx = np.where(np.all((Xs[i, :-2, 1:] >= Xs[i, -2, 1:]) & (Xs[i, :-2, 1:] <= Xs[i, -1, 1:]), axis=1))[0]

        if istest == 0 and np.random.randint(2)==0 and sumx - sum(Xs[i, idx, 0]) > 0:
            Xs[i, idx, 0] = 0

        s = sum(Xs[i, idx, 0])
        cnt_X[i] = np.log(max(s, 1))
        for d in range(len(nss[fid])):
            Xs[i, :, d+1] = np.round(Xs[i, :, d+1]*(neb-1)/nss[fid][d])
        Xs[i, :-2, 0] *= 1.0 / sum(Xs[i, :-2, 0])

    Xs[:, :, 1:] = np.maximum(Xs[:, :, 1:], -1) + 1
    return Xs, cnt_X

def generate(X_in, nss, nm, nt, nbat, neb, normlen, nl, nd, maxd, istest, PL=[], PH=[], P=[]):
    while True:
        yield gen_train(X_in, nss, nm, nt, nbat, neb, normlen, nl, nd, maxd, istest, PL, PH, P)

def prep_test(V, ds, nss):
    X = np.zeros(int(np.prod(nss))).astype(int)
    vv = V[:, ds].copy()
    for d in range(1, len(nss)):
        vv[:, 0] = vv[:, 0] + vv[:, d] * prod(nss, d)
    cnt = np.bincount(vv[:,0])
    key = np.nonzero(cnt)[0]
    X[key] += cnt[key]
    return X