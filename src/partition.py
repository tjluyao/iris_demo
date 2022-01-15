import numpy as np
from utils import *
import copy

def bkt_loss(nss, dvs, nb, td):
    loss = 1.0
    for d in range(len(nss)):
        nstd = np.power(nb, nss[d] + int(td==d))
        if(nstd<dvs[d]):
            loss *= 1-1.0/nstd
    return 1.0-loss

#L0 bucketing = all random with max cap+2
def rebkt_L0(cap, total, nb, neb):
    maxs = (int)(np.log(neb)/np.log(nb))
    nss = np.ones(len(cap))
    while np.sum(nss) < total:
        valid = []
        for d in range(len(cap)):
            if(nss[d]+1 <= min(cap[d]+2,maxs)):
                valid += [d]
        if(len(valid)==0):
            return None
        id = valid[np.random.randint(len(valid))]
        nss[id]+=1
    return np.power(nb, nss).astype(int)

#L2 bucketing = ensure no subbucketing for specified columns
def rebkt_L2(cap, total, nb, neb, dvcols, dvs):
    maxs = (int)(np.log(neb)/np.log(nb))
    nss = np.ones(len(cap))
    nss[dvcols] = cap[dvcols]
    if(sum(nss)>total):
        print('Total: ' + str(total))
        print('Require: ')
        print(nss)
        return None
    while np.sum(nss) < total:
        minloss, id, ndv = 1, -1, 1
        for d in range(len(nss)):
            if (nss[d] + 1 <= min(maxs, cap[d] + 2)):
                loss = bkt_loss(nss, dvs, nb, d)
                if (loss < minloss or (loss == minloss and dvs[d] > ndv)):
                    minloss, id, ndv = loss, d, dvs[d]
        if (id == -1):
            maxr = -1
            for d in range(len(cap)):
                r = float(dvs[d]) / np.power(nb, nss[d] + 1)
                if (r > maxr and nss[d] + 1 <= maxs):
                    maxr, id = r, d
        nss[id] += 1
    return np.power(nb, nss).astype(int)

def rebucket(ndims, ids, nt, nb, neb, ntype, dvcols=[]):
    total = (int)(np.round(np.log(nt)/np.log(nb)))
    if(len(ids)>total):
        return None
    cap = np.zeros(len(ids)).astype(int)
    dvs = np.zeros(len(ids)).astype(int)
    for i in range(len(ids)):
        dvs[i] = len(ndims[ids[i]])
        cap[i] = max(1,np.ceil(np.log(dvs[i])/np.log(nb)))
    if(ntype=='Training'):
        return rebkt_L2(cap, total, nb, neb, dvcols, dvs)
    elif(ntype=='Range'):
        return rebkt_L2(cap, total, nb, neb, [], dvs)
    elif(ntype=='Point'):
        return rebkt_L2(cap, total, nb, neb, [], dvs)
    return None

def recover_cnts(keys, cnts, ids):
    acc = []
    for k in range(len(ids)):
        cur = 0
        for n in range(0 if k == 0 else ids[k - 1] + 1, ids[k] + 1):
            cur += cnts[n]
        acc += [cur]
    return [keys[id] for id in ids], acc

def bkt_shrink(ccache, ckey, keys, cnts, types, nss, usecpp=0, dvcol=[]):
    keyss = [copy.copy(key) for key in keys]
    cntss = [copy.copy(cnt) for cnt in cnts]
    if(ckey in ccache):
        idss = ccache[ckey]
        for d in range(len(keyss)):
            keyss[d], cntss[d] = recover_cnts(keyss[d], cntss[d], idss[d])
        return keyss, cntss
    else:
        if usecpp:
            keysfloat = []
            for d in range(len(keyss)):
                if types[d] == "R":
                    keysfloat += [[float(v) for v in keyss[d]]]
                elif types[d] == "D":
                    keys = [parse_date(key) for key in keyss[d]]
                    dd = [(date(key[0], key[1], key[2]) - date(1900, 1, 1)).days for key in keys]
                    keysfloat += [dd]
                else:
                    keysfloat += [[1.0] * len(keyss[d])]
            import shrink_cpp
            idss = shrink_cpp.partition(keysfloat, list([int(s) for s in nss]), list([list(cnt) for cnt in cntss]))
            ccache[ckey] = idss
            for d in range(len(keyss)):
                keyss[d], cntss[d] = recover_cnts(keyss[d], cntss[d], idss[d])
        else:
            idss = []
            for d in range(len(keyss)):
                valtup, valdif = [], []
                ids = []
                for i in range(len(keyss[d])):
                    ids += [i]
                for i in range(0, len(keyss[d]) - 1):
                    vd = dif_str(keyss[d][i + 1], keyss[d][i], types[d])
                    valtup += [(cntss[d][i + 1] + cntss[d][i]) * vd]
                    valdif += [vd]
                while (len(keyss[d]) > nss[d]):
                    mintup = min(valtup)
                    id0 = np.where(np.array(valtup)==mintup)[0]
                    mindif = min(np.array(valdif)[id0])
                    minvds = np.where(np.array(valdif)[id0]==mindif)[0]
                    minvd = id0[minvds[int(len(minvds)/2)]]

                    cntss[d][minvd + 1] = cntss[d][minvd] + cntss[d][minvd + 1]
                    keyss[d].pop(minvd)
                    cntss[d].pop(minvd)
                    ids.pop(minvd)
                    valtup.pop(minvd)
                    valdif.pop(minvd)
                    if (minvd > 0):
                        vd = dif_str(keyss[d][minvd], keyss[d][minvd - 1], types[d])
                        valtup[minvd - 1] = (cntss[d][minvd - 1] + cntss[d][minvd]) * vd
                        valdif[minvd - 1] = vd
                    if (minvd < len(valtup) - 1):
                        vd = dif_str(keyss[d][minvd + 1], keyss[d][minvd], types[d])
                        valtup[minvd] = (cntss[d][minvd] + cntss[d][minvd + 1]) * vd
                        valdif[minvd] = vd
                idss += [ids]
            ccache[ckey] = idss
        return keyss, cntss