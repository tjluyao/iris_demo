import warnings
warnings.filterwarnings('ignore')

import sys
import pickle
from parameters import *
from partition import *
from iris_io import *
from model import *
from build_sparse_hist import *
from build_iris import *

import keras
from keras.models import Model, load_model
import tensorflow as tf

# Ensure only CPU is used in Keras
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

col_name=['ORDERKEY', 'PARTKEY', 'SUPPKEY', 'LINENUMBER', 'QUANTITY', 'EXTENDEDPRICE', 'DISCOUNT', 'TAX', 'RETURNFLAG', 'LINESTATUS', 'SHIPDATE', 'COMMITDATE', 'RECEIPTDATE', 'SHIPINSTRUCT', 'SHIPMODE']

def prep(V, ds, nss):
    X = np.zeros(int(np.prod(nss))).astype(int)
    vv = V[:, ds].copy()
    for d in range(1, len(nss)):
        vv[:, 0] = vv[:, 0] + vv[:, d] * prod(nss, d)
    cnt = np.bincount(vv[:,0])
    key = np.nonzero(cnt)[0]
    X[key] += cnt[key]
    return X

def extract_emb(write_pickle, iris_model=[], ncd=0, nd=0):
    #step 0. extract embedding
    if write_pickle:
        vf = iris_model.get_layer('lambda_6').output
        model_vf = Model(iris_model.layers[0].input, vf)

        emb = {}
        xs = np.zeros((1, ncd+2, nd + 1))
        for id in range(options.neb * options.neb):
            xs.fill(0)
            xs[0, 0, 0] = 1
            xs[0,-2:,0] = 1
            for d in range(nd):
                xs[0, 0, d + 1] = id % options.neb
                id = int(id / options.neb)
            xs[0, -2, :] = xs[0, 0, :]
            xs[0, :, 1:] = np.maximum(xs[0, :, 1:], -1) + 1
            v = model_vf.predict(xs)[0]
            emb[','.join([str(int(s)) for s in xs[0,0,1:]])] = v
        pickle.dump(emb, open('emb.pkl', 'wb'))
    else:
        emb = pickle.load(open('emb.pkl', 'rb'))
    return emb

def build(dic, totald, options, budget, iris_model):
    id = 0
    maxd = 4
    atom = options.atom
    nd = options.maxd
    ncd = options.nt

    #load model
    vf = iris_model.get_layer('lambda_6').output
    model_encoding = Model(iris_model.layers[0].input, vf)

    cols = list(range(totald))
    Fnms = dic['Fnms'][id]
    np.random.seed(0)
    np.random.shuffle(Fnms)

    KEYs, CNTs, TYPEs = [dic['Keys'][id][d] for d in cols], [dic['Cnts'][id][d] for d in cols], [dic['Types'][id][d] for d in cols]
    V, KEYo = {}, {}
    neb = options.neb

    nss = np.array([neb] * len(KEYs)).astype(int)
    print("Computing quantizations..")
    KEYsd, CNTsd = bkt_shrink({}, "", KEYs, CNTs, TYPEs, nss, 'freq', options.nusecpp, [])
    print("Reading data..")
    KEYo[neb] = [parse_keys(KEYsd[id], TYPEs[id]) for id in range(len(KEYsd))]
    Vo = readr({}, {}, Fnms[0:int(options.nlen / 10000)], TYPEs, nss, cols)
    V[neb] = parse_raw(Vo, KEYo[neb], nss)
    neb = int(neb / 2)

    while neb > 1:
        nss = np.array([neb]*len(KEYs)).astype(int)
        KEYsd, CNTsd = bkt_shrink({}, "", KEYsd, CNTsd, TYPEs, nss, 'freq', options.nusecpp, [])
        KEYo[neb] = [parse_keys(KEYsd[id], TYPEs[id]) for id in range(len(KEYsd))]
        V[neb] = parse_raw(Vo, KEYo[neb], nss)
        neb = int(neb/2)

    # step 2. run CORDs
    if options.nusecpp:
        print("Running CORDs..")
        import cords
        colsstr = ','.join([str(c) for c in cols])
        cords.CORDS("data/tpch_example/sample", colsstr)

    print("Building summaries..")
    feats, chs = [], []
    with open(options.cords_fnm, 'r') as fcords:
        cnt1 = int(fcords.readline())
        for i in range(cnt1):
            ln = fcords.readline().split('\t')
            chs += [[[int(ln[0])], int(ln[1]), 1, ln[2].rstrip('\n'), [], []]]
        cnt2 = int(fcords.readline())
        for i in range(cnt2):
            ln = fcords.readline().split('\t')
            chs += [[[int(ln[0]), int(ln[1])], int(ln[2]), float(ln[3]), ln[4].rstrip('\n'), [], []]]

    total_size = 0

    for ch in chs:
        chcol = np.array([cols.index(c) for c in ch[0]])

        if total_size + options.atom/256 <= budget and ch[3] == 'AVI' and len(ch[0])>1:
            ch[3] = 'Hist'
        elif total_size + options.atom/256 > budget:
            ch[3] = 'AVI'
            feats += [[]]
            size = 0
            continue

        #Prepare queries
        sl = max(128, options.atom) if ch[3]=='AVI' else (options.nt if ch[3] == 'Iris' else options.atom)
        ns = rebucket(np.array(dic['Cnts'][id]), chcol, sl, options.nb, options.neb, 'Range', 'freq')
        ns = np.minimum(ns, options.neb)
        ns, chcol = map(list, zip(*sorted(zip(ns, chcol), reverse=True)))
        ch[-2], ch[-1] = ns, chcol
        Vr = [[] for _ in range(len(ns))]
        for i in range(len(ns)):
            Vr[i] += [l[chcol[i]] for l in V[int(ns[i])]]
        X = prep(np.array(Vr).transpose(), np.arange(len(ns)), ns)

        if ch[3] == 'Sparse':
            xx = X.reshape(np.flip(ns)).transpose().reshape(int(np.prod(ns)))
            ids = np.nonzero(xx)[0]
            _, ids = map(list, zip(*sorted(zip(xx[ids], ids), reverse=True)))
            feat = []
            size = 0
            for id in ids:
                f = [xx[id]]
                for i in range(len(chcol)):
                    f.insert(0, int(id % ns[-(i+1)]))
                    id =  int(id / ns[-(i+1)])
                feat += [f]
                size += len(f)
                if size >= atom:
                    break
            feat = np.array(feat)
            size = np.prod(feat.shape)
            if total_size + size/256 > budget:
                size = 0
                ch[3] = 'AVI'
                feat = []
            feats += [feat]
        elif ch[3] == 'Hist':
            feat = X.reshape(np.flip(ns)).transpose()
            size = np.prod(feat.shape)
            if total_size + size/256 > budget:
                size = 0
                ch[3] = 'AVI'
                feat = []
            feats += [feat]
        elif ch[3] == 'AVI' and len(ch[0]) > 1:
            feats += [[]]
            size = 0
        elif ch[3] == 'Key' and len(ch[0]) == 1:
            feats += [[]]
            size = 0
        #Base histograms
        elif ch[3] == 'AVI' and len(ch[0]) == 1:
            feat = X.reshape(ns).transpose()
            feats += [feat]
            size = np.prod(feat.shape)
        else: #Iris
            xx = gen(ncd, nd, X, ns, options.normlen, options.nt, 1, options.neb, maxd)
            feat = model_encoding.predict(xx.reshape(1, ncd+2, maxd+1))[0]
            size = np.prod(feat.shape) + sum(ch[4])
            if total_size + size/256 > budget:
                size = 0
                ch[3] = 'AVI'
                feat = []
            feats += [feat]
        total_size += size * 4 / 1024
        if len(ch[0])>1:
            print('\tSummary for columnset ' + (col_name[ch[0][0]]+','+col_name[ch[0][1]]).ljust(25) + '\tw/ #DV ' + str(ch[1]) + ',\tcorr. score ' + str(ch[2]) + '\tbuilt using ' + ch[3])
    print('Storage budget: ' + str(options.storage) + 'KB, total used size: ' + str(total_size) + 'KB')
    print('------------------------------')

    dic_feat = {}
    dic_feat['KEYo'] = KEYo
    dic_feat['feat'] = feats
    dic_feat['cols'] = cols
    dic_feat['TYPEs'] = TYPEs
    dic_feat['ch'] = chs
    pickle.dump(dic_feat, open('feature.pkl', 'wb'))

def infer(dic, weights, options):
    tid = 0
    f = pickle.load(open('feature.pkl', 'rb'))
    KEYo = f['KEYo']
    TYPEs = f['TYPEs']
    feats = f['feat']
    chs = f['ch']
    nqs = len(dic['DVcol'][tid])
    gmq, gmq_ulow, gmq_low, gmq_med, gmq_high = [], [], [], [], []
    model_iris_2d = get_query_model(options.nr, options.nfc, options.nn)
    ids = ['2f', '51', '52', '6']
    cid = 0
    for i in range(len(model_iris_2d.layers)):
        if isinstance(model_iris_2d.layers[i], keras.layers.Dense):
            model_iris_2d.layers[i].set_weights(weights['dense' + ids[cid]])
            cid += 1

    emb = extract_emb(False)
    for i in range(len(feats)):
        if chs[i][3] == 'Hist' or (chs[i][3] == 'AVI' and len(chs[i][0]) == 1):
            feats[i] = warm_up_hist(feats[i], len(chs[i][0]))

    for qi in range(nqs):
        PL = dic['Pred_low'][tid][qi]
        PH = dic['Pred_high'][tid][qi]
        dvcol = dic['DVcol'][tid][qi]
        GT = dic['GT_freq'][tid][qi]

        pred_low_r = PL
        pred_high_r = PH
        for d in range(len(PL)):
            if (TYPEs[dvcol[d]] == 'R'):
                pred_low_r[d] = float(pred_low_r[d])
                pred_high_r[d] = float(pred_high_r[d])
            elif (TYPEs[dvcol[d]] == 'D'):
                dl, dh = parse_date(pred_low_r[d]), parse_date(pred_high_r[d])
                pred_low_r[d] = (date(dl[0], dl[1], dl[2]) - date(1900, 1, 1)).days
                pred_high_r[d] = (date(dh[0], dh[1], dh[2]) - date(1900, 1, 1)).days

        #compute marginals - will remove in future
        margins = []
        types = []
        for i in range(len(dvcol)):
            id = dvcol[i]
            type = ''
            bid = -1
            for j in range(len(chs)):
                if id in chs[j][0] and len(chs[j][0]) == 1 and ((chs[j][3] == 'AVI') or (chs[j][3] == 'Key')):
                    bid = j
                    type = chs[j][3]
                elif id in chs[j][0] and chs[j][3] == 'Sparse':
                    bid = j
                    type = chs[j][3]
                elif id in chs[j][0] and chs[j][3] == 'Hist' and type == '':
                    bid = j
                    type = chs[j][3]
            if(bid > -1):
                if type == 'Key':
                    PLc, PHc, _ = readp([pred_low_r[i]], [pred_high_r[i]], [KEYo[options.neb][id]], [], [options.neb])
                    if pred_low_r[i] == pred_high_r[i]:
                        margins += [1.0 / dic['Rows'][tid]]
                    else:
                        margins += [(PHc[0] - PLc[0] + 1) / len(KEYo[options.neb][id])]
                else:
                    if type == 'AVI':
                        PLc, PHc, _ = readp([pred_low_r[i]], [pred_high_r[i]], [KEYo[options.neb][id]], [], [options.neb])
                        PLm, PHm = [PLc[0]], [PHc[0]]
                        dims = np.array([0])
                    else: #sparse
                        PLm, PHm = [], []
                        for t in range(len(chs[bid][0])):
                           if id != chs[bid][0][t]:
                               PLm += [0]
                               PHm += [options.neb]
                           else:
                               cns = chs[bid][-2][chs[bid][-1].index(id)]
                               PLc, PHc, _ = readp([pred_low_r[i]], [pred_high_r[i]], [KEYo[cns][id]], [], [cns])
                               PLm += [PLc[0]]
                               PHm += [PHc[0]]
                        dims = np.arange(len(chs[bid][0]))
                    if chs[bid][0][0] != chs[bid][-1][0]:
                        PLm.reverse()
                        PHm.reverse()
                    if type == 'Sparse':
                        margins += [infer_sparse_hist(feats[bid], dims, PLm, PHm, True) / options.nlen]
                    elif type == 'Hist' or type == 'AVI':
                        margins += [infer_sparse_hist(feats[bid], dims, PLm, PHm, False) / options.nlen]
                    else:
                        print('Inpossible path')
                        sys.exit()
                types += [type]

        dvs = dvcol.copy()
        chsn = [[[dvs[id]], 0, 1, types[id], margins[id]] for id in range(len(dvs))]
        for i in range(len(chs)):
            ns = chs[i][-2]
            cols = chs[i][-1]

            if len(chs[i][0]) > 1 and chs[i][3] != 'AVI' and set(chs[i][0]).issubset(set(dvs)):
                ky, pl, ph = [], [], []
                for d in range(len(cols)):
                    pl += [pred_low_r[dvs.index(cols[d])]]
                    ph += [pred_high_r[dvs.index(cols[d])]]
                    ky += [KEYo[ns[d]][cols[d]]]
                PLm, PHm, _ = readp(pl, ph, ky, [], ns)

                dims = np.arange(len(chs[i][0]))
                if chs[i][3] == 'Sparse':
                    card = infer_sparse_hist(feats[i], dims, PLm, PHm, True) / options.nlen
                elif chs[i][3] == 'Iris':
                    pll = [PLm[0] - 1, PLm[1] - 1]
                    phh = [PHm[0], PHm[1]]
                    plh = [PLm[0] - 1, PHm[1]]
                    phl = [PHm[0], PLm[1] - 1]
                    for d in range(len(cols)):
                       pll[d] = int(round(pll[d] * (options.neb - 1) / (ns[d] - 1)))
                       phh[d] = int(round(phh[d] * (options.neb - 1) / (ns[d] - 1)))
                       plh[d] = int(round(plh[d] * (options.neb - 1) / (ns[d] - 1)))
                       phl[d] = int(round(phl[d] * (options.neb - 1) / (ns[d] - 1)))
                    pll = np.maximum(pll, -1) + 1
                    phh = np.maximum(phh, -1) + 1
                    plh = np.maximum(plh, -1) + 1
                    phl = np.maximum(phl, -1) + 1
                    emb_ll = str(pll[0]) +',' + str(pll[1]) + ',1,1'
                    emb_hh = str(phh[0]) +',' + str(phh[1]) + ',1,1'
                    emb_lh = str(plh[0]) +',' + str(plh[1]) + ',1,1'
                    emb_hl = str(phl[0]) +',' + str(phl[1]) + ',1,1'
                    card_ll = 0 if np.any(np.array(pll) <= 0) else np.exp(model_iris_2d.predict([feats[i].reshape(1,1,options.nr), emb[emb_ll].reshape(1,1,options.nfc)])[0][0][0])
                    card_hh = np.exp(model_iris_2d.predict([feats[i].reshape(1,1,options.nr), emb[emb_hh].reshape(1,1,options.nfc)])[0][0][0])
                    card_lh = 0 if np.any(np.array(plh) <= 0) else np.exp(model_iris_2d.predict([feats[i].reshape(1,1,options.nr), emb[emb_lh].reshape(1,1,options.nfc)])[0][0][0])
                    card_hl = 0 if np.any(np.array(phl) <= 0) else np.exp(model_iris_2d.predict([feats[i].reshape(1,1,options.nr), emb[emb_hl].reshape(1,1,options.nfc)])[0][0][0])
                    card = (card_hh - card_lh - card_hl + card_ll) / options.normlen
                    card = max(0, min(1, card))
                else:
                    card = infer_sparse_hist(feats[i], dims, PLm, PHm, False) / options.nlen
                chsn += [np.hstack([chs[i], card])]
        chsn = sorted(chsn, key=lambda item: (item[-1], -len(item[0])))

        #don't use Iris for ultra low DV prediction
        for ni in range(len(chsn)):
            if chsn[ni][3] == 'Iris' and chsn[ni][-1] < 0.005:
                chsn[ni][-1] = 1

        #validate and prune based on marginals
        for ni in range(len(chsn)):
            vid = []
            for nj in range(ni + 1, len(chsn)):
                if len(set(chsn[ni][0]).intersection(set(chsn[nj][0]))) == 1 and len(chsn[ni][0])==1 and len(chsn[nj][0])==2 and chsn[ni][-1] < chsn[nj][-1]:
                    chsn[nj][-1] = chsn[ni][-1]
                    vid += [ni]
            for v in vid:
                chsn[v][-1] = 1
        chsn = sorted(chsn, key=lambda item: (item[-1], -len(item[0])))

        #sparse goes first
        chsn0 = []
        for ch in chsn:
            if ch[3] == 'Sparse':
                chsn0 += [ch]
        for ch in chsn:
            if ch[3] == 'Hist':
                chsn0 += [ch]
        for ch in chsn:
            if (ch[3] != 'Hist') and (ch[3] != 'Sparse'):
                chsn0 += [ch]
        chsn = chsn0

        #compute card
        card_fin = 1
        while len(chsn) > 0:
            card_fin = card_fin * chsn[0][-1]
            clique = chsn[0][0].copy()
            chsn0 = []
            for i in range(len(chsn) - 1):
                if len(set(chsn[i + 1][0]).intersection(set(chsn[0][0]))) == 0 and len(set(chsn[i + 1][0]).intersection(set(clique))) == 0 and chsn[i + 1][-1] < 0.98:
                    chsn0 += [chsn[i + 1]]
            chsn = chsn0
        GT = max(GT, 10)
        pred = max(card_fin * dic['Rows'][tid], 10)
        q = max(GT/pred, pred/GT)
        gmq += [np.log(q)]

    print(str(len(gmq)) + ' queries evaluated.')
    print('\tIris\t\tGMQ:' + str(round(np.exp(np.average(gmq)),2)) + ', 95th:' + str(round(np.exp(np.percentile(gmq, 95)),2)))
    if options.storage==60:
        print('\tFor comparison, the following baseline results are pre-computed.')
        print('\tSampling\tGMQ:3.02, 95th:109.98\n\txAVI\t\tGMQ:2.94, 95th:21.00\n\tLM-\t\tGMQ:2.21, 95th:8.08\n\tMSCN\t\tGMQ:3.62, 95th:52.0')

if __name__ == '__main__':
    options = parse_arg()

    #Read LineItem dataset with 5% rows (unused have been deleted to save demo space)
    print("Read base per-column histograms..")
    dic = readd('data/demo_query.txt', 2, 1, 1, sample_rate=0.05, nqs=1000)

    print("Extracting embedding weights from pre-trained model..")
    weights = {}
    iris_full_model = load_model('demo_model').layers[-2]
    weights['Embedding'] = iris_full_model.layers[3].get_weights()
    for i in range(len(iris_full_model.layers)):
        if isinstance(iris_full_model.layers[i], keras.layers.Dense):
            weights[iris_full_model.layers[i].name] = iris_full_model.layers[i].get_weights()
    extract_emb(True, iris_full_model, options.nt, 4)

    build(dic, len(dic['DimR'][0]), options, options.storage, iris_full_model)

    print("Computing cardinality estimates..")
    infer(dic, weights, options)