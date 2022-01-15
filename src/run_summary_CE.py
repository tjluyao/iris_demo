import warnings
warnings.filterwarnings('ignore')

from parameters import *
from build_iris import *
from build_sparse_hist import *
from preproc import *

# Ensure only CPU is used in Keras
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

def build(dic, totald, options, budget, iris_model):
    fid = 0
    nd = options.maxd

    #load model
    vf = iris_model.get_layer('lambda_6').output
    model_encoding = Model(iris_model.layers[0].input, vf)

    #step 1. read data
    cols = list(range(totald))
    Fnms = dic['Fnms'][fid]
    np.random.seed(0)
    np.random.shuffle(Fnms)

    KEYs, CNTs, TYPEs = [dic['Keys'][fid][d] for d in cols], [dic['Cnts'][fid][d] for d in cols], [dic['Types'][fid][d] for d in cols]
    V, KEYo = {}, {}
    neb = options.neb
    sample = reads(dic['Samples'], TYPEs, cols, options.sample_size)
    
    nss = np.array([neb]*len(KEYs)).astype(int)
    if len(options.nusebucket)==0:
        print("Computing quantizations..")
        KEYsd, CNTsd = bkt_shrink({}, "", KEYs, CNTs, TYPEs, nss, options.nusecpp, [])
        KEYo[neb] = [parse_keys(KEYsd[id], TYPEs[id]) for id in range(len(KEYsd))]
    else:
        print("Loading quantizations..")
        fb = pickle.load(open(options.nusebucket, 'rb'))
        KEYo = fb['KEYo']
    print("Reading data..")
    Vo = readr({}, {}, Fnms[0:int(options.nlen / 10000)], TYPEs, nss, cols)
    V[neb] = parse_raw(Vo, KEYo[neb], nss)
    neb = int(neb/2)
    
    while neb > 1:
        nss = np.array([neb]*len(KEYs)).astype(int)
        KEYsd, CNTsd = bkt_shrink({}, "", KEYsd, CNTsd, TYPEs, nss, options.nusecpp, [])
        KEYo[neb] = [parse_keys(KEYsd[id], TYPEs[id]) for id in range(len(KEYsd))]
        V[neb] = parse_raw(Vo, KEYo[neb], nss)
        neb = int(neb/2)
        
    # step 2. run CORDs
    if options.run_cords:
        print("Running CORDs..")
        import cords
        colsstr = ','.join([str(c) for c in cols])
        cords.CORDS(dic['Samples'][fid], options.cords_fnm, colsstr)

    print("Building summaries..")
    chs = []
    with open(options.cords_fnm, 'r') as fcords:
        cnt1 = int(fcords.readline())
        for i in range(cnt1):
            ln = fcords.readline().split('\t')
            chs += [[[int(ln[0])], int(ln[1]), 1, ln[2].rstrip('\n'), [], []]]
        cnt2 = int(fcords.readline())
        for i in range(cnt2):
            ln = fcords.readline().split('\t')
            chs += [[[int(ln[0]), int(ln[1])], int(ln[2]), float(ln[3]), ln[4].rstrip('\n'), [int(ln[5]), int(ln[6])], [], []]]
        cnt3 = int(fcords.readline())
        for i in range(cnt3):
            ln = fcords.readline().split('\t')
            chs += [[[int(ln[0]), int(ln[1]), int(ln[2])], int(ln[3]), float(ln[4]), ln[5].rstrip('\n'), [], []]]
    feats = [[] for _ in range(len(chs))]
    total_size = 0
    cnt_base = 0
    cnt_sparse, cnt_iris, cnt_hist = 0, 0, 0

    for ch in chs:
        if ch[3] == 'AVI' and len(ch[0]) == 1:
            cnt_base += 1
    if cnt_base * options.neb / 256 > budget:
        print("Too small budget, can't build.")
        sys.exit()

    for ps in range(2):
        for cid, ch in enumerate(chs):
            chcol = np.array([cols.index(c) for c in ch[0]])

            # heuristics for model selection
            # each distinct tuple take 1byte/col + 4bytes for the counter
            sparseth = options.max_atom_budget * 4 / (len(ch[0]) + 4)
            # trivial base histograms
            isbase = (len(ch[0]) == 1)
            # picked out by cords + those depending on sotrage budget.
            # cords uses a small sample so that #dv can be underestimated. use 1.5 to compensate
            issparse = ('Sparse' in ch[3]) or ((ch[3] == '-') and (ch[1] * 1.5 <= sparseth))
            # picked out by cords with low corr score
            isavi = ('AVI' in ch[3]) and (not isbase)
            # distinct tuple count and corr score are moderate
            ishist = ('Hist' in ch[3]) or ((ch[3] == '-') and (not issparse) and (ch[2] <= 0.05))
            # all other cases
            isiris = ('Iris' in ch[3]) or ((ch[3] == '-') and (not issparse) and (not ishist))

            if ps == 1 and (ishist or isiris) and (total_size + options.max_atom_budget / 256 > budget):
                size = 0
                ch[3] = 'AVI'
                continue

            if (ps == 0 and (isiris or ishist or isavi)) or (ps == 1 and (isavi or issparse or isbase)):
                continue

            #Base histogram
            if isiris:
                sl = options.nt
            elif issparse:
                sl = options.neb*options.neb
            else:
                sl = max(128, options.max_atom_budget)
            ns = rebucket(np.array(dic['Cnts'][fid]), chcol, sl, options.nb, options.neb, 'Range', 'freq')
            ns = np.minimum(ns, options.neb)
            ns, chcol = map(list, zip(*sorted(zip(ns, chcol), reverse=True)))
            ch[-2], ch[-1] = ns, chcol
            if ch[3] != 'Key':
                Vr = [[] for _ in range(len(ns))]
                for i in range(len(ns)):
                    Vr[i] += [l[chcol[i]] for l in V[int(ns[i])]]
                X = prep_test(np.array(Vr).transpose(), np.arange(len(ns)), ns)

            if ps == 0: # build base histogram and sparse first (ps=0), and then hist and iris (ps=1)
                if ch[3] == 'AVI' and len(ch[0]) > 1:
                    size = 0
                elif ch[3] == 'Key' and len(ch[0]) == 1:
                    size = 0
                elif ch[3] == 'AVI' and len(ch[0]) == 1:  #base histogram
                    feat = X.reshape(ns).transpose()
                    feats[cid] = feat
                    size = np.prod(feat.shape)
                elif issparse: # todo: use a heavy hitter sketch to speedup
                    ch[3] = 'Sparse'
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
                        size += (len(ch[0])+4)/4
                        if size >= options.max_atom_budget:
                            break
                    feat = np.array(feat)
                    size = min(options.max_atom_budget*4/6, max(len(feat), ch[1]))*(len(ch[0])+4)/4
                    if total_size + size/256 > budget:
                        size = 0
                        ch[3] = 'AVI'
                        feat = []
                    else:
                        cnt_sparse += 1
                    feats[cid] = feat
                else:
                    size = 0
            else:
                if ishist:
                    ch[3] = 'Hist'
                    feat = X.reshape(np.flip(ns)).transpose()
                    size = np.prod(feat.shape)
                    cnt_hist += 1
                    feats[cid] = feat
                elif isiris:
                    ch[3] = 'Iris'
                    xx = gen_test(options.ncd, nd, X, ns, options.normlen, options.nt, options.nm, options.neb, options.maxd)
                    feat = model_encoding.predict(xx.reshape(1, options.ncd+2, options.maxd+1))[0]
                    size = np.prod(feat.shape)
                    cnt_iris += 1
                    feats[cid] = feat
            total_size += size * 4 / 1024
            if len(ch[0])>1 and options.isdemo:
                print('\tSummary for columnset ' + (options.col_name[ch[0][0]]+','+options.col_name[ch[0][1]]).ljust(25) + '\tw/ #DV ' + str(ch[1]) + ',\tcorr. score ' + str(ch[2]) + '\tbuilt using ' + ch[3])
    print('Storage budget: ' + str(options.storage) + 'KB, total used size: ' + str(total_size) + 'KB')
    print('Base ' + str(cnt_base) + ', Sparse ' + str(cnt_sparse) + ', Hist '  + str(cnt_hist) + ', Iris ' + str(cnt_iris))
    print('------------------------------')

    dic_feat = {}
    dic_feat['sample'] = sample
    dic_feat['feat'] = feats
    dic_feat['TYPEs'] = TYPEs
    dic_feat['ch'] = chs
    pickle.dump(dic_feat, open('tmp/feature-' + os.path.splitext(os.path.basename(options.input_fnm))[0] + '-' + str(options.storage) + '.pkl', 'wb'))

    dic_bucket = {}
    dic_bucket['KEYo'] = KEYo
    pickle.dump(dic_bucket, open('tmp/bucket-' + os.path.splitext(os.path.basename(options.input_fnm))[0] + '-' + str(options.storage) + '.pkl' if len(options.nusebucket)==0 else options.nusebucket, 'wb'))

def infer(dic, weights, options):
    tid = 0
    ff = pickle.load(open('tmp/feature-' + os.path.splitext(os.path.basename(options.input_fnm))[0] + '-' + str(options.storage) + '.pkl', 'rb'))
    fb = pickle.load(open('tmp/bucket-' + os.path.splitext(os.path.basename(options.input_fnm))[0] + '-' + str(options.storage) + '.pkl' if len(options.nusebucket)==0 else options.nusebucket, 'rb'))
    feats = ff['feat']
    chs = ff['ch']
    sample = ff['sample']
    TYPEs = ff['TYPEs']
    KEYo = fb['KEYo']

    nqs = len(dic['DVcol'][tid])
    gmq, gmq_low, gmq_med, gmq_high = [], [], [], []
    model_iris_2d = get_query_model(options.nr, options.nfc, options.nn)
    ids = ['2f', '51', '52', '6']
    for i in range(len(model_iris_2d.layers)):
        if isinstance(model_iris_2d.layers[i], keras.layers.Dense):
            model_iris_2d.layers[i].set_weights(weights[model_iris_2d.layers[i].name])
    #print(chs)

    emb = pickle.load(open('tmp/emb-' + os.path.splitext(os.path.basename(options.model_fnm))[0] + '.pkl', 'rb'))
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
                    PLc, PHc, _ = readp([pred_low_r[i]], [pred_high_r[i]], [KEYo[options.neb][id]], [options.neb])
                    if pred_low_r[i] == pred_high_r[i]:
                        margins += [1.0 / dic['Rows'][tid]]
                    else:
                        margins += [(PHc[0] - PLc[0] + 1) / len(KEYo[options.neb][id])]
                else:
                    if type == 'AVI':
                        PLc, PHc, _ = readp([pred_low_r[i]], [pred_high_r[i]], [KEYo[options.neb][id]], [options.neb])
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
                               PLc, PHc, _ = readp([pred_low_r[i]], [pred_high_r[i]], [KEYo[cns][id]], [cns])
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
        #chsn = [[[dvs[id]], 0, 1, types[id], margins[id]] for id in range(len(dvs))]
        chsn = [[[dvs[id]], 0, 1, 'AVI', margins[id]] for id in range(len(dvs))]

        for i in range(len(chs)):
            ns = chs[i][-2]
            cols = chs[i][-1]

            if len(chs[i][0]) > 1 and chs[i][3] != 'AVI' and set(chs[i][0]).issubset(set(dvs)):
                ky, pl, ph = [], [], []
                for d in range(len(cols)):
                    pl += [pred_low_r[dvs.index(cols[d])]]
                    ph += [pred_high_r[dvs.index(cols[d])]]
                    ky += [KEYo[ns[d]][cols[d]]]
                PLm, PHm, _ = readp(pl, ph, ky, ns)

                dims = np.arange(len(chs[i][0]))
                if chs[i][3] == 'Sparse':
                    card = infer_sparse_hist(feats[i], dims, PLm, PHm, True) / options.nlen
                elif chs[i][3] == 'Iris':
                    #Compute lower bound from small sample. TODO: C++ impl.
                    lb, nsmp = 0, len(sample[0])
                    for r in range(nsmp):
                        flag = True
                        for d in range(len(cols)):
                            if TYPEs[cols[d]] == 'C':
                                if sample[cols[d]][r] != pl[d]:
                                    flag = False
                                    break
                            else:
                                if sample[cols[d]][r] < pl[d] or sample[cols[d]][r] > ph[d]:
                                    flag = False
                                    break
                        if flag:
                            lb += 1/nsmp

                    #Compute upper bound from base histograms
                    ub = np.min([margins[dvs.index(d)] for d in cols])

                    #Compute Iris predictions
                    for d in range(len(cols)):
                        PLm[d] = int(round(PLm[d] * (options.neb - 1) / (ns[d] - 1)))
                        PHm[d] = int(round(PHm[d] * (options.neb - 1) / (ns[d] - 1)))
                    PLm, PHm = np.maximum(PLm, -1) + 1, np.maximum(PHm, -1) + 1

                    pred_emb = np.hstack([emb[str(PLm[0]) + ',' + str(PLm[1])], emb[str(PHm[0]) + ',' + str(PHm[1])]])

                    card = 0 if np.any(np.array(PLm) <= 0) or np.any(np.array(PHm) <= 0) else np.exp(
                        model_iris_2d.predict([feats[i].reshape(1, 1, options.nr), pred_emb.reshape(1, 1, 2*options.nr)])[0][0][0])
                    card = max(0, min(1, card/options.normlen))
                    card = min(ub, max(card, lb))
                else:
                    card = infer_sparse_hist(feats[i], dims, PLm, PHm, False) / options.nlen
                chsn += [np.hstack([chs[i], card])]
        chsn = sorted(chsn, key=lambda item: (item[-1], -len(item[0])))

        #validate and prune search space based on marginals
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
        sp, hist, iris = 0, 0, 0
        for ch in chsn:
            if ch[3] == 'Sparse' or ch[3] ==  'Hist':
                chsn0 += [ch]
                if ch[3] == 'Sparse':
                    sp += 1
                else:
                    hist += 1
        for ch in chsn:
            if (ch[3] != 'Hist') and (ch[3] != 'Sparse'):
                chsn0 += [ch]
        chsn = chsn0

        #compute card
        card_fin = 1
        while len(chsn) > 0:
            card_fin = card_fin * chsn[0][-1]
            if chsn[0][3] == 'Iris' and iris == 0:
                for ch in chsn:
                    if ch[3] == 'Iris':
                        iris+=1
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
        if GT/dic['Rows'][tid] <= 0.005:
            gmq_low += [np.log(q)]
        elif GT/dic['Rows'][tid] <= 0.02:
            gmq_med += [np.log(q)]
        else:
            gmq_high += [np.log(q)]
        #print(pred, GT, q)

    print(options.cords_fnm + ' Evaluated ' + str(len(gmq)) + ' queries')
    print('GMQ:' + str(np.exp(np.average(gmq))))
    print('95th:' + str(np.exp(np.percentile(gmq, 95))))
    if len(gmq_low)>0:
        print('GMQ low:' + str(np.exp(np.average(gmq_low))))
        print('95th low:' + str(np.exp(np.percentile(gmq_low, 95))))
    if len(gmq_med)>0:
        print('GMQ med:' + str(np.exp(np.average(gmq_med))))
        print('95th med:' + str(np.exp(np.percentile(gmq_med, 95))))
    if len(gmq_high)>0:
        print('GMQ high:' + str(np.exp(np.average(gmq_high))))
        print('95th high:' + str(np.exp(np.percentile(gmq_high, 95))))
    if options.isdemo:
        print(options.demoresult)

if __name__ == '__main__':
    options = parse_arg()
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    dic = readd(options.data_dir + '/' + options.input_fnm, sample_rate=options.input_rate, nqs=1000)
    options.nlen = options.input_rate * dic['Rows'][0]

    # Turn storage from how many X of that is used by a production system to actual KBs
    # overhead: options.neb(xi) B/col bucket boundaries, options.sample_size*4/1024 KB/col small sample
    # and 0.5KB base histogram (options.neb bins, counted already during construction)
    options.max_atom_budget *= options.storage
    if options.storage < 0.5:
        options.sample_size *= 0.5
    options.storage = len(dic['Types'][0]) * (options.storage * 4 - options.neb/1024 - options.sample_size/256)
    print('Storage budget ' + str(options.storage) + 'KB, max atom budget ' + str(4*options.max_atom_budget/1024) + 'KB, sample size ' + str(options.sample_size) + ' rows')

    print("Extracting embedding weights from pre-trained model (can be cached offline)..")
    weights = {}
    iris_full_model = load_model(options.model_fnm).layers[-2]
    weights['Embedding'] = iris_full_model.layers[3].get_weights()
    for i in range(len(iris_full_model.layers)):
        if isinstance(iris_full_model.layers[i], keras.layers.Dense):
            weights[iris_full_model.layers[i].name] = iris_full_model.layers[i].get_weights()

    if options.extract_emb:
        extract_emb(iris_full_model, options.model_fnm, options.neb, options.ncd)

    build(dic, len(dic['DimR'][0]), options, options.storage, iris_full_model)
    
    print("Computing cardinality estimates..")
    infer(dic, weights, options)
