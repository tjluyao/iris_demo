import sys
from utils import *

np.set_printoptions(threshold=sys.maxsize)

from keras.utils import multi_gpu_model
from keras.optimizers import Adam

from parameters import *
from preproc import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
options = parse_arg(istest=0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(s) for s in range(options.ngpus)])


dic_training = readd(options.data_dir + '/train/training_histiris_q50.txt', istest=0, nqs=300)
nqs = 300
dic_val = readd(options.data_dir + '/test/test_16relationsb128.txt', istest=2, nqs=nqs)
print('Training using ' + str(len(dic_training['Table']))+ ' w/ augmentation multiplier ' + str(options.nl) + '.')
print('Validating using ' + str(len(dic_val['Table'])) + ' relations.')

path = options.output_dir + '/' + options.model_fnm
if not os.path.exists(path):
    os.mkdir(path)

print('nt (\ell)=' + str(options.nt) + ', nd (max dim)=' + str(options.maxd) + ', nr (\eta)=' + str(options.nr) + ', output to ' + path)

with tf.device('/cpu:0'):
    model = get_card_model(options.nr, options.ncd, options.maxd, options.neb, options.ne, options.nfc, options.nn)

ada = Adam(options.nlr)
if options.ngpus>1:
    model = multi_gpu_model(model, gpus=options.ngpus)
print('Using ' + str(options.ngpus) + ' GPUs')
model.compile(optimizer=ada, loss='mae')

print(model.summary() if options.ngpus == 1 else model.layers[-2].summary())

icache, bcache, ccache = {}, {}, {}
istest = 0
nl = max(1, int(options.nl * len(dic_training['Table'])))
R, B, PL, PH = prep_train(nl, nqs, options.nt, options.nb, options.neb, options.nlen, dic_training, istest, options.nusecpp, icache, bcache, ccache)
g = generate(R, B, options.nm, options.nt, options.nbat, options.neb, options.normlen, nl, options.mind, options.maxd, istest, PL, PH)

istest = 1
nl = int(2 * nqs * len(dic_val['Table']))
nlen = 5000000
options.nbat = nqs
Rv, Bv, PLv, PHv = prep_train(nl, nqs, options.nt, options.nb, options.neb, nlen, dic_val, istest, options.nusecpp, icache, bcache, ccache)
gv = generate(Rv, Bv, options.nm, options.nt, options.nbat, options.neb, options.normlen, nl, options.mind, options.maxd, istest, PLv, PHv)

nworkers = 4 * options.ngpus
queuesize = nworkers+2

checkpointer = [WarmUpLearningRateScheduler(options.nst, options.nlr/10.0),
                ModelCheckpoint(path + '/' + options.model_fnm + '_{epoch:02d}')]

model.fit_generator(g, steps_per_epoch=options.nst, epochs=options.nep, validation_data=gv, validation_steps=(nl / options.nbat), verbose=options.nverbose,
                    use_multiprocessing=True, max_queue_size=queuesize, shuffle=True, workers=nworkers, callbacks=checkpointer)
