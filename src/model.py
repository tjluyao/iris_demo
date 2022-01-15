import keras.backend as K
from keras.layers import Input, Dense, Embedding, Lambda, concatenate, multiply, Conv1D, Reshape, BatchNormalization, Dropout, LeakyReLU, RepeatVector, multiply, Activation, MaxPooling1D, Add
from keras.models import Model, load_model
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

class WarmUpLearningRateScheduler(Callback):
    def __init__(self, warmup_batches, init_lr, verbose=0):
        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))


def get_card_model(nr, ncd, nd, neb, ne, nfc = 128, nn=3):
    ndr = 0.3
    i = Input(shape=(ncd + 2, nd + 1,))
    xv = Lambda(lambda x: K.repeat_elements(K.expand_dims(x[:, :, 0]), nr, axis=2))(i)

    x = Lambda(lambda x: x[:, :, 1:])(i)
    print(x.shape, ncd, nd)

    x = Reshape(((ncd + 2) * nd, 1))(x)
    x = Embedding(neb + 1 + 1 + 1 + 1, ne, mask_zero=False, name='emb1')(x)
    x = Lambda(lambda x: x, output_shape=lambda s: s)(x)
    x = Reshape(((ncd + 2), nd * ne))(x)
    for n in range(nn):
        x = Dense(nfc, use_bias=True, name='dense1' + str(n))(x)
        x = Activation('elu')(x)
        x = Dropout(ndr)(x)
    x = Dense(nr, name='dense1f')(x)
    x = Activation('elu')(x)
    x = Dropout(ndr)(x)

    xk = Lambda(lambda x: x[:, :-2, :])(x)
    xf = Lambda(lambda x: x[:, :-2, :])(xv)
    xk = multiply([xk, xf])

    e = Lambda(lambda x: K.sum(x[:, :-2], axis=1), output_shape=(lambda shape: (shape[0], shape[2])))(xk)

    for n in range(nn):
        e = Dense(nfc, use_bias=True, name='dense2' + str(n))(e)
        e = Activation('elu')(e)
        e = Dropout(ndr)(e)
    e = Dense(nfc, use_bias=True, name='dense2f')(e)
    e = Activation('elu')(e)
    e = Dropout(ndr)(e)

    qr = Lambda(lambda x: x[:, -2:, :])(x)

    qr = Reshape((1, 2*nr))(qr)
    qr = Dense(nfc, use_bias=True, name='denseq')(qr)
    qr = Activation('elu')(qr)
    qr = Dropout(ndr)(qr)
    qr = Lambda(lambda x: x[:, 0, :], output_shape=(lambda shape: (shape[0], shape[2])))(qr)

    cr = multiply([e, qr])

    cr = Dense(nfc, use_bias=True, name='dense51')(cr)
    cr = Activation('elu')(cr)
    cr = Dropout(ndr)(cr)
    cr = Dense(nfc, use_bias=True, name='dense52')(cr)
    cr = Activation('elu')(cr)
    cr = Dropout(ndr)(cr)

    cr = Dense(1, name='dense6')(cr)

    counter = Model(i, cr)
    return counter

def get_query_model(nr, nfc = 128, nn=3):
    ndr = 0.3
    sketch = Input(shape=(1, nr))
    predicate_emb = Input(shape=(1, 2 * nfc))

    e = sketch
    for n in range(nn):
        e = Dense(nfc, use_bias=True, name='dense2' + str(n))(e)
        e = Activation('elu')(e)
        e = Dropout(ndr)(e)
    e = Dense(nfc, use_bias=True, name='dense2f')(e)
    e = Activation('elu')(e)
    e = Dropout(ndr)(e)

    qr = Dense(nfc, use_bias=True, name='denseq')(predicate_emb)
    qr = Activation('elu')(qr)
    qr = Dropout(ndr)(qr)
    qr = Lambda(lambda x: x[:, 0, :], output_shape=(lambda shape: (shape[0], shape[2])))(qr)

    cr = multiply([e, qr])

    cr = Dense(nfc, use_bias=True, name='dense51')(cr)
    cr = Activation('elu')(cr)
    cr = Dropout(ndr)(cr)
    cr = Dense(nfc, use_bias=True, name='dense52')(cr)
    cr = Activation('elu')(cr)
    cr = Dropout(ndr)(cr)

    cr = Dense(1, name='dense6')(cr)

    counter = Model([sketch, predicate_emb], cr)
    return counter