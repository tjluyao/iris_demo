import keras.backend as K
from keras.layers import Input, Dense, Embedding, Lambda, Reshape, Dropout, multiply, Activation
from keras.models import Model

def get_query_model(nr, nfc = 128, nn=3):
    ndr = 0.5
    sketch = Input(shape=(1, nr))
    query_emb = Input(shape=(1, nfc))

    e = sketch
    for n in range(nn):
        e = Dense(nfc, use_bias=True, name='dense2' + str(n))(e)
        e = Activation('elu')(e)
        e = Dropout(ndr)(e)
    e = Dense(nfc, use_bias=True, name='dense2f')(e)
    e = Activation('elu')(e)
    e = Dropout(ndr)(e)

    cr = multiply([e, query_emb])

    cr = Dense(nfc, use_bias=True, name='dense51')(cr)
    cr = Activation('elu')(cr)
    cr = Dropout(ndr)(cr)
    cr = Dense(nfc, use_bias=True, name='dense52')(cr)
    cr = Activation('elu')(cr)
    cr = Dropout(ndr)(cr)

    cr = Dense(1, name='dense6')(cr)

    counter = Model([sketch, query_emb], cr)
    return counter