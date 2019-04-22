import json

import keras
from keras import Sequential
from keras.layers import Embedding, Dropout, Bidirectional, LSTM, Dense, Activation

from config import *
from data_utils import *

np.random.seed(81)
word2id, embedding_matrix, vocab = load_embeddings(file_path="glove.6B.100d.txt", embedding_size=100)
with open('word2id.json', 'w') as fout:
    json.dump(word2id, fp=fout)

X, y, max_len = get_data(word2id)


def get_model(max_features, embedding_dims, embedding_matrix, max_len,
              emb_dropout, rnn_units, rnn_dropout, recurrent_dropout):
    model = Sequential()
    model.add(Embedding(input_dim=max_features,
                        output_dim=embedding_dims,
                        weights=[embedding_matrix],
                        input_length=max_len
                        , trainable=False))
    model.add(Dropout(emb_dropout))
    model.add(Bidirectional(LSTM(units=rnn_units, dropout=rnn_dropout, recurrent_dropout=recurrent_dropout)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model


model = get_model(len(word2id.keys()), embedding_dims, embedding_matrix, max_len, emb_dropout, rnn_units,
                  rnn_dropout, recurrent_dropout)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=["accuracy"])
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1,
                                             mode='auto')
model.fit(pad_sequence(X, max_len), y, batch_size=batch_size, epochs=epochs,
          validation_split=0.1, callbacks=[earlystop_cb])

model.save_weights("weights.h5", overwrite=True)
