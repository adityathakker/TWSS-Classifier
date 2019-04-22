from config import *
from data_utils import *
from train import get_model

max_len = 25
word2id, embedding_matrix, vocab = load_embeddings("glove.6B.100d.txt", 100)

model = get_model(len(word2id.keys()), embedding_dims, embedding_matrix, max_len, emb_dropout, rnn_units,
                  rnn_dropout, recurrent_dropout)
model.load_weights("weights.h5")

while True:
    text = input("Enter sentence:")
    X = preprocess(text, word2id)
    print(X)
    print(model.predict(pad_sequence(X, max_len)))
