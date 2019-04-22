import re

import numpy as np
import pandas as pd
from tqdm import tqdm

PAD = "<pad>"
UNK = "<unk>"


def load_embeddings(file_path, embedding_size):
    embedding = []
    vocab = []
    line_number = 0
    with open(file_path, 'r', encoding='UTF-8') as f:
        print("Loading Embeddings...")
        for each_line in tqdm(f):

            line_number += 1
            row = each_line.split(' ')

            if len(row) == 2:
                continue
            vocab.append(row[0])
            if len(row[1:]) != embedding_size:
                print(row[0])
                print(len(row[1:]))
            embedding.append(np.asarray(row[1:], dtype='float32'))

    word2id = dict(zip(vocab, range(2, len(vocab) + 2)))
    word2id[PAD] = 0
    word2id[UNK] = 1

    extra_embedding = [np.zeros(embedding_size), np.random.uniform(-0.1, 0.1, embedding_size)]
    embedding = np.append(extra_embedding, embedding, 0)
    return word2id, embedding, vocab


def get_data(word2id):
    print("Getting Data...")
    def is_twss(c):
        if c == "TWSS":
            return [1, 0]
        else:
            return [0, 1]

    def preprocess(text):
        text = re.sub(' +', ' ', text.strip().lower())
        text = re.sub(r'[^\w\s]', '', text)
        text_list = text.split()
        return [word2id.get(each_token, 1) for each_token in text_list]

    max_len = 0
    df = pd.read_csv("TWSS_dataset.csv")
    X_list = []
    for item in tqdm(df.iloc[:, 0]):
        tokens = preprocess(str(item))
        max_len = max(len(tokens), max_len)
        X_list.append(tokens)
    X = np.array(X_list)
    y = np.array([np.array(is_twss(item)) for item in df.iloc[:, 1]]).reshape(-1, 1)

    shuffle_indices = np.random.permutation(np.arange(X.shape[0]))
    X = X[shuffle_indices]
    y = y[shuffle_indices]
    return X, y, max_len


def pad_sequence(sequences, max_len):
    if max_len <= 0:
        return sequences
    shape = (len(sequences), max_len)
    padded_sequences = np.full(shape, 0)
    for i, each_sequence in enumerate(sequences):
        if len(each_sequence) > max_len:
            padded_sequences[i] = each_sequence[:max_len]
        else:
            padded_sequences[i, :len(each_sequence)] = each_sequence
    return padded_sequences


if __name__ == "__main__":
    word2id, embedding_matrix, vocab = load_embeddings("glove.6B.100d.txt", 100)
    X, y, max_len = get_data(word2id)
    print(X)
    print(y)
    print(max_len)
