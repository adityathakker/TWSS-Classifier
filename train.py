import re
from collections import Counter

import numpy as np
import pandas as pd

PAD = "<pad>"
UNK = "<unk>"


def get_data():
    def is_twss(c):
        if c == "TWSS":
            return 1
        else:
            return 0

    def preprocess(text, vocab=None):
        text = re.sub(' +', ' ', text.strip().lower())
        text = re.sub(r'[^\w\s]', '', text)
        if vocab:
            text_list = []
            for token in text.split():
                if token in vocab:
                    text_list.append(token)
                else:
                    text_list.append(UNK)
            return " ".join(text_list)
        else:
            return text

    df = pd.read_csv("TWSS_dataset.csv")
    word_counter = Counter()
    for item in df.iloc[:, 0]:
        word_counter.update(preprocess(str(item)).split())
    vocab = [item[0] for item in word_counter.most_common(5000)]

    X = np.array([preprocess(str(item), vocab) for item in df.iloc[:, 0]]).reshape(-1, 1)
    y = np.array([is_twss(item) for item in df.iloc[:, 1]]).reshape(-1, 1)

    data = np.column_stack((X, y))
    np.random.shuffle(data)
    return data


data = get_data()
print(data)
