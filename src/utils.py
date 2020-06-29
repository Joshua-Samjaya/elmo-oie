import re
import random
import numpy as np
import nltk
from nltk.tree import Tree
import sys, os

def get_vocab_datum(data, threshold=1, log=False):
    vocab = {}
    for datum in data:
        words = datum['text'].split()
        for word in words:
            vocab[word] = vocab.get(word, 0)+1
    selected = set()
    for word in vocab.keys():
        if vocab[word] >= threshold:
            selected.add(word)
    if log:
        print(len(vocab)-len(selected), 'words were filtered out because their counts are less than', threshold)
    return selected

def filter_data_by_vocab(data, vocab, token='<unk>'):
    vocab.add(token)
    for datum in data:
        datum['text'] = ' '.join([i if i in vocab else token for i in datum['text'].split()])
    return data

def wordvec_add_unknown_vocab(wordvec, vocab, var=1, length=50):
    words = wordvec.keys()
    word = next(iter(words))
    length = len(wordvec[word])
    count = 0
    f = open('log.txt', 'w')
    for i in vocab:
        if i not in words:
            f.write(i+'\n')
            count += 1
            wordvec[i] = np.random.uniform(-var, var, length)
    f.close()
    print(count,"words are not in trained embeddings")
    return wordvec

def get_lookup(wordvec):
    # get the lookup table from word vector
    words = list(wordvec.keys())
    element = words[0]
    dim = len(wordvec[element])
    print("Embedding dim = "+str(dim))
    word_map = {}
    W = np.zeros((len(words)+1, dim), dtype=np.float32)
    for i in range(len(words)):
        word = words[i]
        W[i+1] = wordvec[word]
        word_map[word] = i+1
    return W, word_map


def get_label_map(data):
    y_map = set()
    for datum in data:
        y_map.add(datum['y'])
    label_map = {}
    for i, y in enumerate(y_map):
        label_map[y] = i
    return label_map


def get_maxlen(data_list):
    maxlen = 0
    for i in data_list:
        maxleni = max(map(lambda x: x['num_words'], i))
        maxlen = max(maxlen, maxleni)
    return maxlen


def sample_from_data(data, ratio=0.1):
    num_samples = int(ratio*len(data))
    sample = np.random.choice(data, num_samples, replace=False)
    return sample


def sample_from_numpy(X, y, ratio=0.1):
    num_samples = int(ratio*X.shape[0])
    sample = np.random.choice(X.shape[0], num_samples, replace=False)
    return X[sample], y[sample]


def consolidate_labels(labels):
    """
    Return a consolidated list of labels, e.g., O-A1 -> O, A1-I -> A
    """
    return map(consolidate_label , labels)


def consolidate_label(label):
    """
    Return a consolidated label, e.g., O-A1 -> O, A1-I -> A
    """
    return label.split("-")[0] if label.startswith("O") else label