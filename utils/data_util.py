#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
import os
import pickle
import logging
from collections import Counter

import numpy as np
from util import read_data
from defs import NONE, NUM, UNK, EMBED_SIZE, START_TOKEN, END_TOKEN
from vocab import Vocab

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

DATA_PATH = './data'

FDIM = 4

def featurize(embeddings, word):
    """
    Featurize a word given embeddings.
    """
    case = casing(word)
    word = normalize(word)
    case_mapping = {c: one_hot(FDIM, i) for i, c in enumerate(CASES)}
    wv = embeddings.get(word, embeddings[UNK])
    fv = case_mapping[case]
    return np.hstack((wv, fv))

def evaluate(model, X, Y):
    cm = ConfusionMatrix(labels=LBLS)
    Y_ = model.predict(X)
    for i in range(Y.shape[0]):
        y, y_ = np.argmax(Y[i]), np.argmax(Y_[i])
        cm.update(y,y_)
    cm.print_table()
    return cm.summary()

def load_data(pointer = False, predict = False, interactive = False, vocab_size=15000):
    logger.info("Loading vocab files...")
    train_vocab = Vocab.load("train_vocab_" + str(vocab_size))
    if interactive:
        return train_vocab
    logger.info("Loading data files...")
    if not predict:
        if not pointer:
            train_data = load_file("train_data.pkl")
            train_data_labels = load_file("train_data_labels.pkl")
            dev_data = load_file("dev_data.pkl")
            dev_labels = load_file("dev_data_labels.pkl")
            return train_vocab, (train_data, train_data_labels, [{} for i in range(len(train_data))]), (dev_data, dev_labels, [{} for i in range(len(dev_data))])
        train_pointer = load_file("train_pointer_files.pkl")
        dev_pointer = load_file("dev_pointer_files.pkl")
        return train_vocab, train_pointer, dev_pointer
    if not pointer:
        test_data = load_file("test_data.pkl")
        test_data_labels = load_file("test_data_labels.pkl")
        return train_vocab, (test_data, test_data_labels, [{} for i in range(len(test_data))])
    test_pointer = load_file("test_pointer_files.pkl")
    return train_vocab, test_pointer
    

def load_and_preprocess_test_data(args):
    logger.info("Loading Test Data...")
    test, test_labels = read_data(args.data_test, args.data_test_labels)
    logger.info("Done. Read %d sentences", len(test))
    
    vocab = Vocab.load("train_vocab_" + str(args.vocab_size))
    
    test_data = vocab.vectorize(test)
    test_data_labels = vocab.vectorize(test_labels)
    test_pointer = vocab.vectorize_pointer(test, test_labels)
    
    save_file("test_pointer_files.pkl", test_pointer)
    save_file("test_data.pkl", test_data)
    save_file("test_data_labels.pkl", test_data_labels)
    if not args.pointer:
        return vocab, (test_data, test_data_labels, [{} for i in range(len(test_data))])

    return vocab, test_pointer
    

def load_and_preprocess_data(args, predict=False):
    if predict:
        return load_and_preprocess_test_data(args)
    logger.info("Loading Training data...")
    
    train, train_labels = read_data(args.data_train, args.data_train_labels)

    logger.info("Done. Read %d sentences", len(train))
    logger.info("Loading dev data...")
    dev, dev_labels = read_data(args.data_dev, args.data_dev_labels)
    logger.info("Done. Read %d sentences", len(dev))

    train_vocab = Vocab.build(train, vocab_size=args.vocab_size)
    train_vocab.save("train_vocab_" + str(args.vocab_size), args.vocab_size)

    # now process all the input data.
    logger.info("Vectorizing train and dev data...")
    train_data = train_vocab.vectorize(train)
    train_data_labels = train_vocab.vectorize(train_labels)
    dev_data = train_vocab.vectorize(dev)
    dev_data_labels = train_vocab.vectorize(dev_labels)
    train_pointer = train_vocab.vectorize_pointer(train, train_labels)
    dev_pointer = train_vocab.vectorize_pointer(dev, dev_labels)
    logger.info("Saving vectorized data...")
    save_file("train_pointer_files.pkl", train_pointer)
    save_file("dev_pointer_files.pkl", dev_pointer)
    save_file("train_data.pkl", train_data)
    save_file("dev_data.pkl", dev_data)
    save_file("train_data_labels.pkl", train_data_labels)
    save_file("dev_data_labels.pkl", dev_data_labels)
    logger.info("Finished data processing.")
    if not args.pointer:
        return train_vocab, (train_data, train_data_labels, [{} for i in range(len(train_data))]), (dev_data, dev_data_labels, [{} for i in range(len(dev_data))])

    return train_vocab, train_pointer, dev_pointer


def save_file(filename, data):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    # Save the tok2id map.
    with open(os.path.join(DATA_PATH, filename), "wb") as f:
        pickle.dump(data, f)

def load_file(filename):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    # Save the tok2id map.
    with open(os.path.join(DATA_PATH, filename), "rb") as f:
        data = pickle.load(f)
    return data
