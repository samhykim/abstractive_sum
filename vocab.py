import os
import pickle
import logging
from collections import Counter

import numpy as np
from utils.util import read_data
from utils.defs import NONE, NUM, UNK, EMBED_SIZE, START_TOKEN, END_TOKEN

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

DATA_PATH = "./data"

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return NUM
    else: return word.lower()


def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}

def build_reverse_dict(tok2id):
    id2tok = {}
    for token, id in tok2id.items():
        id2tok[id] = token
    id2tok[0] = NONE
    return id2tok

class Vocab(object):
    """
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    def __init__(self, tok2id, id2tok, max_length):
        self.tok2id = tok2id
        self.id2tok = id2tok
        self.START = tok2id[START_TOKEN]
        self.END = tok2id[END_TOKEN]
        self.max_length = max_length

    def vectorize_example(self, sentence):
        sentence = [self.tok2id.get(normalize(word), self.tok2id[UNK]) for word in sentence]
        return sentence

    def vectorize(self, data):
        return [self.vectorize_example(sentence) for sentence in data]

    def vectorize_pointer(self, data, labels):
        data_processed = []
        labels_processed = []
        libraries = []
        for i, sentence in enumerate(data):
            sentence_processed = []
            label_processed = []
            unknowns = {}
            library = {}
            num_unknowns = 0
            for word in sentence:
                curr_word = self.tok2id.get(word.lower(), self.tok2id[UNK])
                if curr_word == self.tok2id[UNK]:
                    if word not in unknowns:
                        unknowns[word] = len(self.tok2id) + 1 + num_unknowns
                        library[len(self.tok2id) + 1 + num_unknowns] = word
                        num_unknowns += 1
                    sentence_processed.append(unknowns[word])
                else:
                    sentence_processed.append(curr_word)
            for word in labels[i]:
                curr_word = self.tok2id.get(word.lower(), self.tok2id[UNK])
                if curr_word == self.tok2id[UNK]:
                    if word in unknowns:
                        label_processed.append(unknowns[word])
                    else:
                        label_processed.append(curr_word)
                else:
                    label_processed.append(curr_word)
            data_processed.append(sentence_processed)
            labels_processed.append(label_processed)
            libraries.append(library)
        return data_processed, labels_processed, libraries
        
    @classmethod
    def build(cls, data, vocab_size=15000):
        # Preprocess data to construct an embedding
        # Reserve 0 for the special NIL token.
        # Add vocab from articles
        tok2id = build_dict([normalize(word) for sentence in data for word in sentence], max_words=vocab_size, offset=1)
        # Add special tokens
        tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id) + 1))

        assert sorted(tok2id.items(), key=lambda t: t[1])[0][1] == 1
        logger.info("Built dictionary for %d features.", len(tok2id))

        id2tok = build_reverse_dict(tok2id)

        max_length = max(len(sentence) for sentence in data)

        return cls(tok2id, id2tok, max_length)

    def save(self, basename):
        # Make sure the directory exists.
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        # Save the tok2id map.
        with open(os.path.join(DATA_PATH, basename + ".pkl"), "wb") as f:
            pickle.dump([self.tok2id, self.id2tok, self.max_length], f)

        # Create vocab.txt txt file
        with open(os.path.join(DATA_PATH, basename + ".txt"), "w") as f:
            for token, id in sorted(self.tok2id.items(), key=lambda x:x[1]):
                f.write("{0}\t{1}\n".format(token, id))


    @classmethod
    def load(cls, basename):
        # Make sure the directory exists.
        assert os.path.exists(DATA_PATH) and os.path.exists(os.path.join(DATA_PATH, basename + ".pkl"))
        # Save the tok2id map.
        with open(os.path.join(DATA_PATH, basename + ".pkl"), "rb") as f:
            tok2id, id2tok, max_length = pickle.load(f)
        return cls(tok2id, id2tok, max_length)
