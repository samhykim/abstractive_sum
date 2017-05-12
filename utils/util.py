#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2016-17: Homework 3
util.py: General utility routines
Arun Chaganty <chaganty@stanford.edu>
"""

from __future__ import division

import sys
import time
import logging
import StringIO
from collections import defaultdict, Counter, OrderedDict
import numpy as np
from numpy import array, zeros, allclose

logger = logging.getLogger("hw3")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def read_data(articles, headlines):
    """
    Reads a input stream @fstream (e.g. output of `open(fname, 'r')`) in txt file format.
    @returns a list of examples [(tokens)], [(labels)]. @tokens and @labels are lists of string.
    """

    ret_articles = []
    ret_headlines = []

    for article in articles:
        article = article.strip()
        words = article.split(" ")
        ret_articles.append(words)

    for headline in headlines:
        headline = headline.strip()
        words = headline.split(" ")
        ret_headlines.append(words)
    
    assert len(ret_articles) == len(ret_headlines)
    return ret_articles, ret_headlines  
        

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)


def minibatches(data, labels, minibatch_size, shuffle=True):

    data_size = len(data)
    data = np.array(data)
    labels = np.array(labels)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
        
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        if minibatch_indices.shape[0] < minibatch_size:
            if shuffle:
                np.random.shuffle(indices)
            minibatch_indices = indices[:minibatch_size]
        yield data[minibatch_indices], labels[minibatch_indices]


def print_sentence(output, sentence, labels, predictions):

    spacings = [max(len(sentence[i]), len(labels[i]), len(predictions[i])) for i in range(len(sentence))]
    # Compute the word spacing
    output.write("x : ")
    for token, spacing in zip(sentence, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

    output.write("y*: ")
    for token, spacing in zip(labels, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

    output.write("y': ")
    for token, spacing in zip(predictions, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")
