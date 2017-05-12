#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3: Grooving with GRUs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from collections import Counter

import tensorflow as tf
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from utils.util import Progbar, minibatches, read_data
from utils.data_util import load_and_preprocess_data, load_data, save_file, load_file
from vocab import Vocab
from model import Model
#from lstm_enc_dec import Config
#import lstm_enc_dec
#import test_model
#from lstm_enc_dec_attention_bi import Config, Summarizer
from lstm_enc_dec_pointer import Config, Summarizer

#from test_model import Config

RESULTS_PATH = "./results"

matplotlib.use('TkAgg')
logger = logging.getLogger("final_project")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def plot_length_dist(data, name):
    lengths = Counter()
    for sentence in data:
        lengths[len(sentence)] += 1
    key, val = zip(*filter(lambda x: x[0] < 200, lengths.items()))

    plt.figure()
    plt.bar(key, val)
    plt.xlabel('lengths')
    plt.ylabel('Frequency')
    plt.title('Sentence Lengths distributions')
    plt.savefig(name)

def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            if "Variable" in saved_var_name:
                continue
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def restore_checkpoint(sess, saver, checkpoint_dir, pointer = False):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    #ckpt = tf.train.get_checkpoint_state(checkpoint_dir + "/checkpoint")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        logger.info('restoring...')
        if pointer:
            return optimistic_restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

def dir_name(args):
    basename = "dev"
    if args.attention:
        basename += "_attention"
    if args.bidirectional:
        basename += "_bi"
    if args.pointer:
        basename += "_pointer"
    if args.beam_search:
        basename += "_beam"
    return basename 


def save_predictions(actual, predicted, args):
    #assert len(actual) == len(predicted)
    with open(os.path.join(RESULTS_PATH, dir_name(args) + "_prediction.txt"), "w") as file:
        for i in range(len(actual)):
            file.write("y*: " + " ".join(actual[i]) + "\n")
            file.write("y': " + predicted[i] + "\n\n")

def train(args):
    config = Config(args)
    config.model_output = args.checkpoint_dir

    logger.info("Loading data...",)
    start = time.time()
    if not args.load:
        train_vocab, train, dev = load_and_preprocess_data(args)
    else:
        train_vocab, train, dev = load_data(args.pointer, vocab_size=args.vocab_size)


    logger.info("took %.2f seconds", time.time() - start)


    print("")

    plot_length_dist(train[0], "train_length_dist.png")
    plot_length_dist(train[1], "label_length_dist.png")
    #return


    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        #model = test_model.Test(config, train_vocab, labels_vocab)
        model = Summarizer(config, train_vocab, True, args.attention, bidirectional = args.bidirectional, pointer = args.pointer)
        logger.info("took %.2f seconds", time.time() - start)


        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


        with tf.Session() as sess:
            sess.run(init)
            restore_checkpoint(sess, saver, config.model_output)
            model.fit(sess, saver, train, dev)
            #model.fit(sess, saver, args.data_train)
            return 


def predict(args):
    config = Config(args)
    config.model_output = args.checkpoint_dir

    #config.batch_size = 1
    logger.info("Loading data...")
    start = time.time()
    if not args.load:
        train_vocab, test = load_and_preprocess_data(args, predict=True)
    else:
        train_vocab, test = load_data(args.pointer, predict = True, vocab_size=args.vocab_size)
    test_articles, test_headlines = read_data(args.data_test, args.data_test_labels)
    logger.info("took %.2f seconds", time.time() - start)
    print("")
    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = Summarizer(config, train_vocab, False, args.attention, args.beam_search, args.bidirectional, args.pointer)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            restore_checkpoint(sess, saver, config.model_output, args.pointer)
            logger.info("Predicting...")
            start = time.time()

            predicted = model.predict(sess, test)
            logger.info("took %.2f seconds", time.time() - start)
            save_predictions(test_headlines, predicted, args)

def interactive(args):
    config = Config(args)
    config.model_output = args.checkpoint_dir

    config.batch_size = 1

    logger.info("Loading data...")
    start = time.time()
    train_vocab = load_data(args.pointer, interactive = True, vocab_size=args.vocab_size)
    logger.info("took %.2f seconds", time.time() - start)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = Summarizer(config, train_vocab, False, args.attention, args.beam_search, args.bidirectional, args.pointer)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            restore_checkpoint(sess, saver, config.model_output, args.pointer)
            # Interactive mode
            model.interactive(sess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs a sequence model to test latching behavior of memory, e.g. 100000000 -> 1')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='Train model')
    command_parser.add_argument('-dt', '--data-train',  type=argparse.FileType('r'), default="data_processed/train_text", help="Training data")
    command_parser.add_argument('-dtl', '--data-train-labels', type=argparse.FileType('r'), default="data_processed/train_headlines", help="Training labels")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data_processed/dev_text", help="Development data")
    command_parser.add_argument('-ddl', '--data-dev-labels', type=argparse.FileType('r'), default="data_processed/dev_headlines", help="Development labels")
    command_parser.add_argument('-l', '--load', type=int, default=1, help="Load data")
    command_parser.add_argument('-a', '--attention', type=int, default=1, help="Use attention")

    command_parser.add_argument('-bi', '--bidirectional', type=int, default=1, help="Use bidirectional lstm encoding")
    command_parser.add_argument('-p', '--pointer', type=int, default=1, help="Use pointer")
    command_parser.add_argument('-c', '--checkpoint_dir', type=str, default="./checkpoints", help="Checkpoint directory")
    command_parser.add_argument('-v', '--vocab_size', type=int, default=15000, help="vocab size")
    
    command_parser.set_defaults(func=train)

    command_parser = subparsers.add_parser('predict', help='Predict with model')
    command_parser.add_argument('-dts', '--data-test', type=argparse.FileType('r'), default="data_processed/test_text", help="Test data")
    command_parser.add_argument('-dtsl', '--data-test-labels', type=argparse.FileType('r'), default="data_processed/test_headlines", help="Test labels")
    command_parser.add_argument('-a', '--attention', type=int, default=1, help="Use attention")
    command_parser.add_argument('-b', '--beam_search', type=int, default=1, help="Use beam search")
    command_parser.add_argument('-bi', '--bidirectional', type=int, default=1, help="Use bidirectional lstm encoding")
    command_parser.add_argument('-p', '--pointer', type=int, default=1, help="Use pointer")
    command_parser.add_argument('-c', '--checkpoint_dir', type=str, default="./checkpoints", help="Checkpoint directory")
    command_parser.add_argument('-l', '--load', type=int, default=1, help="Load data")
    command_parser.add_argument('-v', '--vocab_size', type=int, default=15000, help="vocab size")

    command_parser.set_defaults(func=predict)

    command_parser = subparsers.add_parser('interactive', help='Interactive mode')
    command_parser.add_argument('-a', '--attention', type=int, default=1, help="Use attention")
    command_parser.add_argument('-b', '--beam_search', type=int, default=1, help="Use beam search")
    command_parser.add_argument('-bi', '--bidirectional', type=int, default=1, help="Use bidirectional lstm encoding")
    command_parser.add_argument('-p', '--pointer', type=int, default=1, help="Use pointer")
    command_parser.add_argument('-c', '--checkpoint_dir', type=str, default="./checkpoints", help="Checkpoint directory")
    command_parser.add_argument('-v', '--vocab_size', type=int, default=15000, help="vocab size")    

    command_parser.set_defaults(func=interactive)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
