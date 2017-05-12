#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3: Grooving with GRUs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

#from util import Progbar, minibatches
from model import Model
from utils.util import Progbar, minibatches
from beam_search import beam_search

logger = logging.getLogger("final_project")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """All model hyperparameters """
    max_enc_length = 60 # Length of sequence used.
    max_dec_length = 10
    batch_size = 64
    n_epochs = 10
    lr = 0.01
    max_grad_norm = 5.
    lstm_num_units = 256 #The number of units in the LSTM cell
    #Biases of the forget gate are initialized by default to 1
    # in order to reduce the scale of forgetting at the beginning of the training.
    lstm_forget_bias = 1.0
    lstm_use_peepholes = False #set True to enable diagonal/peephole connections.
    lstm_cell_clip = None #Clip value float (optional)
    lstm_initializer = tf.contrib.layers.xavier_initializer()
    embed_size = 128
    n_features = 1
    model_output = './checkpoints/model'
    buckets = [(max_enc_length, max_dec_length)]
    num_samples = 512
    vocab_size = 10004 # vocab size of 10000 + 4 special tokens

    def __init__(self, args):
        pass 

#class LSTMCell(tf.nn.rnn_cell.LSTMCell):
class LSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(self, config):
#        tf.nn.rnn_cell.LSTMCell.__init__(self, num_units = config.lstm_num_units, \
#            use_peepholes = config.lstm_use_peepholes, cell_clip = config.lstm_cell_clip, \
#            initializer = config.lstm_initializer, forget_bias = config.lstm_forget_bias)

        tf.nn.rnn_cell.BasicLSTMCell.__init__(self, num_units = config.lstm_num_units)

class Summarizer(Model):
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        NOTE: You do not have to do anything here.
        """
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_enc_length), name="input_placeholder")
        self.label_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_dec_length), name="label_placeholder")
        self.mask_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.max_dec_length - 1), name="mask_placeholder")
        self.dropout_placeholder = tf.placeholder(tf.float32, (), name="dropout")

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout = 1):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        feed_dict = {
            self.input_placeholder: inputs_batch,
            }
        if labels_batch is not None:
            feed_dict[self.label_placeholder] = labels_batch
        if dropout != None:
            feed_dict[self.dropout_placeholder] = dropout
        def _create_mask(dec):
            if dec == None:
                return None
            dec_mask = []
            for i in range(self.config.batch_size):
                if len(dec.shape) < 2:
                    dec_single = dec
                else:
                    dec_single = dec[i,:]
                batch_mask = []
                for j in range(len(dec_single) - 1):
                    if dec_single[j+1] == 0: # pad ID (offset by 1) 
                        batch_mask.append(0.0)
                    else: batch_mask.append(1.0)
                dec_mask.append(batch_mask)
            return dec_mask

        feed_dict[self.mask_placeholder] = _create_mask(labels_batch)

        return feed_dict
    

    def add_encoder_op(self, embeddings, is_train = True):
        dropout_rate = self.dropout_placeholder

        states = []
        with tf.variable_scope("ENCODE"):
            cell = LSTMCell(self.config)
            batch_size = tf.shape(embeddings)[0]
            state_t = cell.zero_state(batch_size, tf.float32)
        
            states = []
            with tf.variable_scope("ENCODING") as scope:
                for time_step in range(self.config.max_enc_length):
                    if time_step > 0:
                        scope.reuse_variables()
                    if time_step == 0:
                        state_t_dropout = state_t
                    input_x = embeddings[:,time_step]
                    _, state_t = cell(input_x, state_t_dropout, scope=scope)
                    states.append(state_t)
                    if is_train:
                        state_t_dropout = (tf.nn.dropout(state_t[0], dropout_rate), tf.nn.dropout(state_t[1], dropout_rate))
                    else:
                        state_t_dropout = state_t
            return states
    
    def add_decoder_train_op(self, embeddings, states):
        state_t = states[-1]
        with tf.variable_scope("DECODE"):
            cell = LSTMCell(self.config)
            preds = []

            dropout_rate = self.dropout_placeholder

            U = tf.get_variable(name="U", initializer=tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units, self.config.lstm_num_units))
            b = tf.get_variable(name="b", initializer=tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units,))

            with tf.variable_scope("DECODING") as scope:
                for time_step in range(self.config.max_dec_length):
                    if time_step > 0:
                        scope.reuse_variables()
                    state_drop_t = (tf.nn.dropout(state_t[0], dropout_rate), tf.nn.dropout(state_t[1], dropout_rate))
                    input_x = embeddings[:, time_step]
                    o_t, state_t = cell(input_x, state_drop_t, scope = scope)
                    o_drop_t = tf.nn.dropout(o_t, dropout_rate)
                    y_t = tf.matmul(o_drop_t, U) + b
                    preds.append(y_t)
        
            preds = tf.stack(preds, axis=1)
            assert preds.get_shape().as_list() == [None, self.config.max_dec_length, self.config.lstm_num_units], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.max_dec_length, self.config.vocab_size], preds.get_shape().as_list())
            return preds
                
    def run_cell(self, word_ind, label_embeddings, U, b, cell, time_step, state_t, scope):
        if time_step > 0:
            scope.reuse_variables()
        word_embedding = tf.reshape(tf.nn.embedding_lookup(label_embeddings, word_ind), (-1, self.config.embed_size))
        o_t, state_t = cell(word_embedding, state_t, scope = scope)
        word = tf.matmul(o_t, U) + b # size=(batch_size, lstm_num_units)

        # projection
        if self.output_projection is not None:
            word = tf.matmul(word, self.output_projection[0]) + self.output_projection[1]
        return word, state_t

    def add_decoder_predict_op(self, embeddings, states, label_embeddings, do_beam_search=True):
        state_t = states[-1]
        with tf.variable_scope("DECODE"):
            cell = LSTMCell(self.config)
            preds = []
            preds_proj = []

        
            U = tf.get_variable(name="U", initializer=tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units, self.config.lstm_num_units))
            b = tf.get_variable(name="b", initializer=tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units,))
            word_ind = [self.labels_vocab.START for _ in range(self.config.batch_size)]

            if do_beam_search:
                with tf.variable_scope("DECODING") as scope:
                    preds_proj = beam_search(word_ind, label_embeddings, U, b, cell, state_t, self.config.max_dec_length, self.config.batch_size, self.run_cell, scope)
            else:
                with tf.variable_scope("DECODING") as scope:
                    for time_step in range(self.config.max_dec_length):
                        word = self.run_cell(word_ind, label_embeddings, U, b, cell, time_step, state_t, scope)


                        preds.append(word)
                        word_ind = tf.argmax(word, axis=1)
                        preds_proj.append(word_ind)

                preds_proj = tf.stack(preds_proj, axis=1)
                preds = tf.stack(preds, axis=1)

            assert preds_proj.get_shape().as_list() == [self.config.batch_size, self.config.max_dec_length]
            return preds_proj, preds
       
    def add_test_op(self):
        text, labels, label_embeddings = self.add_embedding(False)
        states = self.add_encoder_op(text, is_train = False)
        return self.add_decoder_predict_op(labels, states, label_embeddings)

    def add_prediction_op(self):
        text, labels = self.add_embedding()
        states = self.add_encoder_op(text)
        return self.add_decoder_train_op(labels, states)
        
    def add_embedding(self, is_train = True):
        train_embeddings = np.array(np.random.randn(len(self.train_vocab.tok2id) + 1, self.config.embed_size), dtype=np.float32)
        label_embeddings = np.array(np.random.randn(len(self.labels_vocab.tok2id) + 1, self.config.embed_size), dtype=np.float32)
        
        logger.info("Initialized embeddings.")

        embedding = tf.Variable(train_embeddings, name="train_embeddings")
        label_embedding = tf.Variable(label_embeddings, name="label_embeddings")
        train_embeddings = tf.reshape(tf.nn.embedding_lookup(embedding, self.input_placeholder), (-1, self.config.max_enc_length, self.config.embed_size))
        label_embeddings = tf.reshape(tf.nn.embedding_lookup(label_embedding, self.label_placeholder), (-1, self.config.max_dec_length, self.config.embed_size))
        if is_train: return train_embeddings, label_embeddings
        else: return train_embeddings, label_embeddings, label_embedding

    def add_projection(self):
    	w = tf.get_variable(name='proj_w', initializer=tf.contrib.layers.xavier_initializer(),shape=(self.config.lstm_num_units, self.config.vocab_size), dtype=np.float32)
    	b = tf.get_variable(name='proj_b', initializer=tf.constant_initializer(0), shape=(self.config.vocab_size,), dtype=np.float32)
    	return (w,b)

    def sampled_sequence_loss(self, inputs, targets, weights, loss_function):
    	log_perps_list = []
    	for i in xrange(self.config.max_dec_length - 1):
    		crossent = loss_function(inputs[:,i], targets[:,i]) 
    		log_perps_list.append(tf.reduce_sum(crossent * weights[:,i]))
    	log_perps = tf.add_n(log_perps_list)
    	total_size = tf.reduce_sum(weights)
    	return log_perps / total_size



    def add_loss_op(self, preds, sampled=True):
    	w,b = self.output_projection

    	def sampled_loss(inputs, labels):
			labels = tf.reshape(labels, [-1, 1])
			return tf.nn.sampled_softmax_loss(weights=tf.transpose(w), 
                                  biases=b, 
                                  inputs=inputs,
                                  labels=labels, 
                                  num_sampled=self.config.num_samples, 
                                  num_classes=self.config.vocab_size)

        if sampled:
        	loss = self.sampled_sequence_loss(preds[:,:-1], self.label_placeholder[:,1:], self.mask_placeholder, sampled_loss)
        else:
        	loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds[:,:-1], labels=self.label_placeholder[:,1:])
        	loss = tf.reduce_mean(tf.multiply(loss_op, self.mask_placeholder))
        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.
        This version also returns the norm of gradients.
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, train, labels):
        prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses, grad_norms = [], []
        for i, (enc_batch, dec_batch) in enumerate(minibatches(train, labels, self.config.batch_size)):
            #print batch
            loss = self.train_on_batch(sess, enc_batch, dec_batch)
            #losses.append(loss)
            #grad_norms.append(grad_norm)
            #loss = 1
            prog.update(i + 1, [("train loss", loss)])

        return loss

    def preprocess_sequence_data(self, train, labels):
        ret_train, ret_labels = [], []
        for sentence in train:
            sentence = sentence[:self.config.max_enc_length]
            if len(sentence) < self.config.max_enc_length:
                sentence += [0]*(self.config.max_enc_length - len(sentence))
            ret_train.append(sentence)

        for sentence in labels:
            # Add <s> (start) and </s> (end) tokens to the decoder
            sentence = [self.labels_vocab.START] + sentence + [self.labels_vocab.END]
            sentence = sentence[:self.config.max_dec_length]
            if len(sentence) < self.config.max_dec_length:
                sentence += [0]*(self.config.max_dec_length - len(sentence)) # padding
            ret_labels.append(sentence)
        return ret_train, ret_labels


    def fit(self, sess, saver, train, labels):
        best_score = float("inf")

        train_processed, labels_processed = self.preprocess_sequence_data(train, labels)
        # train_labels = self.preprocess_sequence_data(labels, )
        # dev_set = self.preprocess_sequence_data(dev_set_raw)

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, train_processed, labels_processed)
            
            if score < best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            # if self.report:
            #     self.report.log_epoch()
            #     self.report.save()
        return best_score

    def predict(self, sess, dev, labels):
        predicted_sentences = []

        dev_processed, labels_processed = self.preprocess_sequence_data(dev, labels)
        print self.config.batch_size
        for i, (enc_batch, dec_batch) in enumerate(minibatches(dev_processed, labels_processed, self.config.batch_size, shuffle=False)):
            print enc_batch.shape, dec_batch.shape
            feed = self.create_feed_dict(enc_batch, dec_batch)
            
            outputs = sess.run(self.pred_proj, feed_dict=feed)
            for i in range(outputs.shape[0]):
				predicted = " ".join([self.labels_vocab.id2tok[i] for i in outputs[i,:]])
				print predicted
				predicted_sentences.append(predicted)

        return predicted_sentences



 
    def build(self, is_train):
        self.add_placeholders()
        self.output_projection = self.add_projection()
        if is_train:
            self.pred = self.add_prediction_op()
            self.loss = self.add_loss_op(self.pred)
            self.train_op = self.add_training_op(self.loss)
        else:
            self.pred_proj, self.pred = self.add_test_op()
            self.loss = 0
            #self.loss = self.add_loss_op(self.pred)


    def __init__(self, config, train_vocab, labels_vocab, is_train = True):
        self.config = config
        self.encoder_inputs = None
        self.decoder_inputs = None
        self.grad_norm = None
        self.train_vocab = train_vocab
        self.labels_vocab = labels_vocab

        self.build(is_train)



