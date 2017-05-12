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
from beam_search import BeamSearch

logger = logging.getLogger("final_project")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """All model hyperparameters """
    max_enc_length = 200 # Length of sequence used.
    max_dec_length = 30
    batch_size = 64
    n_epochs = 1
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
    num_samples = 2056
    deep_output_size = 50
    vocab_size = 100004 # vocab size of 10000 + 4 special tokens

    def __init__(self, args):
        pass 

class LSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(self, config):
        tf.nn.rnn_cell.BasicLSTMCell.__init__(self, num_units = config.lstm_num_units, forget_bias = config.lstm_forget_bias)

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
    

    def add_encoder_op(self, embeddings, is_train=True):
        dropout_rate = self.dropout_placeholder

        with tf.variable_scope("ENCODE"):
            cell = LSTMCell(self.config)
            batch_size = tf.shape(embeddings)[0]
            state_t = cell.zero_state(batch_size, tf.float32)
        
            c_states = []
            m_states = []
            with tf.variable_scope("ENCODING") as scope:
                for time_step in range(self.config.max_enc_length):
                    if time_step > 0:
                        scope.reuse_variables()
                    if time_step == 0:
                        state_t_dropout = state_t
                    input_x = embeddings[:,time_step]
                    _, state_t = cell(input_x, state_t_dropout, scope=scope)
                    c_states.append(state_t[0])
                    m_states.append(state_t[1])
                    if is_train:
                        state_t_dropout = (tf.nn.dropout(state_t[0], dropout_rate), tf.nn.dropout(state_t[1], dropout_rate))
                    else:
                        state_t_dropout = state_t
            return (tf.stack(c_states, axis=0), tf.stack(m_states, axis=0))
   
    def get_output(self, input, cell, state, dropout_rate = None, old_states = None, scope = None):

        U, b, W_ac, W_am, U_o, V_o, C_oc, C_om, W_o = self.decode_params
        
        c_states, m_states = old_states
        
        def context(states, W_a, state_t):
            state_W = tf.matmul(state_t, W_a)
            scores = tf.transpose(tf.reduce_sum(tf.multiply(states, state_W), axis=2))
            scores_normalized = tf.nn.softmax(scores)
            scored_states = tf.multiply(tf.transpose(states, perm=[2, 1, 0]), scores_normalized)
            context = tf.transpose(tf.reduce_sum(scored_states, axis=2))
            return context

        def add_attention(context_c, context_m, o_t):
            t_i_hat = tf.matmul(o_t, U_o) + tf.matmul(input, V_o) + tf.matmul(context_c, C_oc) + tf.matmul(context_m, C_om)
            t_i = []
            for i in range(self.config.deep_output_size):
                t_i.append(tf.reduce_max(tf.stack([t_i_hat[:,i*2], t_i_hat[:,i*2+1]], axis=1), axis=1))
            t_i = tf.stack(t_i, axis=1)
            return tf.matmul(t_i, W_o)
            
        o_t, state_t = cell(input, state, scope)
        
        if dropout_rate == None:
            o_drop_t = o_t
            state_drop_t = state_t
        else:
            o_drop_t = tf.nn.dropout(o_t, dropout_rate)
            state_drop_t = (tf.nn.dropout(state_t[0], dropout_rate), tf.nn.dropout(state_t[1], dropout_rate))
        
        if (W_ac == None or W_am == None or U_o == None or V_o == None or \
            C_oc == None or C_om == None or W_o == None):
            return tf.matmul(o_drop_t, U) + b, state_drop_t

        context_c = context(c_states, W_ac, state_drop_t[0])
        context_m = context(m_states, W_am, state_drop_t[1])                    
        
        return add_attention(context_c, context_m, o_drop_t), state_drop_t

    def get_decode_params(self):
        with tf.variable_scope("DECODE"):
            U = tf.get_variable(name="U", initializer=tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units, self.config.lstm_num_units))
            #U = tf.get_variable(name="U", initializer=tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units, self.config.vocab_size))
            b = tf.get_variable(name="b", initializer=tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units,))
            #b = tf.get_variable(name="b", initializer=tf.contrib.layers.xavier_initializer(), shape=(self.config.vocab_size,))

            W_ac, W_am, U_o, V_o, C_oc, C_om, W_o = None, None, None, None, None, None, None

            if self.use_attention:
                W_ac = tf.get_variable(name="W_ac", initializer = tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units, self.config.lstm_num_units))
                W_am = tf.get_variable(name="W_am", initializer = tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units, self.config.lstm_num_units))
                U_o = tf.get_variable(name="U_o", initializer = tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units, 2 *self.config.deep_output_size))
                V_o = tf.get_variable(name="V_o", initializer = tf.contrib.layers.xavier_initializer(), shape=(self.config.embed_size, 2 * self.config.deep_output_size))
                C_oc = tf.get_variable(name="C_oc", initializer = tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units, 2 *self.config.deep_output_size))
                C_om = tf.get_variable(name="C_om", initializer = tf.contrib.layers.xavier_initializer(), shape=(self.config.lstm_num_units, 2 *self.config.deep_output_size))
                W_o = tf.get_variable(name="W_o", initializer = tf.contrib.layers.xavier_initializer(), shape=(self.config.deep_output_size, self.config.vocab_size))

            return U, b, W_ac, W_am, U_o, V_o, C_oc, C_om, W_o

    def add_decoder_train_op(self, embeddings, states):
        
        c_states, m_states = states
        state_t = (c_states[-1,:,:], m_states[-1,:,:])

        
        with tf.variable_scope("DECODE"):
            cell = LSTMCell(self.config)
            preds = []
            
            dropout_rate = self.dropout_placeholder                

            with tf.variable_scope("DECODING") as scope:
                for time_step in range(self.config.max_dec_length):
                    if time_step > 0:
                        scope.reuse_variables()
                    state_drop_t = (tf.nn.dropout(state_t[0], dropout_rate), tf.nn.dropout(state_t[1], dropout_rate))
                    input_x = embeddings[:, time_step]
                    y_t, state_t = self.get_output(input_x, cell, state_drop_t, dropout_rate=dropout_rate, old_states=states, scope=scope)
                    preds.append(y_t)
        
            preds = tf.stack(preds, axis=1)
            return preds
                

    def add_decoder_predict_op(self, embeddings, states):
        c_states, m_states = states
        state_t = (c_states[-1,:,:], m_states[-1,:,:])
            
        with tf.variable_scope("DECODE"):
            cell = LSTMCell(self.config)
            preds = []
            
            dropout_rate = self.dropout_placeholder
            top_hypotheses, values, indices = None, None, None
            word_ind = [self.labels_vocab.START for _ in range(self.config.batch_size)]       

            with tf.variable_scope("DECODING") as scope:
                for time_step in range(self.config.max_dec_length):
                    if time_step > 0:
                        scope.reuse_variables()

                    if self.beam_search is not None:

                        top_hypotheses, values, indices, state_t = self.beam_search.beam_search_single_step(self, time_step, top_hypotheses, values, indices,
                            cell, state_t, None, states, scope)

                    else:
                        # use greedy decoding
                        word_embedding = tf.reshape(tf.nn.embedding_lookup(self.label_embeddings, word_ind), (self.config.batch_size, self.config.embed_size))
                        y_t, state_t = self.get_output(word_embedding, cell, state_t, dropout_rate=None, old_states=states, scope=scope)
                        
                        word_ind = tf.argmax(y_t, axis=1)
                        preds.append(word_ind)

            if self.beam_search is not None:
                # hypothesis (batch_size, k, max_length)
                # values (batch_size, k)
                top_values, top_indices = tf.nn.top_k(values) # (batch_size, 1)
                for i in range(self.config.batch_size):
                    preds.append(top_hypotheses[i, top_indices[i,0], :])
                
                #top_hypothesis, _ = tf.nn.top_k(tf.reshape(top_hypotheses, [self.config.batch_size, self.config.max_dec_length,-1]), 1) # get best hypothesis
                #preds = tf.reshape(top_hypothesis, [-1, self.config.max_dec_length])
                preds = tf.stack(preds, axis=0)
            else:
                preds = tf.stack(preds, axis=1)
            return preds
       
    def add_test_op(self):
        text, labels = self.add_embedding()
        states = self.add_encoder_op(text, False)
        self.decode_params = self.get_decode_params()
        return self.add_decoder_predict_op(labels, states)

    def add_prediction_op(self):
        text, labels = self.add_embedding()
        states = self.add_encoder_op(text)
        self.decode_params = self.get_decode_params()
        return self.add_decoder_train_op(labels, states)
        
    def add_embedding(self):
        train_embeddings = np.array(np.random.randn(len(self.train_vocab.tok2id) + 1, self.config.embed_size), dtype=np.float32)
        label_embeddings = np.array(np.random.randn(len(self.labels_vocab.tok2id) + 1, self.config.embed_size), dtype=np.float32)
        
        logger.info("Initialized embeddings.")

        embedding = tf.Variable(train_embeddings, name="train_embeddings")
        label_embedding = tf.Variable(label_embeddings, name="label_embeddings")
        self.label_embeddings = label_embedding
        train_embeddings = tf.reshape(tf.nn.embedding_lookup(embedding, self.input_placeholder), (-1, self.config.max_enc_length, self.config.embed_size))
        label_embeddings = tf.reshape(tf.nn.embedding_lookup(label_embedding, self.label_placeholder), (-1, self.config.max_dec_length, self.config.embed_size))
        return train_embeddings, label_embeddings
    
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

    def add_loss_op(self, preds, sampled=False):
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
                else:
                    logger.info("Saver not working")
            print("")
            # if self.report:
            #     self.report.log_epoch()
            #     self.report.save()
        return best_score

    def predict(self, sess, dev, labels):
        predicted_sentences = []

        dev_processed, labels_processed = self.preprocess_sequence_data(dev, labels)
        prog = Progbar(target=1 + int(len(dev_processed) / self.config.batch_size))
        # predict in batches
        for i, (enc_batch, dec_batch) in enumerate(minibatches(dev_processed, labels_processed, self.config.batch_size, shuffle=False)):
            #print enc_batch.shape, dec_batch.shape
            feed = self.create_feed_dict(enc_batch, dec_batch)
            
            outputs = sess.run(self.test_op, feed_dict=feed)
            for j in range(outputs.shape[0]):
                sentence = [self.labels_vocab.id2tok[k] for k in outputs[j,:]]
                if "</s>" in sentence:
                    sentence = sentence[:sentence.index("</s>")]
                predicted = " ".join(sentence)
                print predicted
                predicted_sentences.append(predicted)
            prog.update(i + 1)

        return predicted_sentences


 
    def build(self, is_train):
        self.add_placeholders()
        self.output_projection = self.add_projection()

        if is_train:
            self.pred = self.add_prediction_op()
            self.loss = self.add_loss_op(self.pred)
            self.train_op = self.add_training_op(self.loss)
        else:
            self.test_op = self.add_test_op()

    def __init__(self, config, train_vocab, labels_vocab, is_train=True, use_attention=True, beam_search=None):
        self.use_attention = use_attention
        self.beam_search = None
        if beam_search:
            self.beam_search = BeamSearch(self)

        self.config = config
        self.encoder_inputs = None
        self.decoder_inputs = None
        self.grad_norm = None
        self.train_vocab = train_vocab
        self.labels_vocab = labels_vocab

        self.build(is_train)



