import os
import time
from beeprint import pp
import numpy as np
from tensorflow.python.ops import nn_ops

import tensorflow as tf
import math
from tensorflow.python.ops import embedding_ops, rnn_cell, rnn

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope


class seq2seq(object):
    def __init__(self, sess, config, vocab_size, log_dir):
        self.batch_size = batch_size = config.batch_size
        self.utt_cell_size = utt_cell_size = config.cell_size
        self.vocab_size = vocab_size
        self.encoder_batch = encoder_batch = tf.placeholder(dtype=tf.int32, shape=(None, None), name="encoder_seq")
        self.decoder_batch = decoder_batch = tf.placeholder(dtype=tf.int32, shape=(None, None), name="decoder_seq")
        self.encoder_lens = encoder_lens = tf.placeholder(dtype=tf.int32, shape=(None), name="encoder_lens")
        # include GO sent and EOS
        self.decoder_lens = decoder_lens = tf.placeholder(dtype=tf.int32, shape=(None), name="decoder_lens")
        self.learning_rate = tf.Variable(float(config.init_lr), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * config.lr_decay)

        max_encode_sent_len = array_ops.shape(self.encoder_batch)[1]
        max_decode_sent_len = array_ops.shape(self.decoder_batch)[1]
        # the decoder is in format GO sent EOS. Mostly we either work with GO sent or sent EOS.
        max_decode_sent_len_minus_one = max_decode_sent_len - 1
        with variable_scope.variable_scope("word-embedding"):
            embedding = tf.get_variable("embedding", [vocab_size, config.embed_size], dtype=tf.float32)
            encoder_embedding = embedding_ops.embedding_lookup(embedding,
                                                               tf.squeeze(tf.reshape(self.encoder_batch, [-1, 1]),
                                                                          squeeze_dims=[1]))
            encoder_embedding = tf.reshape(encoder_embedding, [-1, max_encode_sent_len, config.embed_size])
            # Tony decoder embedding we need to remove the last column
            decoder_embedding = embedding_ops.embedding_lookup(embedding,
                                                               tf.squeeze(tf.reshape(
                                                                   self.decoder_batch[:, 0:max_decode_sent_len_minus_one],
                                                                   [-1, 1]), squeeze_dims=[1]))
            decoder_embedding = tf.reshape(decoder_embedding, [-1, max_decode_sent_len_minus_one, config.embed_size])

        with tf.variable_scope('seqToseq'):
            with tf.variable_scope('enc'):
                if config.cell_type == "gru":
                    cell_enc = tf.nn.rnn_cell.GRUCell(utt_cell_size)
                elif config.cell_type == "lstm":
                    cell_enc = tf.nn.rnn_cell.LSTMCell(utt_cell_size, state_is_tuple=True)
                else:
                    raise ValueError("unknown cell type")

                if config.keep_prob < 1.0:
                    cell_enc = rnn_cell.DropoutWrapper(cell_enc, output_keep_prob=config.keep_prob, input_keep_prob=config.keep_prob)

                if config.num_layer > 1:
                    cell_enc = rnn_cell.MultiRNNCell([cell_enc] * config.num_layer, state_is_tuple=True)

                _, encoder_last_state = rnn.dynamic_rnn(cell_enc, encoder_embedding, sequence_length=encoder_lens,
                                                        dtype=tf.float32)
                if config.num_layer > 1:
                    encoder_last_state = encoder_last_state[-1]

            with tf.variable_scope('dec'):
                if config.cell_type == "gru":
                    cell_dec = tf.nn.rnn_cell.GRUCell(utt_cell_size)
                elif config.cell_type == "lstm":
                    cell_dec = tf.nn.rnn_cell.LSTMCell(utt_cell_size, state_is_tuple=True)
                else:
                    raise ValueError("unknown cell type")

                # Tony: decoder length - 1 since we don't want to feed in EOS
                output, _ = rnn.dynamic_rnn(cell_dec, decoder_embedding, initial_state=encoder_last_state,
                                            sequence_length=decoder_lens-1, dtype=tf.float32)
                W = tf.get_variable('linear_W', [utt_cell_size, vocab_size], dtype=tf.float32)
                b = tf.get_variable('linear_b', [vocab_size], dtype=tf.float32, initializer=tf.zeros_initializer)

                # This part adapted from http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
                logits_flat = tf.matmul(tf.reshape(output, [-1, utt_cell_size]), W) + b
                labels_flat = tf.squeeze(tf.reshape(self.decoder_batch[:, 1:max_decode_sent_len], [-1, 1]), squeeze_dims=[1])
                weights = tf.to_float(tf.sign(labels_flat))
                self.losses = nn_ops.sparse_softmax_cross_entropy_with_logits(logits_flat, labels_flat)
                self.losses *= weights
                self.mean_loss = tf.reduce_sum(self.losses) / tf.cast(batch_size, tf.float32)
                tf.scalar_summary('cross_entropy_loss', self.mean_loss)
                self.merged = tf.merge_all_summaries()

                # choose a optimizer
                if config.op == "adam":
                    optim = tf.train.AdamOptimizer(self.learning_rate)
                elif config.op == "rmsprop":
                    optim = tf.train.RMSPropOptimizer(self.learning_rate)
                else:
                    optim = tf.train.GradientDescentOptimizer(self.learning_rate)

                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), config.grad_clip)
                self.train_ops = optim.apply_gradients(zip(grads, tvars))
                self.saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V2)

            if log_dir is not None:
                self.print_model_stats(tf.trainable_variables())
                train_log_dir = os.path.join(log_dir, "train")
                print("Save summary to %s" % log_dir)
                self.train_summary_writer = tf.train.SummaryWriter(train_log_dir, sess.graph)

    def print_model_stats(self, tvars):
        total_parameters = 0
        for variable in tvars:
            print("Trainable %s" % variable.name)
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total number of trainble parameters is %d" % total_parameters)

    def train(self, global_t, sess, train_feed):
        losses = []
        local_t = 0
        total_word_num = 0
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            encoder_len, decoder_len, encoder_x, decoder_y = batch

            fetches = [self.train_ops, self.mean_loss, self.merged]
            feed_dict = {self.encoder_batch: encoder_x, self.decoder_batch: decoder_y, self.encoder_lens: encoder_len,
                         self.decoder_lens: decoder_len}
            _, loss, summary = sess.run(fetches, feed_dict)
            losses.append(loss)
            global_t += 1
            local_t += 1
            total_word_num += np.sum(decoder_len-1)  # since we remove GO for prediction
            if local_t % (train_feed.num_batch / 50) == 0:
                train_loss = np.sum(losses) / total_word_num * train_feed.batch_size
                print("%.2f train loss %f perleixty %f" %
                      (local_t / float(train_feed.num_batch), float(train_loss), np.exp(train_loss)))
        train_loss = np.sum(losses) / total_word_num * train_feed.batch_size
        print("Train loss %f perleixty %f" % (float(train_loss), np.exp(train_loss)))

        return global_t, train_loss

    def valid(self, name, sess, valid_feed):
        """
        No training is involved. Just forward path and compute the metrics
        :param name: the name, ususally TEST or VALID
        :param sess: the tf session
        :param valid_feed: the data feed
        :return: average loss
        """
        losses = []
        total_word_num = 0

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break

            encoder_len, decoder_len, encoder_x, decoder_y = batch

            fetches = [self.mean_loss]

            feed_dict = {self.encoder_batch: encoder_x, self.decoder_batch: decoder_y, self.encoder_lens: encoder_len,
                         self.decoder_lens: decoder_len}

            loss = sess.run(fetches, feed_dict)
            total_word_num += np.sum(decoder_len - 1) # since we remove GO for prediction
            losses.append(loss)

        # print final stats
        valid_loss = float(np.sum(losses) / total_word_num * valid_feed.batch_size)
        print("%s loss %f and perplexity %f" % (name, valid_loss, np.exp(valid_loss)))
        return valid_loss
