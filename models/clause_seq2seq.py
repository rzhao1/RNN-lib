import os
import time
import numpy as np
from tensorflow.python.ops import nn_ops

import tensorflow as tf
import math
from tensorflow.python.ops import rnn_cell, rnn
from loop_functions import *

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope


class Utt2Seq(object):
    def __init__(self, sess, config, vocab_size, feature_size, max_decoder_size, log_dir, forward):
        self.batch_size = config.batch_size
        self.forward = forward
        self.utt_cell_size = utt_cell_size = config.cell_size
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.max_decoder_size = max_decoder_size
        self.beam_size = config.beam_size

        self.encoder_batch = tf.placeholder(dtype=tf.float32, shape=(None, None, feature_size), name="encoder_utts")
        self.decoder_batch = tf.placeholder(dtype=tf.int32, shape=(None, max_decoder_size), name="decoder_seq")
        self.encoder_lens = tf.placeholder(dtype=tf.int32, shape=None, name="encoder_lens")
        # include GO sent and EOS

        self.learning_rate = tf.Variable(float(config.init_lr), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * config.lr_decay)

        max_encode_sent_len = array_ops.shape(self.encoder_batch)[1]
        # As our current version decoder is fixed-size,
        # the decoder is in format GO sent EOS. Mostly we either work with GO sent or sent EOS. that is max_decoder-1
        max_decode_minus_one = self.max_decoder_size - 1

        with variable_scope.variable_scope("clause-embedding"):
            embed_w = tf.get_variable('embed_w', [feature_size, config.clause_embed_size], dtype=tf.float32)
            embed_b = tf.get_variable('embed_b', [config.clause_embed_size], dtype=tf.float32,
                                      initializer=tf.zeros_initializer)

            encoder_embedding = tf.matmul(tf.reshape(self.encoder_batch, [-1, self.feature_size]), embed_w) + embed_b
            encoder_embedding = tf.tanh(encoder_embedding)
            encoder_embedding = tf.reshape(encoder_embedding, [-1, max_encode_sent_len, config.clause_embed_size])


        with tf.variable_scope('seq2seq'):
            with tf.variable_scope('enc'):
                if config.cell_type == "gru":
                    cell_enc = tf.nn.rnn_cell.GRUCell(utt_cell_size)
                elif config.cell_type == "lstm":
                    cell_enc = tf.nn.rnn_cell.LSTMCell(utt_cell_size, state_is_tuple=True)
                else:
                    print("WARNING: unknown cell type. Use Basic RNN as default")
                    cell_enc = tf.nn.rnn_cell.BasicRNNCell(utt_cell_size)

                if config.keep_prob < 1.0:
                    cell_enc = rnn_cell.DropoutWrapper(cell_enc, output_keep_prob=config.keep_prob,
                                                       input_keep_prob=config.keep_prob)

                if config.num_layer > 1:
                    cell_enc = rnn_cell.MultiRNNCell([cell_enc] * config.num_layer, state_is_tuple=True)

                encoder_outputs, encoder_last_state = rnn.dynamic_rnn(cell_enc, encoder_embedding, dtype=tf.float32,
                                                                      sequence_length=self.encoder_lens)
                if config.num_layer > 1:
                    encoder_last_state = encoder_last_state[-1]


            # post process the decoder embedding inputs and encoder_last_state
            with variable_scope.variable_scope("word-embedding"):
                embedding = tf.get_variable("embedding", [vocab_size, config.embed_size], dtype=tf.float32)
                # Tony decoder embedding we need to remove the last column
                decoder_embedding, initial_state = self.get_dec_inp_embedding(embedding, encoder_last_state,
                                                                              self.decoder_batch[:, 0:max_decode_minus_one])

            with variable_scope.variable_scope('dec'):
                if config.cell_type == "gru":
                    cell_dec = tf.nn.rnn_cell.GRUCell(utt_cell_size)
                elif config.cell_type == "lstm":
                    cell_dec = tf.nn.rnn_cell.LSTMCell(utt_cell_size, state_is_tuple=True)
                else:
                    print("WARNING: unknown cell type. Use Basic RNN as default")
                    cell_dec = tf.nn.rnn_cell.BasicRNNCell(utt_cell_size)

                # output project to vocabulary size
                cell_dec = tf.nn.rnn_cell.OutputProjectionWrapper(cell_dec, vocab_size)

                # run decoder to get sequence outputs
                dec_outputs, beam_symbols, beam_path, log_beam_probs = self.beam_rnn_decoder(
                    decoder_embedding=decoder_embedding,
                    initial_state=encoder_last_state,
                    embedding = embedding,
                    cell=cell_dec)

                self.logits = dec_outputs
                logits_flat = tf.reshape(tf.pack(dec_outputs, 1), [-1, vocab_size])
                # skip GO in the beginning
                labels_flat = tf.squeeze(tf.reshape(self.decoder_batch[:, 1:], [-1, 1]), squeeze_dims=[1])
                # mask out labels equals PAD_ID = 0
                weights = tf.to_float(tf.sign(labels_flat))
                self.losses = nn_ops.sparse_softmax_cross_entropy_with_logits(logits_flat, labels_flat)
                self.losses *= weights
                self.loss_sum = tf.reduce_sum(self.losses)
                self.real_loss = self.loss_sum / tf.reduce_sum(weights)

                tf.scalar_summary('cross_entropy_loss', self.real_loss)
                self.merged = tf.merge_all_summaries()

                # choose a optimizer
                if config.op == "adam":
                    optim = tf.train.AdamOptimizer(self.learning_rate)
                elif config.op == "rmsprop":
                    optim = tf.train.RMSPropOptimizer(self.learning_rate)
                else:
                    optim = tf.train.GradientDescentOptimizer(self.learning_rate)

                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.real_loss, tvars), config.grad_clip)
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
            var_parameters = 1
            for dim in shape:
                var_parameters *= dim.value
            total_parameters += var_parameters
        print("Total number of trainble parameters is %d" % total_parameters)

    def beam_rnn_decoder(self, decoder_embedding, initial_state, cell, embedding, scope=None):
        """
        :param decoder_inputs: B * max_enc_len
        :param initial_state: B * cell_size
        :param cell:
        :param scope: the name scope
        :return: decoder_outputs, the last decoder_state
        """
        beam_symbols, beam_path, log_beam_probs = [], [], []

        with variable_scope.variable_scope(scope or "embedding_rnn_decoder"):
            loop_function = self.get_loop_function(embedding, self.vocab_size, beam_symbols, beam_path, log_beam_probs)
            outputs, state = rnn_decoder(decoder_embedding, initial_state, cell, loop_function=loop_function, scope=scope)
            return outputs, beam_symbols, beam_path, log_beam_probs

    def get_loop_function(self, embedding, num_symbol, beam_symbols, beam_path, log_beam_probs):
        if self.forward:
            loop_function = beam_and_embed(embedding, self.beam_size,
                                           num_symbol, beam_symbols,
                                           beam_path, log_beam_probs)
        else:
            loop_function = None
        return loop_function

    def get_dec_inp_embedding(self, embedding, initial_state, decoder_inputs):
        """
        Map the decoder inputs into embedding. If using beam search, also tile the inputs by beam_size times
        """
        _, seq_len = decoder_inputs.get_shape()
        if self.forward:
            # for beam search, we need to duplicate the input (GO symbols) by beam_size times
            # the initial state is also tiled by beam_size times
            emb_inp = []
            for i in range(seq_len):
                embed = embedding_ops.embedding_lookup(embedding, decoder_inputs[:, i])
                embed = tf.reshape(tf.tile(embed, [1, self.beam_size]), [-1, self.vocab_size])
                emb_inp.append(embed)

            initial_state = tf.reshape(tf.tile(initial_state, [1, self.beam_size]), [-1, self.utt_cell_size])
        else:
            emb_inp = [embedding_ops.embedding_lookup(embedding, decoder_inputs[:, i]) for i in range(seq_len)]

        return emb_inp, initial_state


    def train(self, global_t, sess, train_feed):
        losses = []
        local_t = 0
        total_word_num = 0
        start_time = time.time()
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            encoder_len, decoder_len, encoder_x, decoder_y = batch
            fetches = [self.train_ops, self.loss_sum, self.merged]
            feed_dict = {self.encoder_batch: encoder_x, self.decoder_batch: decoder_y, self.encoder_lens: encoder_len}
            _, loss, summary = sess.run(fetches, feed_dict)
            self.train_summary_writer.add_summary(summary, global_t)
            losses.append(loss)
            global_t += 1
            local_t += 1
            total_word_num += np.sum(decoder_len-np.array(1))  # since we remove GO for prediction
            if local_t % (train_feed.num_batch / 50) == 0:
                train_loss = np.sum(losses) / total_word_num
                print("%.2f train loss %f perplexity %f" %
                      (local_t / float(train_feed.num_batch), float(train_loss), np.exp(train_loss)))
        end_time = time.time()
        train_loss = np.sum(losses) / total_word_num
        print("Train loss %f perplexity %f and step %f"
              % (float(train_loss), np.exp(train_loss), (end_time-start_time)/float(local_t)))

        return global_t, train_loss

    def valid(self, name, sess, valid_feed):
        """
        No training is involved. Just forward path and compute the metrics
        :param name: the name, usually TEST or VALID
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

            fetches = [self.loss_sum]
            feed_dict = {self.encoder_batch: encoder_x, self.decoder_batch: decoder_y, self.encoder_lens: encoder_len}
            loss = sess.run(fetches, feed_dict)
            total_word_num += np.sum(decoder_len-np.array(1)) # since we remove GO for prediction
            losses.append(loss)

        # print final stats
        valid_loss = float(np.sum(losses) / total_word_num)
        print("%s loss %f and perplexity %f" % (name, valid_loss, np.exp(valid_loss)))
        return valid_loss

    def test(self, name, sess, test_feed, num_batch=None):
        """
              No training is involved. Just forward path and compute the metrics
              :param name: the name, usually TEST or VALID
              :param sess: the tf session
              :param test_feed: the data feed
              :param num_batch: if None run for the whole feed. Otherwise stop when num_batch is reached
              :return: average loss
              """
        local_t = 0
        predictions = []
        labels = []

        while True:
            batch = test_feed.next_batch()
            if batch is None or (num_batch is not None and local_t > num_batch):
                break

            encoder_len, decoder_len, encoder_x, decoder_y = batch
            fetches = [self.logits]
            feed_dict = {self.encoder_batch: encoder_x, self.decoder_batch: decoder_y, self.encoder_lens: encoder_len}
            logits = sess.run(fetches, feed_dict)
            max_ids = np.squeeze(np.array(logits), axis=0)
            max_ids = np.argmax(np.transpose(max_ids, [1,0,2]), axis=2)
            local_t += 1

            for b_id in range(test_feed.batch_size):
                sent_ids = max_ids[b_id]
                label_ids = decoder_y[b_id, 0:decoder_len[b_id]-1]

                first_eos = np.argwhere(sent_ids == test_feed.EOS_ID)
                if len(first_eos) > 0:
                    sent_ids = sent_ids[0:first_eos[0][0]]
                pred = " ".join([test_feed.rev_vocab[w_id] for w_id in sent_ids])
                label = " ".join([test_feed.rev_vocab[w_id] for w_id in label_ids])
                print("LABEL >> %s ||| MODEL >> %s" % (label, pred))
                predictions.append(pred)
                labels.append(label)
        return predictions, labels

