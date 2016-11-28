import os
import time
import numpy as np
from common_functions import *
from tensorflow.python.ops import nn_ops
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
import nltk.translate.bleu_score as bleu


class Word2Seq(object):

    def __init__(self, sess, config, vocab_size, eos_id, log_dir=None, forward=False):
        self.batch_size = config.batch_size
        self.cell_size = cell_size = config.cell_size
        self.vocab_size = vocab_size
        self.max_decoder_size = config.max_dec_len
        self.forward = forward
        self.beam_size = config.beam_size
        self.word_embed_size = config.embed_size
        self.is_lstm_cell = config.cell_type == "lstm"
        self.eos_id = eos_id

        self.encoder_batch = tf.placeholder(dtype=tf.int32, shape=(None, None), name="encoder_seq")
        # max_decoder_size including GO and EOS
        self.decoder_batch = tf.placeholder(dtype=tf.int32, shape=(None, config.max_decoder_size), name="decoder_seq")
        self.encoder_lens  = tf.placeholder(dtype=tf.int32, shape=(None), name="encoder_lens")

        # include GO sent and EOS
        self.learning_rate = tf.Variable(float(config.init_lr), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(tf.mul(self.learning_rate, config.lr_decay))

        max_encode_sent_len = array_ops.shape(self.encoder_batch)[1]
        # As our current version decoder is fixed-size,
        # the decoder is in format GO sent EOS. Mostly we either work with GO sent or sent EOS. that is max_decoder-1
        max_decode_minus_one = self.max_decoder_size - 1

        with variable_scope.variable_scope("word_embedding"):
            embedding = tf.get_variable("embedding", [vocab_size, config.embed_size], dtype=tf.float32)

            encoder_embedding = embedding_ops.embedding_lookup(embedding, tf.squeeze(tf.reshape(self.encoder_batch, [-1, 1]),
                                                                          squeeze_dims=[1]))
            encoder_embedding = tf.reshape(encoder_embedding, [-1, max_encode_sent_len, config.embed_size])

        with tf.variable_scope('seq2seq'):
            with tf.variable_scope('enc'):
                if config.cell_type == "gru":
                    cell_enc = tf.nn.rnn_cell.GRUCell(cell_size)
                elif config.cell_type == "lstm":
                    cell_enc = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
                else:
                    print("WARNING: unknown cell type. Use Basic RNN as default")
                    cell_enc = tf.nn.rnn_cell.BasicRNNCell(cell_size)

                if config.keep_prob < 1.0:
                    cell_enc = rnn_cell.DropoutWrapper(cell_enc, output_keep_prob=config.keep_prob,
                                                       input_keep_prob=config.keep_prob)

                if config.num_layer > 1:
                    cell_enc = rnn_cell.MultiRNNCell([cell_enc] * config.num_layer, state_is_tuple=True)

                encoder_outputs, encoder_last_state = rnn.dynamic_rnn(cell_enc, encoder_embedding,
                                                                      sequence_length=self.encoder_lens,
                                                        dtype=tf.float32)
                if config.num_layer > 1:
                    encoder_last_state = encoder_last_state[-1]

            # post process the decoder embedding inputs and encoder_last_state
            # Tony decoder embedding we need to remove the last column
            decoder_embedding, encoder_last_state = self.prepare_for_beam(embedding, encoder_last_state,
                                                                          self.decoder_batch[:,
                                                                          0:max_decode_minus_one])

            with tf.variable_scope('dec'):

                if config.cell_type == "gru":
                    cell_dec = tf.nn.rnn_cell.GRUCell(cell_size)
                elif config.cell_type == "lstm":
                    cell_dec = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
                else:
                    print("WARNING: unknown cell type. Use Basic RNN as default")
                    cell_dec = tf.nn.rnn_cell.BasicRNNCell(cell_size)

                # output project to vocabulary size
                cell_dec = tf.nn.rnn_cell.OutputProjectionWrapper(cell_dec, vocab_size)

                # No back propagation and forward only for testing
                # run decoder to get sequence outputs
                dec_outputs, beam_symbols, beam_path, log_beam_probs = self.beam_rnn_decoder(
                    decoder_embedding=decoder_embedding,
                    initial_state=encoder_last_state,
                    embedding=embedding,
                    cell=cell_dec)

                self.logits = dec_outputs
                self.beam_symbols = beam_symbols
                self.beam_path = beam_path
                self.log_beam_probs = log_beam_probs

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
                self.saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V1)

            if log_dir is not None:
                self.print_model_stats(tf.trainable_variables())
                train_log_dir = os.path.join(log_dir, "train")
                print("Save summary to %s" % log_dir)
                self.train_summary_writer = tf.train.SummaryWriter(train_log_dir, sess.graph)

    @staticmethod
    def print_model_stats(tvars):
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

    def beam_rnn_decoder(self, decoder_embedding, initial_state, cell, embedding, scope=None):
        """
        :param decoder_inputs: B * max_enc_len
        :param initial_state: B * cell_size
        :param cell:
        :param scope: the name scope
        :return: decoder_outputs, the last decoder_state
        """
        beam_symbols, beam_path, log_beam_probs = [], [], []
        loop_function = self.get_loop_function(embedding, self.vocab_size, beam_symbols, beam_path, log_beam_probs)
        outputs, state = rnn_decoder(decoder_embedding, initial_state, cell, loop_function=loop_function, scope=scope)
        return outputs, beam_symbols, beam_path, log_beam_probs

    def get_loop_function(self, embedding, num_symbol, beam_symbols, beam_path, log_beam_probs):
        if self.forward:
            loop_function = beam_and_embed(embedding, self.beam_size,
                                           num_symbol, beam_symbols,
                                           beam_path, log_beam_probs,
                                           self.eos_id)
        else:
            loop_function = None
        return loop_function

    def prepare_for_beam(self, embedding, initial_state, decoder_inputs):
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
                embed = tf.reshape(tf.tile(embed, [1, self.beam_size]), [-1, self.word_embed_size])
                emb_inp.append(embed)

            if type(initial_state) is tf.nn.seq2seq.rnn_cell.LSTMStateTuple:
                tile_c = tf.reshape(tf.tile(initial_state.c, [1, self.beam_size]), [-1, self.cell_size])
                tile_h = tf.reshape(tf.tile(initial_state.h, [1, self.beam_size]), [-1, self.cell_size])
                initial_state = tf.nn.seq2seq.rnn_cell.LSTMStateTuple(tile_c, tile_h)
            else:
                initial_state = tf.reshape(tf.tile(initial_state, [1, self.beam_size]), [-1, self.cell_size])
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
                print("%.2f train loss %f perleixty %f" %
                      (local_t / float(train_feed.num_batch), float(train_loss), np.exp(train_loss)))
        end_time = time.time()

        train_loss = np.sum(losses) / total_word_num
        print("Train loss %f perleixty %f with step %f"
              % (float(train_loss), np.exp(train_loss), (end_time-start_time)/float(local_t)))

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
        all_refs = []
        all_n_bests = [] # 2D list. List of List of N-best
        local_t = 0
        fetch = [self.logits, self.beam_symbols, self.beam_path, self.log_beam_probs]
        while True:
            batch = test_feed.next_batch()
            if batch is None or (num_batch is not None and local_t > num_batch):
                break
            encoder_len, decoder_len, encoder_x, decoder_y = batch
            feed_dict = {self.encoder_batch: encoder_x, self.decoder_batch: decoder_y, self.encoder_lens: encoder_len}
            pred, symbol, path, probs = sess.run(fetch, feed_dict)
            # [B*beam x vocab]*dec_len, [B*beam]*dec_len, [B*beam]*dec_len [B*beamx1]*dec_len
            # label: [B x max_dec_len+1]

            beam_symbols_matrix = np.array(symbol)
            beam_path_matrix = np.array(path)
            beam_log_matrix = np.array(probs)

            for b_idx in range(test_feed.batch_size):
                ref = list(decoder_y[b_idx, 1:])
                # remove padding and EOS symbol
                ref = [r for r in ref if r not in [test_feed.PAD_ID, test_feed.EOS_ID]]
                b_beam_symbol = beam_symbols_matrix[:, b_idx * self.beam_size:(b_idx + 1) * self.beam_size]
                b_beam_path = beam_path_matrix[:, b_idx * self.beam_size:(b_idx + 1) * self.beam_size]
                b_beam_log = beam_log_matrix[:, b_idx * self.beam_size:(b_idx + 1) * self.beam_size]
                # use lattic to find the N-best list
                n_best = get_n_best(b_beam_symbol, b_beam_path, b_beam_log, self.beam_size, test_feed.EOS_ID)

                all_refs.append(ref)
                all_n_bests.append(n_best)

            local_t += 1

        # get error
        return self.beam_error(all_refs, all_n_bests, name, test_feed.rev_vocab)

    def beam_error(self, all_refs, all_n_best, name, rev_vocab):
        all_bleu = []
        for ref, n_best in zip(all_refs, all_n_best):
            ref = [rev_vocab[word] for word in ref]
            local_bleu = []
            for score, best in n_best:
                best = [rev_vocab[word] for word in best]
                print("Label>> %s ||| Hyp>> %s" % (" ".join(ref), " ".join(best)))
                try:
                    local_bleu.append(bleu.sentence_bleu([ref], best))
                except ZeroDivisionError:
                    local_bleu.append(0.0)
            print("*"*20)
            all_bleu.append(local_bleu)

        # begin evaluation of @n
        reports = []
        for b in range(self.beam_size):
            avg_best_bleu = np.mean([np.max(local_bleu[0:b + 1]) for local_bleu in all_bleu])
            record = "%s@%d BLEU %f" % (name, b + 1, float(avg_best_bleu))
            reports.append(record)
            print(record)

        return reports


class Word2SeqAutoEncoder(object):

    def __init__(self, sess, config, vocab_size, eos_id, log_dir=None, forward=False):
        self.batch_size = config.batch_size
        self.cell_size = cell_size = config.cell_size
        self.vocab_size = vocab_size
        self.max_decoder_size = config.max_dec_len
        self.forward = forward
        self.beam_size = config.beam_size
        self.word_embed_size = config.embed_size
        self.is_lstm_cell = config.cell_type == "lstm"
        self.eos_id = eos_id

        self.encoder_batch = tf.placeholder(dtype=tf.int32, shape=(None, None), name="encoder_seq")
        # max_decoder_size including GO and EOS
        self.decoder_batch = tf.placeholder(dtype=tf.int32, shape=(None, self.max_decoder_size), name="decoder_seq")
        self.encoder_lens = tf.placeholder(dtype=tf.int32, shape=(None), name="encoder_lens")

        # include GO sent and EOS
        self.learning_rate = tf.Variable(float(config.init_lr), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(tf.mul(self.learning_rate, config.lr_decay))

        max_encode_sent_len = array_ops.shape(self.encoder_batch)[1]
        # As our current version decoder is fixed-size,
        # the decoder is in format GO sent EOS. Mostly we either work with GO sent or sent EOS. that is max_decoder-1
        max_decode_minus_one = self.max_decoder_size - 1

        with variable_scope.variable_scope("word_embedding"):
            embedding = tf.get_variable("embedding", [vocab_size, config.embed_size], dtype=tf.float32)
            encoder_embedding = embedding_ops.embedding_lookup(embedding, tf.squeeze(tf.reshape(self.encoder_batch, [-1, 1]), squeeze_dims=[1]))
            encoder_embedding = tf.reshape(encoder_embedding, [-1, max_encode_sent_len, config.embed_size])

        with tf.variable_scope('seq2seq'):

            with tf.variable_scope('enc'):
                if config.cell_type == "gru":
                    cell_enc = tf.nn.rnn_cell.GRUCell(cell_size)
                elif config.cell_type == "lstm":
                    cell_enc = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
                else:
                    print("WARNING: unknown cell type. Use Basic RNN as default")
                    cell_enc = tf.nn.rnn_cell.BasicRNNCell(cell_size)

                if config.keep_prob < 1.0:
                    encoder_embedding = tf.nn.dropout(encoder_embedding, keep_prob=config.keep_prob)
                    cell_enc = rnn_cell.DropoutWrapper(cell_enc, output_keep_prob=config.keep_prob)

                if config.num_layer > 1:
                    cell_enc = rnn_cell.MultiRNNCell([cell_enc] * config.num_layer, state_is_tuple=True)

                encoder_outputs, encoder_last_state = rnn.dynamic_rnn(cell_enc, encoder_embedding,
                                                                      sequence_length=self.encoder_lens,
                                                                      dtype=tf.float32)

            # post process the decoder embedding inputs and encoder_last_state
            # Tony decoder embedding we need to remove the last column
            decoder_embedding, decoder_init_state = self.prepare_for_beam(embedding, encoder_last_state,
                                                                          self.decoder_batch[:, 0:max_decode_minus_one])

            with tf.variable_scope('dec'):
                if config.cell_type == "gru":
                    cell_dec = tf.nn.rnn_cell.GRUCell(cell_size)
                elif config.cell_type == "lstm":
                    cell_dec = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
                else:
                    print("WARNING: unknown cell type. Use Basic RNN as default")
                    cell_dec = tf.nn.rnn_cell.BasicRNNCell(cell_size)

                # output project to vocabulary size
                cell_dec = tf.nn.rnn_cell.OutputProjectionWrapper(cell_dec, vocab_size)

                if config.keep_prob < 1.0:
                    cell_dec = rnn_cell.DropoutWrapper(cell_dec, output_keep_prob=config.keep_prob)

                if config.num_layer > 1:
                    cell_dec = rnn_cell.MultiRNNCell([cell_dec] * config.num_layer, state_is_tuple=True)

                # No back propagation and forward only for testing
                # run decoder to get sequence outputs
                dec_outputs, beam_symbols, beam_path, log_beam_probs = self.beam_rnn_decoder(
                    decoder_embedding=decoder_embedding,
                    initial_state=decoder_init_state,
                    embedding=embedding,
                    cell=cell_dec)

                self.decoder_logits = dec_outputs
                self.beam_symbols = beam_symbols
                self.beam_path = beam_path
                self.log_beam_probs = log_beam_probs

            with tf.variable_scope("reconstruction"):
                if config.cell_type == "gru":
                    cell_reconstruct = tf.nn.rnn_cell.GRUCell(cell_size)
                elif config.cell_type == "lstm":
                    cell_reconstruct = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
                else:
                    print("WARNING: unknown cell type. Use Basic RNN as default")
                    cell_reconstruct = tf.nn.rnn_cell.BasicRNNCell(cell_size)

                # output project to vocabulary size
                cell_reconstruct = tf.nn.rnn_cell.OutputProjectionWrapper(cell_reconstruct, vocab_size)
                # No back propagation and forward only for testing
                reconstruct_embedding = encoder_embedding[:, 0:-1, :]
                    #tf.slice(encoder_embedding, [0, 0, 0], [-1, max_encode_sent_len-1, config.embed_size])

                if config.keep_prob < 1.0:
                    reconstruct_embedding = tf.nn.dropout(reconstruct_embedding, keep_prob=config.keep_prob)
                    cell_reconstruct = rnn_cell.DropoutWrapper(cell_reconstruct, output_keep_prob=config.keep_prob)

                if config.num_layer > 1:
                    cell_reconstruct = rnn_cell.MultiRNNCell([cell_reconstruct] * config.num_layer, state_is_tuple=True)

                self.reconstruct_logits, _ = rnn.dynamic_rnn(cell_reconstruct, reconstruct_embedding,
                                                             sequence_length=self.encoder_lens-1,
                                                             initial_state=encoder_last_state,
                                                             dtype=tf.float32)

            with tf.variable_scope("loss_calculation"):
                # There are two loss. The decoder loss and reconstruction loss
                decoder_logits = tf.pack(dec_outputs, 1)
                decoder_labels = tf.slice(self.decoder_batch, [0, 1], [-1, -1])
                decoder_weights = tf.to_float(tf.sign(decoder_labels))

                reconstruct_labels = tf.slice(self.encoder_batch, [0, 1], [-1, -1])
                reconstruct_weights = tf.to_float(tf.sign(reconstruct_labels))

                decoder_loss = nn_ops.sparse_softmax_cross_entropy_with_logits(decoder_logits, decoder_labels)
                reconstruct_loss = nn_ops.sparse_softmax_cross_entropy_with_logits(self.reconstruct_logits, reconstruct_labels)

                # mask out invalid positions
                decoder_loss *= decoder_weights
                reconstruct_loss *= reconstruct_weights

                # get final losses
                self.decoder_loss_sum = tf.reduce_sum(decoder_loss)
                self.reconstruct_loss_sum = tf.reduce_sum(reconstruct_loss)
                self.decoder_loss_avg = self.decoder_loss_sum / tf.reduce_sum(decoder_weights)
                self.reconstruct_loss_avg = self.reconstruct_loss_sum / tf.reduce_sum(reconstruct_weights)
                combine_loss = self.decoder_loss_avg + self.reconstruct_loss_avg

            if log_dir is not None:
                # choose a optimizer
                if config.op == "adam":
                    optim = tf.train.AdamOptimizer(self.learning_rate)
                elif config.op == "rmsprop":
                    optim = tf.train.RMSPropOptimizer(self.learning_rate)
                else:
                    optim = tf.train.GradientDescentOptimizer(self.learning_rate)

                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(combine_loss, tvars), config.grad_clip)
                self.train_ops = optim.apply_gradients(zip(grads, tvars))
                self.saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V1)

                self.print_model_stats(tf.trainable_variables())
                train_log_dir = os.path.join(log_dir, "train")
                print("Save summary to %s" % log_dir)
                self.train_summary_writer = tf.train.SummaryWriter(train_log_dir, sess.graph)

    @staticmethod
    def print_model_stats(tvars):
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

    def beam_rnn_decoder(self, decoder_embedding, initial_state, cell, embedding, scope=None):
        """
        :param decoder_inputs: B * max_enc_len
        :param initial_state: B * cell_size
        :param cell:
        :param scope: the name scope
        :return: decoder_outputs, the last decoder_state
        """
        beam_symbols, beam_path, log_beam_probs = [], [], []
        loop_function = self.get_loop_function(embedding, self.vocab_size, beam_symbols, beam_path, log_beam_probs)
        outputs, state = rnn_decoder(decoder_embedding, initial_state, cell, loop_function=loop_function, scope=scope)
        return outputs, beam_symbols, beam_path, log_beam_probs

    def get_loop_function(self, embedding, num_symbol, beam_symbols, beam_path, log_beam_probs):
        if self.forward:
            loop_function = beam_and_embed(embedding, self.beam_size,
                                           num_symbol, beam_symbols,
                                           beam_path, log_beam_probs,
                                           self.eos_id)
        else:
            loop_function = None
        return loop_function

    def prepare_for_beam(self, embedding, initial_state, decoder_inputs):
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
                embed = tf.reshape(tf.tile(embed, [1, self.beam_size]), [-1, self.word_embed_size])
                emb_inp.append(embed)

            if type(initial_state) is tf.nn.seq2seq.rnn_cell.LSTMStateTuple:
                tile_c = tf.reshape(tf.tile(initial_state.c, [1, self.beam_size]), [-1, self.cell_size])
                tile_h = tf.reshape(tf.tile(initial_state.h, [1, self.beam_size]), [-1, self.cell_size])
                initial_state = tf.nn.seq2seq.rnn_cell.LSTMStateTuple(tile_c, tile_h)
            else:
                initial_state = tf.reshape(tf.tile(initial_state, [1, self.beam_size]), [-1, self.cell_size])
        else:
            emb_inp = [embedding_ops.embedding_lookup(embedding, decoder_inputs[:, i]) for i in range(seq_len)]

        return emb_inp, initial_state

    def train(self, global_t, sess, train_feed):
        decoder_losses = []
        reconstruct_losses = []

        local_t = 0
        start_time = time.time()
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            encoder_len, decoder_len, encoder_x, decoder_y = batch

            fetches = [self.train_ops, self.decoder_loss_avg, self.reconstruct_loss_avg]
            feed_dict = {self.encoder_batch: encoder_x, self.decoder_batch: decoder_y, self.encoder_lens: encoder_len}
            _, decoder_loss, reconstruct_loss = sess.run(fetches, feed_dict)

            decoder_losses.append(decoder_loss)
            reconstruct_losses.append(reconstruct_loss)

            global_t += 1
            local_t += 1
            if local_t % (train_feed.num_batch / 50) == 0:
                train_dec_loss = np.mean(decoder_losses)
                train_reconstruct_loss = np.mean(reconstruct_losses)
                print("%.2f train decoder loss %f perleixty %f :: train reconstruct loss %f perleixty %f" %
                      (local_t / float(train_feed.num_batch), float(train_dec_loss), np.exp(train_dec_loss),
                       float(train_reconstruct_loss), np.exp(train_reconstruct_loss)))
        end_time = time.time()

        train_dec_loss = np.mean(decoder_losses)
        train_reconstruct_loss = np.mean(reconstruct_losses)

        print("Train decoder loss %f perleixty %f :: train reconstruct loss %f perleixty %f with time %f"
              % ( float(train_dec_loss), np.exp(train_dec_loss), float(train_reconstruct_loss),
                  np.exp(train_reconstruct_loss), (end_time-start_time)/float(local_t)))

        return global_t, train_dec_loss

    def valid(self, name, sess, valid_feed):
        """
        No training is involved. Just forward path and compute the metrics
        :param name: the name, ususally TEST or VALID
        :param sess: the tf session
        :param valid_feed: the data feed
        :return: average loss
        """
        decoder_losses = []
        reconstruct_losses = []

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break

            encoder_len, decoder_len, encoder_x, decoder_y = batch
            fetches = [self.decoder_loss_avg, self.reconstruct_loss_avg]
            feed_dict = {self.encoder_batch: encoder_x, self.decoder_batch: decoder_y, self.encoder_lens: encoder_len}
            decoder_loss, reconstruct_loss = sess.run(fetches, feed_dict)
            decoder_losses.append(decoder_loss)
            reconstruct_losses.append(reconstruct_loss)

        # print final stats
        valid_dec_loss = np.mean(decoder_losses)
        valid_reconstruct_loss = np.mean(reconstruct_losses)

        print("Valid decoder loss %f perleixty %f :: valid reconstruct loss %f perleixty %f"
              % (float(valid_dec_loss), np.exp(valid_dec_loss),
                 float(valid_reconstruct_loss), np.exp(valid_reconstruct_loss)))
        return valid_dec_loss

    def test(self, name, sess, test_feed, num_batch=None):
        all_refs = []
        all_n_bests = [] # 2D list. List of List of N-best
        local_t = 0
        fetch = [self.decoder_logits, self.beam_symbols, self.beam_path, self.log_beam_probs]
        while True:
            batch = test_feed.next_batch()
            if batch is None or (num_batch is not None and local_t > num_batch):
                break
            encoder_len, decoder_len, encoder_x, decoder_y = batch
            feed_dict = {self.encoder_batch: encoder_x, self.decoder_batch: decoder_y, self.encoder_lens: encoder_len}
            pred, symbol, path, probs = sess.run(fetch, feed_dict)
            # [B*beam x vocab]*dec_len, [B*beam]*dec_len, [B*beam]*dec_len [B*beamx1]*dec_len
            # label: [B x max_dec_len+1]

            beam_symbols_matrix = np.array(symbol)
            beam_path_matrix = np.array(path)
            beam_log_matrix = np.array(probs)

            for b_idx in range(test_feed.batch_size):
                ref = list(decoder_y[b_idx, 1:])
                # remove padding and EOS symbol
                ref = [r for r in ref if r not in [test_feed.PAD_ID, test_feed.EOS_ID]]
                b_beam_symbol = beam_symbols_matrix[:, b_idx * self.beam_size:(b_idx + 1) * self.beam_size]
                b_beam_path = beam_path_matrix[:, b_idx * self.beam_size:(b_idx + 1) * self.beam_size]
                b_beam_log = beam_log_matrix[:, b_idx * self.beam_size:(b_idx + 1) * self.beam_size]
                # use lattic to find the N-best list
                n_best = get_n_best(b_beam_symbol, b_beam_path, b_beam_log, self.beam_size, test_feed.EOS_ID)

                all_refs.append(ref)
                all_n_bests.append(n_best)

            local_t += 1

        # get error
        return self.beam_error(all_refs, all_n_bests, name, test_feed.rev_vocab)

    def beam_error(self, all_refs, all_n_best, name, rev_vocab):
        all_bleu = []
        for ref, n_best in zip(all_refs, all_n_best):
            ref = [rev_vocab[word] for word in ref]
            local_bleu = []
            for score, best in n_best:
                best = [rev_vocab[word] for word in best]
                print("Label>> %s ||| Hyp>> %s" % (" ".join(ref), " ".join(best)))
                try:
                    local_bleu.append(bleu.sentence_bleu([ref], best))
                except ZeroDivisionError:
                    local_bleu.append(0.0)
            print("*"*20)
            all_bleu.append(local_bleu)

        # begin evaluation of @n
        reports = []
        for b in range(self.beam_size):
            avg_best_bleu = np.mean([np.max(local_bleu[0:b + 1]) for local_bleu in all_bleu])
            record = "%s@%d BLEU %f" % (name, b + 1, float(avg_best_bleu))
            reports.append(record)
            print(record)

        return reports