import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import os
import numpy as np

from tensorflow.python.ops.nn import rnn_cell


class MultiTaskSeqClassifier(object):
    """
    This models treat LM as a policy problem, where you make one step prediciton given the sent state
    """
    def __init__(self, sess, config, data_feed, log_dir):

        vocab_size = len(data_feed.vocab)

        with tf.name_scope("io"):
            self.inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name="input_seq")
            self.input_lens = tf.placeholder(dtype=tf.int32, shape=(None, ), name="seq_len")
            self.da_labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="dialog_acts")
            self.senti_labels = tf.placeholder(dtype=tf.float32, shape=(None, data_feed.feature_size[data_feed.SENTI_ID]),
                                               name="sentiments")

            self.learning_rate = tf.Variable(float(config.init_lr), trainable=False)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * config.lr_decay)

        max_sent_len = array_ops.shape(self.inputs)[1]
        batch_size =  array_ops.shape(self.inputs)[0]

        with variable_scope.variable_scope("word-embedding"):
            embedding = tf.get_variable("embedding", [vocab_size, config.embed_size], dtype=tf.float32)
            input_embedding = embedding_ops.embedding_lookup(embedding, tf.squeeze(tf.reshape(self.inputs, [-1, 1]),
                                                                                   squeeze_dims=[1]))

            input_embedding = tf.reshape(input_embedding, [-1, max_sent_len, config.embed_size])

        with variable_scope.variable_scope("rnn"):
            if config.cell_type == "gru":
                cell = rnn_cell.GRUCell(config.cell_size)
            elif config.cell_type == "lstm":
                cell = rnn_cell.LSTMCell(config.cell_size, use_peepholes=False, forget_bias=1.0)
            elif config.cell_type == "rnn":
                cell = rnn_cell.BasicRNNCell(config.cell_size)
            else:
                raise ValueError("unknown RNN type")

            if config.keep_prob < 1.0:
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=config.keep_prob, input_keep_prob=config.keep_prob)

            if config.num_layer > 1:
                cell = rnn_cell.MultiRNNCell([cell] * config.num_layer, state_is_tuple=True)

            # and enc_last_state will be same as the true last state
            outputs, _ = tf.nn.dynamic_rnn(
                cell,
                input_embedding,
                dtype=tf.float32,
                sequence_length=self.input_lens,
            )
            # get the TRUE last outputs
            last_outputs = tf.reduce_sum(tf.mul(outputs, tf.expand_dims(tf.one_hot(self.input_lens - 1, max_sent_len), -1)), 1)

            self.dialog_acts = self.fnn(last_outputs, data_feed.feature_size[data_feed.DA_ID], [100], "dialog_act_fnn")
            self.sentiments = self.fnn(last_outputs, data_feed.feature_size[data_feed.SENTI_ID], [100], "setiment_fnn")

        self.loss = tf.reduce_sum(nn_ops.sparse_softmax_cross_entropy_with_logits(self.dialog_acts, self.da_labels)) \
                    + tf.reduce_sum(nn_ops.softmax_cross_entropy_with_logits(self.sentiments, self.senti_labels))
        self.loss /= tf.to_float(batch_size)

        tf.scalar_summary("entropy_loss", self.loss)
        self.summary_op = tf.merge_all_summaries()

        # weight decay
        tvars = tf.trainable_variables()
        for v in tvars:
            print("Trainable %s" % v.name)
        # optimization
        if config.op == "adam":
            print("Use Adam")
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif config.op == "rmsprop":
            print("Use RMSProp")
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            print("Use SGD")
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.grad_clip)
        self.train_ops = optimizer.apply_gradients(zip(grads, tvars))
        self.saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V2)

        if log_dir is not None:
            train_log_dir = os.path.join(log_dir, "train")
            print("Save summary to %s" % log_dir)
            self.train_summary_writer = tf.train.SummaryWriter(train_log_dir, sess.graph)

    def fnn(self, inputs, output_size, hidden_sizes, scope, activation=tf.tanh):
        """
        A short hand for creating MLP
        :param inputs: 2D input tensor
        :param output_size: a scalar of the output size
        :param hidden_sizes: a list of hidden layer
        :param activation:the activation function
        :return: the final outputs
        """
        with variable_scope.variable_scope(scope):
            prev_inputs = inputs
            for idx, h_size in enumerate(hidden_sizes):
                w = tf.get_variable("W-%d" % idx, [prev_inputs.get_shape()[1], h_size], dtype=tf.float32)
                b = tf.get_variable("Bias-%d" % idx, [h_size], dtype=tf.float32, initializer=tf.zeros_initializer)
                outputs = activation(tf.matmul(prev_inputs, w) + b)
                prev_inputs = outputs
            w = tf.get_variable("OutputW", [prev_inputs.get_shape()[1], output_size], dtype=tf.float32)
            b = tf.get_variable("OutputBias", [output_size], dtype=tf.float32, initializer=tf.zeros_initializer)
            outputs = tf.matmul(prev_inputs, w) + b
            return outputs

    @staticmethod
    def metric(name, predictions, ground_truth, verbose=True):
        # compute various metrics and print them
        acc = float(np.average(predictions == ground_truth))
        if verbose:
            print("%s accuracy is %.3f" % (name, acc))
        return acc

    @staticmethod
    def _update_dict(prev_dict, new_input, key):
        new_array = prev_dict.get(key, None)
        if new_array is not None:
            new_array = np.vstack((new_array, new_input))
        else:
            new_array = new_input
        prev_dict[key] = new_array

    def train(self, global_t, sess, train_feed):
        """
        Run forward and backward path to train the model
        :param global_t: the global update step
        :param sess: tf session
        :param train_feed: data feed
        :return: average loss
        """

        losses = []
        predictions = dict()
        ground_truth = dict()
        local_t = 0

        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            inputs, input_lens, outputs = batch
            feed_dict = {self.inputs: inputs, self.input_lens: input_lens,
                         self.da_labels: outputs[train_feed.DA_ID],
                         self.senti_labels: outputs[train_feed.SENTI_ID]}
            fetch = [self.train_ops, self.loss, self.summary_op, self.dialog_acts, self.sentiments]

            _, loss, summary, da_pred, senti_pred = sess.run(fetch, feed_dict)
            self.train_summary_writer.add_summary(summary, global_t)

            # save statistics
            losses.append(loss)
            self._update_dict(predictions, np.argmax(da_pred, axis=1), train_feed.DA_ID)
            self._update_dict(ground_truth, outputs[train_feed.DA_ID], train_feed.DA_ID)

            self._update_dict(predictions, senti_pred, train_feed.SENTI_ID)
            self._update_dict(ground_truth, outputs[train_feed.SENTI_ID], train_feed.SENTI_ID)

            global_t += 1
            local_t += 1

            # print intermediate progress of training every 10%
            if local_t % (train_feed.num_batch / 10) == 0:
                print("%.2f train loss %f" % (local_t / float(train_feed.num_batch), float(np.mean(losses))))

        # print final metrics
        print("TRAIN loss %f" % (float(np.mean(losses))))
        self.metric("TRAIN", predictions, ground_truth)
        return global_t, float(np.mean(losses))

    def valid(self, name, sess, valid_feed):
        """
        No training is involved. Just forward path and compute the metrics
        :param name: the name, ususally TEST or VALID
        :param sess: the tf session
        :param valid_feed: the data feed
        :return: average loss
        """
        losses = []
        predictions = dict()
        ground_truth = dict()

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            inputs, input_lens, outputs = batch
            feed_dict = {self.inputs: inputs, self.input_lens: input_lens,
                         self.da_labels: outputs[valid_feed.DA_ID],
                         self.senti_labels: outputs[valid_feed.SENTI_ID]}
            fetch = [self.loss, self.dialog_acts, self.sentiments]

            loss, da_pred, senti_pred = sess.run(fetch, feed_dict)

            # save statistics
            losses.append(loss)
            self._update_dict(predictions, np.argmax(da_pred, axis=1), valid_feed.DA_ID)
            self._update_dict(ground_truth, outputs[valid_feed.DA_ID], valid_feed.DA_ID)

            self._update_dict(predictions, senti_pred, valid_feed.SENTI_ID)
            self._update_dict(ground_truth, outputs[valid_feed.SENTI_ID], valid_feed.SENTI_ID)

        # print final stats
        valid_loss = float(np.mean(losses))
        print("%s loss %f" % (name, valid_loss))
        self.metric(name, predictions, ground_truth)
        return valid_loss


