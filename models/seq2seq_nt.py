import os
import time
from beeprint import pp
import numpy as np
import tensorflow as tf
import math
from data_utils.split_data import WordSeqCorpus
from data_utils.data_feed import WordSeqDataFeed
from tensorflow.python.ops import embedding_ops,rnn_cell,rnn

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

class seq2seq(object):
    def __init__(self, sess, config, vocab_size,  log_dir):
        self.batch_size = batch_size = config.batch_size
        self.utt_cell_size = utt_cell_size = config.cell_size
        self.vocab_size = vocab_size
        self.encoder_batch =encoder_batch= tf.placeholder(dtype=tf.int32,shape=(None,None),name="encoder_seq")
        self.decoder_batch =decoder_batch=tf.placeholder(dtype=tf.int32,shape=(None,None),name="decoder_seq")
        self.encoder_lens = encoder_lens = tf.placeholder(dtype=tf.int32,shape=(None),name="encoder_lens")
        self.decoder_lens = decoder_lens = tf.placeholder(dtype=tf.int32,shape=(None),name="decoder_lens")
        self.learning_rate = tf.Variable(float(config.init_lr), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * config.lr_decay)

        max_encode_sent_len = array_ops.shape(self.encoder_batch)[1]
        max_decode_sent_len = array_ops.shape(self.decoder_batch)[1]
        with variable_scope.variable_scope("word-embedding"):
            embedding = tf.get_variable("embedding", [vocab_size, config.embed_size], dtype=tf.float32)
            encoder_embedding = embedding_ops.embedding_lookup(embedding, tf.squeeze(tf.reshape(self.encoder_batch, [-1, 1]),
																				   squeeze_dims=[1]))
            encoder_embedding = tf.reshape(encoder_embedding, [-1, max_encode_sent_len, config.embed_size])
            decoder_embedding = embedding_ops.embedding_lookup(embedding, tf.squeeze(tf.reshape(self.decoder_batch, [-1, 1]),
																					 squeeze_dims=[1]))
            decoder_embedding = tf.reshape(decoder_embedding, [-1, max_decode_sent_len, config.embed_size])

        with tf.variable_scope('seqToseq') as scope:
            with tf.variable_scope('enc'):
                gru_cell_enc = tf.nn.rnn_cell.GRUCell(utt_cell_size)
                _,encoder_state = rnn.dynamic_rnn(gru_cell_enc,encoder_embedding,sequence_length=encoder_lens,dtype = tf.float32)

            with tf.variable_scope('dec'):
                with tf.variable_scope('gru_dec'):
                    gru_cell_dec = rnn_cell.GRUCell(utt_cell_size)
                    output,_ = rnn.dynamic_rnn(gru_cell_dec,decoder_embedding,initial_state = encoder_state,
												 sequence_length = decoder_lens,dtype=tf.float32)
		W = tf.get_variable('linear_W',[vocab_size,utt_cell_size],dtype = tf.float32)
		b = tf.get_variable('linear_b',[vocab_size],dtype= tf.float32)

		#This part adapted from http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

		output = output[:,1:,:]
		self.logits_flat = tf.matmul(tf.reshape(output,[-1,utt_cell_size]),tf.transpose(W)) + b
		#logits_flat=logits_flat[1:decoder_lens-1,-1]
		self.labels_flat = tf.reshape(decoder_batch[:,1:],[-1,1])
		weights = tf.sign(self.labels_flat)
		weights = tf.cast(weights,tf.float32)
                tvars = tf.trainable_variables()
                for v in tvars:
                    print("Trainable %s" % v.name)
		self.losses = tf.nn.seq2seq.sequence_loss_by_example([self.logits_flat], [self.labels_flat],[weights],average_across_timesteps=False)
		#masked_losses = tf.reshape(tf.sign(tf.to_float(labels_flat)) * losses,tf.shape(decoder_batch))
		#mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.cast(decoder_lens,tf.float32)
		self.mean_loss = tf.reduce_sum(self.losses)/tf.cast(batch_size,tf.float32)
		tf.scalar_summary('summary/batch_loss' , self.mean_loss)
		optim = tf.train.AdamOptimizer(self.learning_rate)
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), config.grad_clip)
                self.train_ops = optim.apply_gradients(zip(grads, tvars))
                self.saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V1)
        self.merged = tf.merge_all_summaries()



    def train(self,global_t,sess,train_feed):
        losses = []
        local_t = 0
        total_word_num=0
        while True:
            batch = train_feed.next_batch()
            if batch is None:
               break
            encoder_len,decoder_len,encoder_x,decoder_y = batch

            fetches =[self.train_ops, self.mean_loss, self.merged,self.losses,self.labels_flat,self.logits_flat]
            feed_dict={self.encoder_batch:encoder_x,self.decoder_batch:decoder_y,self.encoder_lens:encoder_len,self.decoder_lens:decoder_len}
            _, loss, summary, med_loss, labels, logits = sess.run(fetches,feed_dict)
            losses.append(loss)
            if math.isnan(loss):
                print (labels)
                print (logits)
            global_t += 1
            local_t += 1
            total_word_num += np.sum(decoder_len)
            if local_t % (train_feed.num_batch / 10) == 0:
                print (np.sum(losses))
                train_loss = np.sum(losses) / total_word_num * train_feed.batch_size
                print("%.2f train loss %f" % (local_t / float(train_feed.num_batch), float(train_loss)))

        return global_t, losses



    def valid(self, name, sess, valid_feed):
        """
        No training is involved. Just forward path and compute the metrics
        :param name: the name, ususally TEST or VALID
        :param sess: the tf session
        :param valid_feed: the data feed
        :return: average loss
        """
        losses = []
        local_t = 0
        total_word_num = 0

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            encoder_len, decoder_len, encoder_x, decoder_y = batch
            fetches = [self.train_ops, self.mean_loss, self.merged,self.losses,self.labels_flat,self.logits_flat]
            feed_dict = {self.encoder_batch: encoder_x, self.decoder_batch: decoder_y, self.encoder_lens: encoder_len,
                         self.decoder_lens: decoder_len}
            _, loss, summary,med_loss,labels,logits = sess.run(fetches, feed_dict)
            print (labels)
            #print(logits)
            losses.append(loss)


        # print final stats
        print("%s loss %f" % (name, float(np.sum(losses) / total_word_num * valid_feed.batch_size)))
        return losses
