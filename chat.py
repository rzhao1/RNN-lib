import os
import sys
import time
from beeprint import pp
import numpy as np
import tensorflow as tf
from data_utils.split_data import WordSeqCorpus
from data_utils.data_feed import WordSeqDataFeed
from models.seq2seq_nt import Word2Seq


# constants
tf.app.flags.DEFINE_string("data_dir", "Data/", "the dir that has the raw corpus file")
tf.app.flags.DEFINE_string("data_file", "clean_data_ran.txt", "the file that contains the raw data")
tf.app.flags.DEFINE_string("work_dir", "seq_working/", "Experiment results directory.")
tf.app.flags.DEFINE_string("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_string("max_vocab_size", 30000, "The top N vocabulary we use.")
tf.app.flags.DEFINE_string("max_enc_len", 100, "The largest number of words in encoder")
tf.app.flags.DEFINE_string("max_dec_len", 50, "The largest number of words in decoder")

FLAGS = tf.app.flags.FLAGS


class Config(object):
    op = "sgd"
    cell_type = "gru"

    use_attention = True

    # general config
    grad_clip = 5.0
    init_w = 0.05
    batch_size = 1
    embed_size = 150
    cell_size = 300
    num_layer = 1
    max_epoch = 1
    line_thres = 2
    max_decoder_size = 15

    # SGD training related
    init_lr = 0.6
    lr_hold = 1
    lr_decay = 0.6
    keep_prob = 1.0
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True



def chat():
    with tf.Session() as sess:
        PAD_ID = 0
        UNK_ID = 1
        GO_ID = 2
        EOS_ID = 3


        config = Config()
        # load corpus
        api = WordSeqCorpus(FLAGS.data_dir, FLAGS.data_file, [7, 1, 2], FLAGS.max_vocab_size,
                            FLAGS.max_enc_len, FLAGS.max_dec_len, Config.line_thres)
        corpus_data = api.get_corpus()
        train_feed = WordSeqDataFeed("Train", config, corpus_data["train"], api.vocab)

       #Construct model
        model = create_model(sess, config, len(train_feed.vocab),forward=True)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            predicted_sentence = model.get_predicted_sentence(sentence, train_feed.vocab, train_feed, sess, model)
            print("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def create_model(sess, config, vocab_size, log_dir=None,forward=False):

    model= Word2Seq(sess, config, vocab_size,forward=True)
    ckpt = tf.train.get_checkpoint_state(FLAGS.work_dir)
    if ckpt:
        print("Reading models parameters from %ss" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.initialize_all_variables())
    return model

if __name__ == "__main__":
    chat()