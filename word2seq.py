import os
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
tf.app.flags.DEFINE_string("max_enc_len", 50, "The largest number of words in encoder")
tf.app.flags.DEFINE_string("max_dec_len", 50, "The largest number of words in decoder")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_bool("forward", False, "Do decoding only")

FLAGS = tf.app.flags.FLAGS


class Config(object):
    op = "sgd"
    cell_type = "gru"

    use_attention = True

    # general config
    grad_clip = 5.0
    init_w = 0.05
    batch_size = 30
    embed_size = 150
    cell_size = 1000
    num_layer = 1
    max_epoch = 1
    line_thres =2
    max_decoder_size=15

    # SGD training related
    init_lr = 0.6
    lr_hold = 1
    lr_decay = 0.6
    keep_prob = 1.0
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True


def main():
    # load corpus
    api = WordSeqCorpus(FLAGS.data_dir, FLAGS.data_file, [7,1,2], FLAGS.max_vocab_size,
                        FLAGS.max_enc_len, FLAGS.max_dec_len, Config.line_thres)
    corpus_data = api.get_corpus()

    # Load configuration
    config = Config()
    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.batch_size = 20
    pp(config)

    # convert to numeric input outputs that fits into TF models
    train_feed = WordSeqDataFeed("Train", config,corpus_data["train"], api.vocab)
    valid_feed = WordSeqDataFeed("Valid", config,corpus_data["valid"], api.vocab)
    test_feed = WordSeqDataFeed("Test", config,corpus_data["test"], api.vocab)

    if not os.path.exists(FLAGS.work_dir):
        os.mkdir(FLAGS.work_dir)

    log_dir = os.path.join(FLAGS.work_dir, "run" + str(int(time.time())))
    os.mkdir(log_dir)


    # begin training
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1*config.init_w, config.init_w)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = Word2Seq(sess, config, len(train_feed.vocab), log_dir, forward=FLAGS.forward)

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            test_model = Word2Seq(sess, test_config, len(train_feed.vocab), None, forward=FLAGS.forward)

        ckp_dir = os.path.join(log_dir, "checkpoints")

        global_t = 0
        patience = 10  # wait for at least 10 epoch before consider early stop
        valid_loss_threshold = np.inf
        best_valid_loss = np.inf
        checkpoint_path = os.path.join(ckp_dir, "word2seq.ckpt")

        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)

        ckpt = tf.train.get_checkpoint_state(ckp_dir)

        if ckpt:
            print("Reading models parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created models with fresh parameters.")
            sess.run(tf.initialize_all_variables())

        for epoch in range(config.max_epoch):
            print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))

            train_feed.epoch_init(config.batch_size, shuffle=True)
            global_t, train_loss = model.train(global_t, sess, train_feed)

            # begin validation
            valid_feed.epoch_init(test_config.batch_size, shuffle=False)
            valid_loss = test_model.valid("VALID", sess, valid_feed)

            test_feed.epoch_init(test_config.batch_size, shuffle=False)
            test_model.valid("TEST", sess, test_feed)

            done_epoch = epoch +1

            # only save a models if the dev loss is smaller
            # Decrease learning rate if no improvement was seen over last 3 times.
            if config.op == "sgd" and done_epoch > config.lr_hold:
                sess.run(model.learning_rate_decay_op)

            if valid_loss < best_valid_loss:
                if valid_loss <= valid_loss_threshold * config.improve_threshold:
                    patience = max(patience, done_epoch * config.patient_increase)
                    valid_loss_threshold = valid_loss

                # still save the best train model
                if FLAGS.save_model:
                    print("Saving model!")
                    model.saver.save(sess, checkpoint_path, global_step=epoch)
                best_valid_loss = valid_loss

            if config.early_stop and patience <= done_epoch:
                print("!!Early stop due to run out of patience!!")
                break

        print("Best valid loss %f and perpleixyt %f" % (best_valid_loss, np.exp(best_valid_loss)))
        print("Done training")

if __name__ == "__main__":
    main()













