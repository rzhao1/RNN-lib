import os
import time
from beeprint import pp
import numpy as np
import tensorflow as tf
from data_utils.split_data import UttSeqCorpus
from data_utils.data_feed import UttSeqDataFeed
from models.clause_seq2seq import Utt2Seq
from config_utils import Utt2SeqConfig

# constants
tf.app.flags.DEFINE_string("data_dir", "Data", "the dir that has the raw corpus file")
tf.app.flags.DEFINE_string("data_file", "combine_result_orson.txt", "the file that contains the raw data")
tf.app.flags.DEFINE_string("work_dir", "seq_working/", "Experiment results directory.")
tf.app.flags.DEFINE_string("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_string("max_vocab_size", 30000, "The top N vocabulary we use.")
tf.app.flags.DEFINE_string("max_enc_len", 5, "The largest number of utterance in encoder")
tf.app.flags.DEFINE_string("max_dec_len", 13, "The largest number of words in decoder")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_bool("forward", False, "Do decoding only")

FLAGS = tf.app.flags.FLAGS


def main():
    # load corpus
    api = UttSeqCorpus(FLAGS.data_dir, FLAGS.data_file, [7,1,2], FLAGS.max_vocab_size,
                       FLAGS.max_enc_len, FLAGS.max_dec_len)
    corpus_data = api.get_corpus()

    # convert to numeric input outputs that fits into TF models
    train_feed = UttSeqDataFeed("Train", corpus_data["train"], api.vocab)
    valid_feed = UttSeqDataFeed("Valid", corpus_data["valid"], api.vocab)
    test_feed = UttSeqDataFeed("Test", corpus_data["test"], api.vocab)

    if not os.path.exists(FLAGS.work_dir):
        os.mkdir(FLAGS.work_dir)

    log_dir = os.path.join(FLAGS.work_dir, "run" + str(int(time.time())))
    os.mkdir(log_dir)
    config = Utt2SeqConfig()

    # for perplexity evaluation
    valid_config = Utt2SeqConfig()
    valid_config.keep_prob = 1.0
    valid_config.batch_size = 200

    # for forward only decoding
    test_config = Utt2SeqConfig()
    test_config.keep_prob = 1.0
    test_config.batch_size = 10
    pp(config)

    # begin training
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1*config.init_w, config.init_w)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = Utt2Seq(sess, config, vocab_size=len(train_feed.vocab), feature_size=train_feed.feat_size,
                            max_decoder_size=train_feed.max_dec_size, log_dir=log_dir, forward=False)

        # for evaluation perplexity on VALID and TEST set
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_model = Utt2Seq(sess, valid_config, len(train_feed.vocab), train_feed.feat_size,
                                  train_feed.max_dec_size, None, False)
        # get a random batch and do forward decoding. Print the most likely response
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            test_model = Utt2Seq(sess, valid_config, len(train_feed.vocab), train_feed.feat_size,
                                  train_feed.max_dec_size, None, True)

        ckp_dir = os.path.join(log_dir, "checkpoints")

        global_t = 0
        patience = 10  # wait for at least 10 epoch before consider early stop
        valid_loss_threshold = np.inf
        best_valid_loss = np.inf
        checkpoint_path = os.path.join(ckp_dir, "utt2seq.ckpt")

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

            # do sampling to see what kind of sentences is generated
            test_feed.epoch_init(test_config.batch_size, shuffle=True)
            test_model.test("TEST", sess, test_feed, 10)

            train_feed.epoch_init(config.batch_size, shuffle=True)
            global_t, train_loss = model.train(global_t, sess, train_feed)

            # begin validation
            valid_feed.epoch_init(valid_config.batch_size, shuffle=False)
            valid_loss = valid_model.valid("VALID", sess, valid_feed)

            test_feed.epoch_init(valid_config.batch_size, shuffle=False)
            valid_model.valid("TEST", sess, test_feed)

            # do sampling to see what kind of sentences is generated
            test_feed.epoch_init(test_config.batch_size, shuffle=True)
            test_model.test("TEST", sess, test_feed, 10)

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




















