import os
import time
from beeprint import pp
import numpy as np
import tensorflow as tf
from data_utils.split_data import WordSeqCorpus
from data_utils.data_feed import WordSeqDataFeed
from models.word_seq2seq import Word2SeqAutoEncoder
from config_utils import Word2SeqAutoConfig as Config

# constants
tf.app.flags.DEFINE_string("data_dir", "Data/", "the dir that has the raw corpus file")
tf.app.flags.DEFINE_string("data_file", "clean_data_ran.txt", "the file that contains the raw data")
tf.app.flags.DEFINE_string("work_dir", "seq_working/", "Experiment results directory.")
tf.app.flags.DEFINE_string("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_string("max_vocab_size", 20000, "The top N vocabulary we use.")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_bool("forward", False, "Do decoding only")
tf.app.flags.DEFINE_string("test_path", "run1478720226", "the dir to load checkpoint for forward only")

FLAGS = tf.app.flags.FLAGS


def main():
    # Load configuration
    config = Config()
    # for perplexity evaluation
    valid_config = Config()
    valid_config.keep_prob = 1.0
    valid_config.batch_size = 200

    # for forward only decoding
    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.batch_size = 10
    pp(config)

    # load corpus
    api = WordSeqCorpus(FLAGS.data_dir, FLAGS.data_file, [98,1,1], FLAGS.max_vocab_size,
                        config.max_enc_len, config.max_dec_len, config.line_thres)
    corpus_data = api.get_corpus()

    # convert to numeric input outputs that fits into TF models
    train_feed = WordSeqDataFeed("Train", config,corpus_data["train"], api.vocab)
    valid_feed = WordSeqDataFeed("Valid", config,corpus_data["valid"], api.vocab)
    test_feed = WordSeqDataFeed("Test", config,corpus_data["test"], api.vocab)

    if not os.path.exists(FLAGS.work_dir):
        os.mkdir(FLAGS.work_dir)

    if FLAGS.forward:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.test_path)
    else:
        log_dir = os.path.join(FLAGS.work_dir, "run" + str(int(time.time())))
        os.mkdir(log_dir)

    # begin training
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1*config.init_w, config.init_w)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = Word2SeqAutoEncoder(sess, config, len(train_feed.vocab), train_feed.EOS_ID,
                                        log_dir=None if FLAGS.forward else log_dir, forward=False)

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_model = Word2SeqAutoEncoder(sess, valid_config, len(train_feed.vocab), train_feed.EOS_ID, None, forward=False)

        # get a random batch and do forward decoding. Print the most likely response
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            test_model = Word2SeqAutoEncoder(sess, test_config, len(train_feed.vocab), train_feed.EOS_ID, None, forward=True)

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

        if not FLAGS.forward:
            for epoch in range(config.max_epoch):
                print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))

                train_feed.epoch_init(config.batch_size, shuffle=True)
                global_t, train_loss = model.train(global_t, sess, train_feed)

                # begin validation
                valid_feed.epoch_init(valid_config.batch_size, shuffle=False)
                valid_loss = valid_model.valid("VALID", sess, valid_feed)

                test_feed.epoch_init(valid_config.batch_size, shuffle=False)
                valid_model.valid("TEST", sess, test_feed)

                # do sampling to see what kind of sentences is generated
                test_feed.epoch_init(test_config.batch_size, shuffle=True)
                test_model.test("TEST", sess, test_feed, num_batch=2)

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
        else:
            # do sampling to see what kind of sentences is generated
            test_feed.epoch_init(test_config.batch_size, shuffle=True)
            test_model.test("TEST", sess, test_feed, num_batch=2)

            # begin validation
            valid_feed.epoch_init(valid_config.batch_size, shuffle=False)
            valid_model.valid("VALID", sess, valid_feed)

            test_feed.epoch_init(valid_config.batch_size, shuffle=False)
            valid_model.valid("TEST", sess, test_feed)

if __name__ == "__main__":
    main()













