import os
import time
from beeprint import pp
import numpy as np
import tensorflow as tf
from data_utils.split_data import WordSeqCorpus
from data_utils.data_feed import WordSeqDataFeed
from models.word_seq2seq import Word2SeqAutoEncoder as Model
from config_utils import Word2SeqAutoConfig as Config

# constants
tf.app.flags.DEFINE_string("data_dir", "Data/", "the dir that has the raw corpus file")
tf.app.flags.DEFINE_string("data_file", "open_subtitle.txt", "the file that contains the raw data")
tf.app.flags.DEFINE_string("vocab_file", "vocab.txt", "the file that contains the given validation data")
tf.app.flags.DEFINE_string("valid_data", "valid.txt", "the file that contains the given testing data")
tf.app.flags.DEFINE_string("test_data", "test.txt", "the file that contains the given vocabulary")
tf.app.flags.DEFINE_string("work_dir", "seq_working/", "Experiment results directory.")
tf.app.flags.DEFINE_string("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_string("max_vocab_size", 20000, "The top N vocabulary we use.")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_bool("resume", False, "Resume training from the ckp at test_path")
tf.app.flags.DEFINE_bool("forward", True, "Do decoding only")
tf.app.flags.DEFINE_string("test_path", "run1480917145", "the dir to load checkpoint for forward only")

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
    api = WordSeqCorpus(FLAGS.data_dir, FLAGS.data_file, FLAGS.valid_data, FLAGS.test_data, FLAGS.vocab_file,
                        config.max_enc_len, config.max_dec_len, config.line_thres)
    corpus_data = api.get_corpus()

    # convert to numeric input outputs that fits into TF models
    train_feed = WordSeqDataFeed("Train", config,corpus_data["train"], api.vocab)
    valid_feed = WordSeqDataFeed("Valid", config,corpus_data["valid"], api.vocab)
    test_feed = WordSeqDataFeed("Test", config,corpus_data["test"], api.vocab)

    if not os.path.exists(FLAGS.work_dir):
        os.mkdir(FLAGS.work_dir)

    if FLAGS.forward or FLAGS.resume:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.test_path)
    else:
        log_dir = os.path.join(FLAGS.work_dir, "run" + str(int(time.time())))
        os.mkdir(log_dir)

    # dump log file to the log_dir
    if not FLAGS.forward:
        with open(os.path.join(log_dir, "meta.txt"), "wb") as f:
            f.write(pp(config, output=False))

    # begin training
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1*config.init_w, config.init_w)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = Model(sess, config, len(train_feed.vocab), train_feed.EOS_ID,
                                        log_dir=None if FLAGS.forward else log_dir, forward=False)

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_model = Model(sess, valid_config, len(train_feed.vocab), train_feed.EOS_ID, None, forward=False)

        # get a random batch and do forward decoding. Print the most likely response
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            test_model = Model(sess, test_config, len(train_feed.vocab), train_feed.EOS_ID, None, forward=True)

        ckp_dir = os.path.join(log_dir, "checkpoints")

        global_t = 0
        patience = 10  # wait for at least 10 epoch before consider early stop
        valid_loss_threshold = np.inf
        best_valid_loss = np.inf
        checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__ + ".ckpt")

        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)

        ckpt = tf.train.get_checkpoint_state(ckp_dir)
        base_epoch = 0

        if ckpt:
            print("Reading models parameters from %s" % ckpt.model_checkpoint_path)
            sess.run(tf.initialize_all_variables())
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            base_epoch = int(ckpt.model_checkpoint_path.split("-")[1]) + 1
            print("Resume from epoch %d" % base_epoch)
        else:
            print("Created models with fresh parameters.")
            sess.run(tf.initialize_all_variables())

        if not FLAGS.forward:
            for epoch in range(base_epoch, config.max_epoch):
                print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))

                if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                    train_feed.epoch_init(config.batch_size, shuffle=True)
                global_t, train_loss = model.train(global_t, sess, train_feed, update_limit=5000)

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
            # dump everything to a file
            test_feed.epoch_init(14, shuffle=False)
            all_n_best = test_model.test("TEST", sess, test_feed, num_batch=None)
            with open(os.path.join(log_dir, "%s_%s_test.txt" % (model.__class__.__name__, config.loop_function)),
                      "wb") as f:
                for best in all_n_best:
                    for score, n in best:
                        f.write(" ".join([train_feed.rev_vocab[word] for word in n]) + " ")
                    f.write("\n")

            # do sampling to see what kind of sentences is generated
            train_feed.epoch_init(test_config.batch_size, shuffle=True)
            test_model.test("TRAIN", sess, train_feed, num_batch=50)

            # do sampling to see what kind of sentences is generated
            test_feed.epoch_init(test_config.batch_size, shuffle=True)
            test_model.test("TEST", sess, test_feed, num_batch=50)

            # begin validation
            valid_feed.epoch_init(valid_config.batch_size, shuffle=False)
            valid_model.valid("VALID", sess, valid_feed)

            test_feed.epoch_init(valid_config.batch_size, shuffle=False)
            valid_model.valid("TEST", sess, test_feed)

if __name__ == "__main__":
    main()













