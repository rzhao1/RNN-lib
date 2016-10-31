import os
import time
from beeprint import pp
import numpy as np
import tensorflow as tf
from data_utils.split_data import UttCorpus
from data_utils.data_feed import UttDataFeed
from models.MultiSeqClassifier import MultiSeqClassifier

# constants
tf.app.flags.DEFINE_string("data_dir", "Data/ucsc_features", "the dir that has the raw corpus file")
tf.app.flags.DEFINE_string("work_dir", "seq_working", "Experiment results directory.")
tf.app.flags.DEFINE_string("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", False, "Create checkpoints")
FLAGS = tf.app.flags.FLAGS

class Config(object):
    op = "rmsprop"
    cell_type = "lstm"
    grad_clip = 5.0
    init_w = 0.05
    batch_size = 20
    embed_size = 64
    cell_size = 256
    num_layer = 1
    max_epoch = 20
    init_lr = 0.6
    lr_hold = 1
    lr_decay = 0.6
    keep_prob = 1.0
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True


def main():
    # load corpus
    api = UttCorpus(FLAGS.data_dir, "combine_result.txt", [7,1,2])
    corpus_data = api.get_corpus()

    # convert to numeric input outputs that fits into TF models
    train_feed = UttDataFeed("Train", corpus_data["train"], api.vocab)
    valid_feed = UttDataFeed("Valid", corpus_data["valid"], api.vocab)
    test_feed = UttDataFeed("Test", corpus_data["test"], api.vocab)

    if FLAGS.forward_only:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.test_path)
    else:
        log_dir = os.path.join(FLAGS.work_dir, "run"+str(int(time.time())))

    config = Config()
    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.batch_size = 200
    pp(config)

    # begin training
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1*config.init_w, config.init_w)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = MultiSeqClassifier(sess, config, len(train_feed.vocab), len(train_feed.dialog_acts), log_dir)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            test_model = MultiSeqClassifier(sess, test_config, len(train_feed.vocab), len(train_feed.dialog_acts), log_dir)

        ckp_dir = os.path.join(log_dir, "checkpoints")

        if not FLAGS.forward_only:
            global_t = 0
            patience = 10  # wait for at least 10 epoch before stop
            dev_loss_threshold = np.inf
            best_dev_loss = np.inf
            checkpoint_path = os.path.join(ckp_dir, "%s.ckpt" % __name__)

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
                test_loss, _, _, _ = test_model.valid("VALID", sess, valid_feed)

                test_feed.epoch_init(test_config.batch_size, shuffle=False)
                test_loss, _, _, _ = test_model.valid("TEST", sess, test_feed)

                done_epoch = epoch +1
                # only save a models if the dev loss is smaller
                # Decrease learning rate if no improvement was seen over last 3 times.
                if done_epoch > config.lr_hold:
                    sess.run(model.learning_rate_decay_op)

                if test_loss < best_dev_loss:
                    if test_loss <= dev_loss_threshold * config.improve_threshold:
                        patience = max(patience, done_epoch *config.patient_increase)
                        dev_loss_threshold = test_loss

                    # still save the best train model
                    if FLAGS.save_model:
                        model.saver.save(sess, checkpoint_path, global_step=epoch)
                    best_dev_loss = test_loss

                if config.early_stop and patience <= done_epoch:
                    print("!!Early stop due to run out of patience!!")
                    break
            print("Best test loss %f and perpleixyt %f" % (best_dev_loss, np.exp(best_dev_loss)))
            print("Done training")
        else:
            # begin evaluation
            # TODO beam search and decoding here
            pass


if __name__ == "__main__":
    main()













