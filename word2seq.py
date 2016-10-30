import os
import time
from beeprint import pp
import numpy as np
import tensorflow as tf
from data_utils.split_data import WordSeqCorpus
from data_utils.data_feed import WordSeqDataFeed

# constants
tf.app.flags.DEFINE_string("data_dir", "Data/", "the dir that has the raw corpus file")
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
    api = WordSeqCorpus(FLAGS.data_dir, "clean_data.txt", FLAGS.line_thres, [7,1,2])
    corpus_data = api.get_corpus()


    # convert to numeric input outputs that fits into TF models
    train_feed = WordSeqDataFeed("Train", corpus_data["train"], api.vocab)
    valid_feed = WordSeqDataFeed("Valid", corpus_data["valid"], api.vocab)
    test_feed = WordSeqDataFeed("Test", corpus_data["test"], api.vocab)

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
            pass
            # TODO init model here
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            pass
            # TODO init test model here

        ckp_dir = os.path.join(log_dir, "checkpoints")

        if not FLAGS.forward_only:
            global_t = 0
            patience = 10 # wait for at least 10 epoch before stop
            dev_loss_threshold = np.inf
            best_dev_loss = np.inf
            checkpoint_path = os.path.join(ckp_dir, "seq-ptb-lm.ckpt")

            for epoch in range(config.max_epoch):
                pass
                # TODO training here
        else:
            # begin evaluation
            # TODO beam search and decoding here
            pass


if __name__ == "__main__":
    main()





