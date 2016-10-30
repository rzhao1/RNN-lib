import os
import time
from beeprint import pp
import numpy as np
import tensorflow as tf
from data_utils.split_data import ReverseDataLoader
from models.SeqLM import SeqLM

# constants
tf.app.flags.DEFINE_string("vocab_size", 128, "Vocabulary size.")
tf.app.flags.DEFINE_string("work_dir", "seq_working", "Experiment results directory.")
tf.app.flags.DEFINE_string("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", False, "Create checkpoints")
tf.app.flags.DEFINE_string("test_path", "run1477517432", "the dir to load checkpoint for forward only")


FLAGS = tf.app.flags.FLAGS


class Config(object):
    op = "rmsprop"
    cell_type = "lstm"
    use_stack = True
    max_stack_depth = 20
    stack_width = 128
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
    min_train_len = 5
    max_train_len = 10
    max_test_len = 20


def main():
    config = Config()

    # convert to numeric input outputs that fits into TF models
    train_feed = ReverseDataLoader("Train", config.min_train_len, config.max_train_len, FLAGS.vocab_size)
    easy_test_feed = ReverseDataLoader("E-Test", config.min_train_len, config.max_train_len, FLAGS.vocab_size)
    hard_test_feed = ReverseDataLoader("H-Test", config.max_train_len, config.max_test_len, FLAGS.vocab_size)

    if FLAGS.forward_only:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.test_path)
    else:
        log_dir = os.path.join(FLAGS.work_dir, "run"+str(int(time.time())))

    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.batch_size = 200

    pp(config)

    # begin training
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1*config.init_w, config.init_w)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = SeqLM(sess, config, vocab=train_feed.vocab, log_dir=None if FLAGS.forward_only else log_dir)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            test_model = SeqLM(sess, test_config, vocab=train_feed.vocab, log_dir=None)

        ckp_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)
        ckpt = tf.train.get_checkpoint_state(ckp_dir)
        if ckpt:
            print("Reading models parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created models with fresh parameters.")
            sess.run(tf.initialize_all_variables())

        if not FLAGS.forward_only:
            global_t = 0
            patience = 10 # wait for at least 10 epoch before stop
            dev_loss_threshold = np.inf
            best_dev_loss = np.inf
            checkpoint_path = os.path.join(ckp_dir, "seq-ptb-lm.ckpt")

            for epoch in range(config.max_epoch):
                print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))
                train_feed.epoch_init(config.batch_size, num_batch=1500, reset_data=True)
                global_t, train_loss = model.train(global_t, sess, train_feed)

                # begin validation
                easy_test_feed.epoch_init(test_config.batch_size, num_batch=50, reset_data=True)
                test_loss, _, _, _ = test_model.test("E-TEST", sess, easy_test_feed, verbose=False)

                hard_test_feed.epoch_init(test_config.batch_size, num_batch=50, reset_data=True)
                test_loss, _, _, _ = test_model.test("H-TEST", sess, hard_test_feed, verbose=False)

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
            hard_test_feed.epoch_init(test_config.batch_size, shuffle=False)
            _, pred_outs, label_outs, raw_outs = test_model.test("TEST", sess, hard_test_feed)
            # dump the result into a file and mark the errors
            eval_f = open(os.path.join(log_dir, "eval.txt"), "wb")
            for pred_seq, label_seq, raw_seq in zip(pred_outs, label_outs, raw_outs):
                line = []
                all_correct = True
                for pred_tkn, label_tkn, raw_tkn in zip(pred_seq, label_seq, raw_seq):
                    if label_tkn != 0:
                        line.append(api.vocab[pred_tkn])
                        if pred_tkn != label_tkn:
                            all_correct = False
                            line.append("*%s*" % api.vocab[label_tkn])
                    elif raw_tkn != 0:
                        line.append(api.vocab[raw_tkn])

                if not all_correct:
                    eval_f.write(" ".join(line)+"\n")
            eval_f.close()


if __name__ == "__main__":
    if FLAGS.forward_only:
        if FLAGS.test_path is None:
            print("Set test_path before forward only")
            exit(1)
    main()





