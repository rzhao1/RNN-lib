class Utt2SeqConfig(object):
    op = "sgd"
    cell_type = "lstm"
    use_attention = False

    # general config
    grad_clip = 5.0
    init_w = 0.05
    batch_size = 20
    clause_embed_size = 300
    embed_size = 150
    cell_size = 500
    num_layer = 2
    max_epoch = 20
    beam_size = 2

    # SGD training related
    init_lr = 0.6
    lr_hold = 1
    lr_decay = 0.6
    keep_prob = 0.6
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True


class Word2SeqConfig(object):
    op = "sgd"
    cell_type = "gru"
    use_attention = True

    # general config
    grad_clip = 5.0
    init_w = 0.05
    batch_size = 30
    embed_size = 300
    cell_size = 500
    num_layer = 1
    max_epoch = 20
    beam_size = 5

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
