class Utt2SeqConfig(object):
    op = "sgd"
    cell_type = "lstm"

    # general config
    init_w = 0.08
    batch_size = 50
    clause_embed_size = 600
    embed_size = 300
    cell_size = 800
    num_layer = 2
    max_epoch = 20
    beam_size = 20

    # SGD training related
    grad_clip = 5.0
    init_lr = 0.5
    lr_hold = 2
    lr_decay = 0.9
    keep_prob = 0.5
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True


class Word2SeqConfig(object):
    op = "adam"
    cell_type = "gru"
    loop_function = "gumble"

    # general config
    grad_clip = 10.0
    init_w = 0.08
    batch_size = 64
    embed_size = 300
    cell_size = 1000
    num_layer = 1
    max_epoch = 20
    beam_size = 5

    line_thres =2
    max_enc_len = 30
    max_dec_len = 30

    # training related
    init_lr = 0.001
    lr_hold = 1
    lr_decay = 0.6
    keep_prob = 0.7
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True


class Word2SeqAutoConfig(object):
    op = "adam"
    cell_type = "gru"
    loop_function = "gumble"

    # general config
    grad_clip = 10.0
    init_w = 0.05
    batch_size = 64
    embed_size = 300
    cell_size = 800
    num_layer = 1
    max_epoch = 20
    beam_size = 5

    line_thres =2
    max_dec_len = 30
    max_enc_len = 30

    # SGD training related
    init_lr = 0.001
    lr_hold = 1
    lr_decay = 0.6
    keep_prob = 0.7
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True


class HybridSeqConfig(object):
    op = "adam"
    cell_type = "gru"
    loop_function = "gumble"

    # general config
    grad_clip = 10.0
    init_w = 0.08
    batch_size = 64
    embed_size = 300
    dec_cell_size = 600
    utt_cell_size = 600
    context_cell_size = 600
    num_layer = 1
    max_epoch = 20
    beam_size = 5

    context_size = 10
    max_enc_size = 50
    max_dec_size = 25

    # SGD training related
    init_lr = 0.001
    lr_hold = 1
    lr_decay = 0.6
    keep_prob = 0.7
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True


class FutureSeqConfig(object):
    op = "adam"
    cell_type = "gru"
    loop_function = "gumble"

    # general config
    grad_clip = 10.0
    init_w = 0.05
    batch_size = 64
    embed_size = 300
    cell_size = 800
    num_layer = 1
    max_epoch = 20
    beam_size = 20

    max_utt_size = 30
    context_size = 5

    # SGD training related
    init_lr = 0.001
    lr_hold = 1
    lr_decay = 0.6
    keep_prob = 0.8
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True