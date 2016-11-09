class Utt2SeqConfig(object):
    op = "adam"
    cell_type = "gru"
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

    # SGD training related
    init_lr = 0.005
    lr_hold = 1
    lr_decay = 0.6
    keep_prob = 0.6
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True