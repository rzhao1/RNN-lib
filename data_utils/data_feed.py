import numpy as np


class WordSeqDataFeed(object):
    # iteration related
    """
    Use case:
    epoch_init(batch_size=20, shuffle=True)
    next_batch() to get the next bactch. The end of batch is None
    """
    ptr = 0
    batch_size = 0
    num_batch = None
    batch_indexes = None
    PAD_ID = 0
    UNK_ID = 1
    GO_ID = 2
    EOS_ID = 3

    def __init__(self, name, config, data, vocab):
        self.name = name
        # plus 4 is because of the 4 built-in words PAD, UNK, GO and EOS
        self.vocab = {word: idx+4 for idx, word in enumerate(vocab)}
        self.vocab["PAD_"] = self.PAD_ID
        self.vocab["UNK_"] = self.UNK_ID
        self.vocab["GO_"] = self.GO_ID
        self.vocab["EOS_"] = self.EOS_ID
        self.decoder_size=config.decoder_size
        # make sure we add 4 new special symbol
        assert len(self.vocab) == (len(vocab)+4)

        data_x, data_y = data

        # convert data into ids
        self.id_xs, self.id_ys = [], []
        for line in data_x:
            self.id_xs.append(self.line_2_ids(line))
        for line in data_y:
            self.id_ys.append(self.line_2_ids(line))

        all_lens = [len(line) for line in self.id_xs]
        self.indexes = list(np.argsort(all_lens))
        self.data_size = len(self.id_xs)
        print("%s feed loads %d samples" % (name, self.data_size))

    def line_2_ids(self, line):
        return [self.vocab.get(word, self.UNK_ID) for word in line.split()]

    def line_2_ids_util(self,line,vocab,default):
        return [vocab.get(word, default) for word in line.split()]

    def _shuffle(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self,selected_index):
        x_rows = [self.id_xs[idx] for idx in selected_index]
        y_rows = [[self.PAD_ID] + self.id_ys[idx] for idx in selected_index]
        encoder_len = np.array([len(row) for row in x_rows], dtype=np.int32)
        decoder_len = np.array([len(row) for row in y_rows], dtype=np.int32)

        max_enc_len = np.max(encoder_len)
        #Fixed-size decoder
        max_dec_len = self.decoder_size

        encoder_x = np.zeros((self.batch_size, max_enc_len), dtype=np.int32)
        decoder_y = np.zeros((self.batch_size, max_dec_len), dtype=np.int32)

        for idx, (x, y) in enumerate(zip(x_rows, y_rows)):
            encoder_x[idx, 0:encoder_len[idx]] = x
            if(decoder_len[idx]>=max_dec_len):
                decoder_y[idx, 0:max_dec_len-2] = y[0:max_dec_len-2]
            else:
                decoder_y[idx,0:decoder_len[idx]]=y
                decoder_y[idx,decoder_len[idx]+1:max_dec_len-2]=0

            decoder_y[idx, max_dec_len - 1] = self.EOS_ID
        return encoder_len, encoder_x, decoder_y

    def epoch_init(self, batch_size, shuffle=True):
        # create batch indexes for computation efficiency
        self.ptr = 0
        self.batch_size = batch_size
        self.num_batch = self.data_size // batch_size
        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])
        if shuffle:
            self._shuffle()

        print("%s begins with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None


# used by utt pre-train
class UttDataFeed(object):
    # iteration related
    """
    Use case:
    epoch_init(batch_size=20, shuffle=True)
    next_batch() to get the next bactch. The end of batch is None
    """
    ptr = 0
    batch_size = 0
    num_batch = None
    batch_indexes = None
    PAD_ID = 0
    UNK_ID = 1
    vocab_offset = len([PAD_ID, UNK_ID])

    # feature names
    SPK_ID = 0
    TEXT_ID = 1
    DA_ID = 2
    SENTI_ID = 3
    OPINION_ID = 4
    EMPATH_ID = 5
    LIWC_ID = 6

    feature_types = ["binary", "multiclass", "multinominal", "regression", "seq"]

    # dialog act labels
    dialog_acts = ["inform", "question", "other", "goodbye", "disconfirm",
                   "confirm", "non-verbal", "non-understand", "request"]

    feature_size = [2, None, len(dialog_acts), 3, None, None, None]


    def __init__(self, name, data, vocab):
        self.name = name
        # plus 4 is because of the 2 built-in words PAD, UNK
        self.vocab = {word: idx + self.vocab_offset for idx, word in enumerate(vocab)}
        self.vocab["PAD"] = self.PAD_ID
        self.vocab["UNK"] = self.UNK_ID

        # convert data into ids
        self.id_text = []
        self.labels = []
        for idx, line in enumerate(data):
            if line == "$$$":
                continue
            self.id_text.append(self.line_2_ids(line))
            self.labels.append(self.line_2_label(None if idx == 0 else data[idx-1],
                                                 line,
                                                 None if idx < len(data) else data[idx+1]))

        # sort the lines in terms of length for computation efficiency
        self.indexes = list(np.argsort([len(line) for line in self.id_text]))
        self.data_size = len(self.id_text)
        print("%s feed loads %d samples" % (name, self.data_size))

    def line_2_ids(self, line):
        return [self.vocab.get(word, self.UNK_ID) for word in line[self.TEXT_ID].split()]

    def line_2_label(self, prev_line, line, next_line):
        # current line
        features = dict()
        features[self.DA_ID] = self.dialog_acts.index(line[self.DA_ID])
        features[self.SENTI_ID] = map(float, line[self.SENTI_ID].split())[0:3]
        return features

    def _shuffle(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, selected_index):
        x_rows = [self.id_text[idx] for idx in selected_index]
        y_rows = [self.labels[idx] for idx in selected_index]
        encoder_len = np.array([len(row) for row in x_rows], dtype=np.int32)
        max_enc_len = np.max(encoder_len)

        encoder_x = np.zeros((self.batch_size, max_enc_len), dtype=np.int32)
        label_y = dict()
        # add dialog act
        label_y[self.DA_ID] = np.zeros(self.batch_size, dtype=np.int32)
        label_y[self.SENTI_ID] = np.zeros((self.batch_size, self.feature_size[self.SENTI_ID]), dtype=np.float32)

        for idx, (x, y) in enumerate(zip(x_rows, y_rows)):
            encoder_x[idx, 0:encoder_len[idx]] = x
            label_y[self.DA_ID][idx] = y[self.DA_ID]
            label_y[self.SENTI_ID][idx] = y[self.SENTI_ID]

        return encoder_x, encoder_len, label_y

    def epoch_init(self, batch_size, shuffle=True):
        # create batch indexes for computation efficiency
        self.ptr = 0
        self.batch_size = batch_size
        self.num_batch = self.data_size // batch_size
        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])
        if shuffle:
            self._shuffle()

        print("::%s begins training with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None

