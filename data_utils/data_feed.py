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

    def __init__(self, name, data, vocab):
        self.name = name
        # plus 4 is because of the 4 built-in words PAD, UNK, GO and EOS
        self.vocab = {word:idx+4 for idx, word in enumerate(vocab)}
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
        print("%s feed loads %d samples" % self.data_size)

    def line_2_ids(self, line):
        return [self.vocab.get(word, self.UNK_ID) for word in line.split()]

    def _shuffle(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, selected_index):
        x_rows = [self.id_xs[idx] for idx in selected_index]
        y_rows = [self.id_ys[idx] for idx in selected_index]
        encoder_len = np.array([len(row) for row in x_rows], dtype=np.int32)
        decoder_len = np.array([len(row) for row in y_rows], dtype=np.int32)

        max_enc_len = np.max(encoder_len)
        max_dec_len = np.max(decoder_len)

        encoder_x = np.zeros((self.batch_size, max_enc_len), dtype=np.int32)
        decoder_y = np.zeros((self.batch_size, max_dec_len), dtype=np.int32)

        for idx, (x, y) in enumerate(zip(x_rows, y_rows)):
            encoder_x[idx, 0:encoder_len[idx]] = x
            decoder_y[idx, 0:decoder_len[idx]] = y

        return encoder_len, decoder_len, encoder_x, decoder_y

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

        print("%s begins training with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None


# used by utt pretrain
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
    GO_ID = 2
    EOS_ID = 3

    def __init__(self, name, data, vocab):
        self.name = name
        # plus 4 is because of the 4 built-in words PAD, UNK, GO and EOS
        self.vocab = {word: idx + 4 for idx, word in enumerate(vocab)}
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
        print("%s feed loads %d samples" % self.data_size)

    def line_2_ids(self, line):
        return [self.vocab.get(word, self.UNK_ID) for word in line.split()]

    def _shuffle(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, selected_index):
        x_rows = [self.id_xs[idx] for idx in selected_index]
        y_rows = [self.id_ys[idx] for idx in selected_index]
        encoder_len = np.array([len(row) for row in x_rows], dtype=np.int32)
        decoder_len = np.array([len(row) for row in y_rows], dtype=np.int32)

        max_enc_len = np.max(encoder_len)
        max_dec_len = np.max(decoder_len)

        encoder_x = np.zeros((self.batch_size, max_enc_len), dtype=np.int32)
        decoder_y = np.zeros((self.batch_size, max_dec_len), dtype=np.int32)

        for idx, (x, y) in enumerate(zip(x_rows, y_rows)):
            encoder_x[idx, 0:encoder_len[idx]] = x
            decoder_y[idx, 0:decoder_len[idx]] = y

        return encoder_len, decoder_len, encoder_x, decoder_y

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

        print("%s begins training with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None

