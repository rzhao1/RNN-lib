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
        self.max_decoder_size = config.max_dec_len
        # make sure we add 4 new special symbol
        assert len(self.vocab) == (len(vocab)+4)
        self.rev_vocab = {v:k for k, v in self.vocab.items()}

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
        y_rows = [[self.GO_ID] + self.id_ys[idx] for idx in selected_index]
        encoder_len = np.array([len(row) for row in x_rows], dtype=np.int32)
        decoder_len = np.array([len(row) for row in y_rows], dtype=np.int32)

        max_enc_len = np.max(encoder_len)
        max_dec_len = self.max_decoder_size

        encoder_x = np.zeros((self.batch_size, max_enc_len), dtype=np.int32)
        decoder_y = np.zeros((self.batch_size, max_dec_len), dtype=np.int32)

        for idx, (x, y) in enumerate(zip(x_rows, y_rows)):
            encoder_x[idx, 0:encoder_len[idx]] = x
            # we discard words that are longer than max_dec_len-2
            if decoder_len[idx] >= max_dec_len:
                decoder_y[idx, :] = y[0:max_dec_len-1] + [self.EOS_ID]
                decoder_len[idx] = max_dec_len
            else:
                decoder_y[idx, 0: decoder_len[idx]+1] = y + [self.EOS_ID]
                decoder_len[idx] += 1 # include the new EOS

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


# used by utt 2 seq
class UttSeqDataFeed(object):
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
    GO_ID = 2 # start a new turn
    CONTINUE_ID = 3 # continue (no conversation floor change)
    EOS_ID = 4
    vocab_offset = len([PAD_ID, UNK_ID, GO_ID, CONTINUE_ID, EOS_ID])

    # feature names
    SPK_ID = 0
    TEXT_ID = 1
    DA_ID = 2
    SENTI_ID = 3
    OPINION_ID = 4
    EMPATH_ID = 5
    LIWC_ID = 6

    def __init__(self, name, data, vocab):
        self.name = name
        # plus 4 is because of the 4 built-in words PAD, UNK, GO and EOS
        self.vocab = {word: idx+self.vocab_offset for idx, word in enumerate(vocab)}
        self.vocab["PAD_"] = self.PAD_ID
        self.vocab["UNK_"] = self.UNK_ID
        self.vocab["GO_"] = self.GO_ID
        self.vocab["CONTINUE_"] = self.CONTINUE_ID
        self.vocab["EOS_"] = self.EOS_ID
        # make sure we add 5 new special symbol
        assert len(self.vocab) == (len(vocab)+5)

        # get reverse vocab
        self.rev_vocab = {v:k for k, v in self.vocab.items()}

        data_x, data_y = data

        # convert data into ids
        self.feat_xs, self.id_ys = [], []
        for x_line, y_line in zip(data_x, data_y):
            x, y = self.line_2_ids_and_vec(x_line, y_line)
            self.feat_xs.append(x)
            self.id_ys.append(y)

        all_lens = [len(line) for line in self.feat_xs]
        self.max_dec_size = np.max([len(line) for line in self.id_ys]) + 1
        self.indexes = list(np.argsort(all_lens))
        self.data_size = len(self.feat_xs)
        self.feat_size = len(self.vocab) + 1
        print("%s feed loads %d samples with max decoder size %d" % (name, self.data_size, self.max_dec_size))

    def line_2_ids_and_vec(self, x_line, y_line):
        # x will be a list of vectors. Each vector has self.feat_size
        # y will a  list of int (ids). Each ID is the word index.
        #  It will also decide the turn taking part, that is the first character is GO_ or CONTINUE_
        y_speaker, y_text = y_line
        x = []
        last_speaker = None
        for x_feat in x_line:
            bow = {}
            # get bag of words first
            for w in x_feat[self.TEXT_ID]:
                bow[self.vocab.get(w, self.UNK_ID)] = bow.get(self.vocab.get(w, self.UNK_ID), 0) + 1
            # get if speaker is different from the decoding speaker
            last_speaker = x_feat[self.SPK_ID]
            same_speaker = 1 if last_speaker != y_speaker else 0
            x.append((bow, same_speaker))

        y = [self.vocab.get(word, self.UNK_ID) for word in y_text]
        if last_speaker != y_speaker:
            y = [self.GO_ID] + y
        else:
            y = [self.CONTINUE_ID] + y

        return x, y

    def _shuffle(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self,selected_index):
        x_rows = [self.feat_xs[idx] for idx in selected_index]
        y_rows = [self.id_ys[idx] for idx in selected_index]
        encoder_len = np.array([len(row) for row in x_rows], dtype=np.int32)
        decoder_len = np.array([len(row) for row in y_rows], dtype=np.int32)

        max_enc_len = np.max(encoder_len)
        max_dec_len = self.max_dec_size

        encoder_x = np.zeros((self.batch_size, max_enc_len, self.feat_size), dtype=np.float32)
        decoder_y = np.zeros((self.batch_size, max_dec_len), dtype=np.int32)

        for idx, (x, y) in enumerate(zip(x_rows, y_rows)):
            for t_id, (utt, floor) in enumerate(x):
                for w_id, cnt in utt.items():
                    encoder_x[idx, t_id, w_id] = cnt
                encoder_x[idx, t_id, -1] = floor

            # we discard words that are longer than max_dec_len-2
            if decoder_len[idx] >= max_dec_len:
                decoder_y[idx, :] = y[0:max_dec_len-1] + [self.EOS_ID]
                decoder_len[idx] = max_dec_len
            else:
                decoder_y[idx, 0: decoder_len[idx]+1] = y + [self.EOS_ID]
                decoder_len[idx] += 1 # include the new EOS

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

        print("%s begins with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None


class HybridSeqDataFeed(object):
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
    CONTINUE_ID =3
    EOS_ID = 4

    def __init__(self, name, config, data, api):
        self.name = name
        # plus 5 is because of the 4 built-in words PAD, UNK, GO, CONTINUE and EOS
        self.vocab = {word: idx+5 for idx, word in enumerate(api.vocab)}
        self.vocab["PAD_"] = self.PAD_ID
        self.vocab["UNK_"] = self.UNK_ID
        self.vocab["GO_"] = self.GO_ID
        self.vocab["CONT_"] = self.CONTINUE_ID
        self.vocab["EOS_"] = self.EOS_ID
        self.max_encoder_size = config.max_enc_size
        self.max_decoder_size = config.max_dec_size
        # make sure we add 4 new special symbol
        assert len(self.vocab) == (len(api.vocab)+5)
        self.rev_vocab = {v:k for k, v in self.vocab.items()}

        # convert movie profile to ids
        self.id_profile = {}
        for key, profile in api.movie_profile.items():
            temp = {w: c for w, c in profile}
            self.id_profile[key] = [temp.get(w, 0) for w in self.vocab]

        # convert data into ids
        self.id_lines = []
        for key, speaker, utt in api.data_lines:
            self.id_lines.append((key, speaker, self.line_2_ids(utt)))

        self.id_xs, self.id_ys = [], []
        for x, y in zip(data[0], data[1]):
            if len(self.id_lines[y][2]) > 5:
                self.id_xs.append(x)
                self.id_ys.append(y)

        all_lens = [len(self.id_lines[line_id][2]) for line_id in self.id_ys]
        self.indexes = list(np.argsort(all_lens))
        self.data_size = len(self.id_xs)
        print("%s feed loads %d samples" % (name, self.data_size))

    def line_2_ids(self, line):
        return [self.vocab.get(word, self.UNK_ID) for word in line.split()]

    def _shuffle(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self,selected_index):
        # return: profile_mask, context_x, context_len, prev_x, prev_len, decoder_y
        # profile_mask B*V
        # context_x B*C*MAX_ENC
        # prev_x B*MAX_ENC
        max_enc_len = self.max_encoder_size
        max_dec_len = self.max_decoder_size

        x_rows = [self.id_xs[idx] for idx in selected_index]
        y_rows = [self.id_ys[idx] for idx in selected_index]

        # break down y labels
        x_context, x_prev_utt, x_prev_spk = [], [], []
        for ctx, prev in x_rows:
            x_context.append([self.id_lines[idx][2][0:max_enc_len] for idx in ctx])
            x_prev_utt.append(self.id_lines[prev][2][0:max_enc_len])
            x_prev_spk.append(self.id_lines[prev][1])

        # break down y labels
        y_profile, y_speakers, y_utts = [], [], []
        for y_id in y_rows:
            key, spk, utt = self.id_lines[y_id]
            y_profile.append(self.id_profile[key])
            y_speakers.append(spk)
            y_utts.append(utt[0:max_dec_len-2])

        context_len = np.array([len(row) for row in x_context], dtype=np.int32)
        prev_len = np.array([len(row) for row in x_prev_utt], dtype=np.int32)
        max_context_len = np.max(context_len)

        profile_x = np.array(y_profile, dtype=np.int32)
        context_x = np.zeros((self.batch_size, max_context_len, max_enc_len), dtype=np.int32)
        prev_x = np.zeros((self.batch_size, max_enc_len), dtype=np.int32)
        decoder_y = np.zeros((self.batch_size, max_dec_len), dtype=np.int32)

        for b_id in range(self.batch_size):
            for c_id, c_x in enumerate(x_context[b_id]):
                context_x[b_id, c_id, 0:len(c_x)] = c_x

            prev_x[b_id, 0:len(x_prev_utt[b_id])] = x_prev_utt[b_id]
            # decide if GO or CONT
            begin_symbol = self.GO_ID if x_prev_spk[b_id] != y_speakers[b_id] else self.CONTINUE_ID
            decoder_y[b_id, 0:len(y_utts[b_id])+2] = [begin_symbol] + y_utts[b_id] + [self.EOS_ID]

        return profile_x, context_len, prev_len, context_x, prev_x, decoder_y

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


class FutureSeqDataFeed(object):
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

    def __init__(self, name, config, data, api):
        self.name = name
        # plus 5 is because of the 4 built-in words PAD, UNK, GO, CONTINUE and EOS
        self.vocab = {word: idx+4 for idx, word in enumerate(api.vocab)}
        self.vocab["PAD_"] = self.PAD_ID
        self.vocab["UNK_"] = self.UNK_ID
        self.vocab["GO_"] = self.GO_ID
        self.vocab["EOS_"] = self.EOS_ID
        self.max_utt_size = config.max_utt_size
        # make sure we add 4 new special symbol
        assert len(self.vocab) == (len(api.vocab)+4)
        self.rev_vocab = {v:k for k, v in self.vocab.items()}

        # convert data into ids
        self.id_lines = []
        for key, speaker, utt in api.data_lines:
            self.id_lines.append((key, speaker, self.line_2_ids(utt)))

        self.id_xyz = data

        self.indexes = range(len(self.id_xyz))
        self.data_size = len(self.id_xyz)
        print("%s feed loads %d samples" % (name, self.data_size))

    def line_2_ids(self, line):
        return [self.vocab.get(word, self.UNK_ID) for word in line.split()]

    def _shuffle(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self,selected_index):
        # return: profile_mask, context_x, context_len, prev_x, prev_len, decoder_y
        # profile_mask B*V
        # context_x B*C*MAX_ENC
        # prev_x B*MAX_ENC
        xyz_rows = [self.id_xyz[idx] for idx in selected_index]

        # break down y labels
        cxt_utts, x_utts, y_utts, z_utts = [], [], [], []
        batch_c_len = []
        for context, x, y, z in xyz_rows:
            batch_c_len.append(len(context))
            cxt_mat = np.zeros((len(context), self.max_utt_size))
            for c_id, c in enumerate(context):
                text = self.id_lines[c][2][0:self.max_utt_size]
                cxt_mat[c_id, 0:len(text)] = text
            cxt_utts.append(cxt_mat)
            x_utts.append(self.id_lines[x][2][0:self.max_utt_size])
            y_utts.append(self.id_lines[y][2][0:self.max_utt_size-2])
            z_utts.append(self.id_lines[z][2][0:self.max_utt_size])

        batch_c_len = np.array(batch_c_len, dtype=np.int32)
        batch_x_len = np.array([len(row) for row in x_utts], dtype=np.int32)
        batch_y_len = np.array([len(row) for row in y_utts], dtype=np.int32)
        batch_z_len = np.array([len(row) for row in z_utts], dtype=np.int32)

        c_batch = np.zeros((self.batch_size, np.max(batch_c_len), self.max_utt_size), dtype=np.int32)
        x_batch = np.zeros((self.batch_size, np.max(batch_x_len)), dtype=np.int32)
        y_batch = np.zeros((self.batch_size, self.max_utt_size), dtype=np.int32)
        z_batch = np.zeros((self.batch_size, np.max(batch_z_len)), dtype=np.int32)

        for b_id in range(self.batch_size):
            c_batch[b_id, 0:batch_c_len[b_id], :] = cxt_utts[b_id]
            x_batch[b_id, 0:len(x_utts[b_id])] = x_utts[b_id]
            y_batch[b_id, 0:len(y_utts[b_id])+2] = [self.GO_ID] + y_utts[b_id] + [self.EOS_ID]
            z_batch[b_id, 0:len(z_utts[b_id])] = z_utts[b_id]

        return batch_c_len, batch_x_len, batch_y_len, batch_z_len, c_batch, x_batch, y_batch, z_batch

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

