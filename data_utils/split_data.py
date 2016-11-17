import os
import numpy as np
import json
from nltk.tokenize.regexp import WordPunctTokenizer


class WordSeqCorpus(object):
    train_x = None
    train_y = None
    valid_x = None
    valid_y = None
    test_x = None
    test_y = None

    def __init__(self, data_dir, data_name, split_size, max_vocab_size, max_enc_len, max_dec_len, line_thres):
        """"
        :param line_thres: how many line will be merged as encoding sentensce
        :param split_size: size of training:valid:test

        """

        self._data_dir = data_dir
        self._data_name = data_name
        self._cache_dir = os.path.join(data_dir, "word_seq_split")
        if not os.path.exists(self._cache_dir):
            os.mkdir(self._cache_dir)

        self.tokenizer = WordPunctTokenizer().tokenize
        self.line_threshold = line_thres
        self.split_size = split_size
        self.max_vocab_size = max_vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        # try to load from existing file
        if not self.load_data():
            with open(os.path.join(data_dir, data_name), "rb") as f:
                self._parse_file(f.readlines(), split_size)

        # clip data
        self.train_x, self.train_y = self.clip_to_max_len(self.train_x, self.train_y)
        self.valid_x, self.valid_y = self.clip_to_max_len(self.valid_x, self.valid_y)
        self.test_x, self.test_y = self.clip_to_max_len(self.test_x, self.test_y)

        # get vocabulary\
        self.vocab = self.get_vocab()

        self.print_stats("TRAIN", self.train_x, self.train_y)
        self.print_stats("VALID", self.valid_x, self.valid_y)
        self.print_stats("TEST", self.test_x, self.test_y)

    def get_vocab(self):
        # get vocabulary dictionary
        vocab_cnt = {}
        for line in self.train_x:
            for tkn in line.split():
                cnt = vocab_cnt.get(tkn, 0)
                vocab_cnt[tkn] = cnt + 1
        vocab_cnt = [(cnt, key) for key, cnt in vocab_cnt.items()]
        vocab_cnt = sorted(vocab_cnt, reverse=True)
        vocab = [key for cnt, key in vocab_cnt]
        cnts = [cnt for cnt, key in vocab_cnt]
        total = np.sum(cnts)
        valid = np.sum(cnts[0:self.max_vocab_size])
        print("Raw vocab cnt %d with valid ratio %f" % (len(vocab), float(valid)/total))

        cnts = [cnt for cnt, key in vocab_cnt]
        total = float(np.sum(cnts))
        valid = float(np.sum(cnts[0:self.max_vocab_size]))
        print("Before cutting. Raw vocab size is %d with valid ratio %f" % (len(vocab), valid/total))
        return vocab[0:self.max_vocab_size]

    def oov(self, name, data):
        oov_cnt = 0
        total_cnt = 0
        for line in data:
            for tkn in line.split():
                total_cnt += 1
                if tkn not in self.vocab:
                    oov_cnt += 1

        print("%s oov %f" % (name, float(oov_cnt)/total_cnt))

    def clip_to_max_len(self, enc_data, dec_data):
        new_enc_data = [" ".join(x.split()[-self.max_enc_len:]) for x in enc_data]
        new_dec_data = [" ".join(x.split()[0:self.max_dec_len]) for x in dec_data]
        return new_enc_data, new_dec_data

    def print_stats(self, name, enc_data, dec_data):
        enc_lens = [len(x.split()) for x in enc_data]
        avg_len = float(np.mean(enc_lens))
        max_len = float(np.max(enc_lens))
        dec_lens = [len(x.split()) for x in dec_data]
        dec_avg_len = float(np.mean(dec_lens))
        dec_max_len = float(np.max(dec_lens))
        print ('%s encoder avg len %.2f max len %.2f of %d lines' % (name, avg_len, max_len, len(enc_data)))
        print ('%s decoder avg len %.2f max len %.2f of %d lines' % (name, dec_avg_len, dec_max_len, len(dec_data)))

    def _parse_file(self, lines, split_size):
        """
        :param lines: Each line is a line from the file
        """
        utterances = []
        speakers = []
        movies = {}

        current_movie = []
        current_name = []
        for line in lines:
            if "FILE_NAME" in line:
                if current_movie:
                    movies[current_name] = current_movie
                current_name = line.strip()
                current_movie = []
            else:
                current_movie.append(line.strip())
        if current_movie:
            movies[current_name] = current_movie

        # shuffle movie here.
        shuffle_keys = movies.keys()
        np.random.shuffle(shuffle_keys)
        for key in shuffle_keys:
            speakers.append("$$$")
            utterances.append("$$$")
            speakers.extend([l.split("|||")[0] for l in movies[key]])
            utterances.extend([" ".join(self.tokenizer(l.split("|||")[1])).lower() for l in movies[key]])

        total_size = len(utterances)
        train_size = int(total_size * split_size[0] / np.sum(split_size))
        valid_size = int(total_size * split_size[1] / np.sum(split_size))

        content_xs = []
        content_ys = []
        # Pointer for decoder input
        print("Begin creating data")
        for idx, (spker, utt) in enumerate(zip(speakers, utterances)):
            # if we are at "a" in $$$ a b c, ignore the input
            if utt == "$$$" or "$$$" in utterances[max(0, idx - self.line_threshold): idx]:
                continue

            content_x = " ".join(utterances[max(0, idx - self.line_threshold):idx])
            if spker == speakers[idx - 1]:
                content_x += " #"
            content_xs.append(content_x)
            content_ys.append(utt)

        # split the data
        self.train_x = train_x = content_xs[0: train_size]
        self.train_y = train_y = content_ys[0: train_size]
        self.valid_x = valid_x = content_xs[train_size: train_size + valid_size]
        self.valid_y = valid_y = content_ys[train_size: train_size + valid_size]
        self.test_x = test_x = content_xs[train_size + valid_size:]
        self.test_y = test_y = content_ys[train_size + valid_size:]

        # begin dumpping data to file
        self.dump_data("train_x.txt", train_x)
        self.dump_data("train_y.txt", train_y)
        self.dump_data("valid_x.txt", valid_x)
        self.dump_data("valid_y.txt", valid_y)
        self.dump_data("test_x.txt", test_x)
        self.dump_data("test_y.txt", test_y)

    def dump_data(self, file_name, lines):
        if os.path.exists(file_name):
            raise ValueError("File already exists. Abort dumping")
        print("Dumping to %s with %d lines" % (file_name, len(lines)))
        with open(os.path.join(self._cache_dir, file_name), "wb") as f:
            for line in lines:
                f.write(line + "\n")

    def load_data(self):
        if not os.path.exists(os.path.join(self._cache_dir, "train_x.txt")):
            return False

        def load_file(file_name):
            with open(os.path.join(self._cache_dir, file_name), "rb") as f:
                lines = f.readlines()
                lines = [l.strip() for l in lines]
            return lines

        print("Loaded from cache")
        self.train_x = load_file("train_x.txt")
        self.train_y = load_file("train_y.txt")
        self.valid_x = load_file("valid_x.txt")
        self.valid_y = load_file("valid_y.txt")
        self.test_x = load_file("test_x.txt")
        self.test_y = load_file("test_y.txt")
        return True

    def get_corpus(self):
        """
        :return: the corpus in train/valid/test
        """
        return {"train": (self.train_x, self.train_y),
                "valid": (self.valid_x, self.valid_y),
                "test": (self.test_x, self.test_y)}


class UttCorpus(object):
    # feature names
    SPK_ID = 0
    TEXT_ID = 1
    DA_ID = 2
    SENTI_ID = 3
    OPINION_ID = 4
    EMPATH_ID = 5
    LIWC_ID = 6

    train_data = None
    valid_data = None
    test_data = None

    def __init__(self, data_dir, data_name, split_size, max_vocab_size):
        """"
        Each utt is a tuple, with various features
        :param split_size: size of training:valid:test

        """
        self._data_dir = data_dir
        self._data_name = data_name
        self.split_size = split_size

        # try to load from existing file
        if not self.load_data():
            with open(os.path.join(data_dir, data_name), "rb") as f:
                self._parse_file(f.readlines(), split_size)

        # get vocabulary
        self.vocab = self.get_vocab(max_vocab_size)

        self.print_stats("train", self.train_data)
        self.print_stats("valid", self.valid_data)
        self.print_stats("test", self.test_data)

    def get_vocab(self, max_vocab_size):
        # get vocabulary dictionary
        vocab_cnt = {}
        for line in self.train_data:
            for tkn in line[self.TEXT_ID].split():
                cnt = vocab_cnt.get(tkn, 0)
                vocab_cnt[tkn] = cnt + 1
        vocab_cnt = [(cnt, key) for key, cnt in vocab_cnt.items()]
        vocab_cnt = sorted(vocab_cnt, reverse=True)
        vocab = [key for cnt, key in vocab_cnt]
        return vocab[0:max_vocab_size]

    def print_stats(self, name, data):
        avg_len = float(np.mean([len(x[self.TEXT_ID].split()) for x in data]))
        print ('%s avg sent len %.2f of %d lines' % (name, avg_len, len(data)))

    def _parse_file(self, lines, split_size):
        """
        :param lines: Each line is a line from the file
        """
        movies = {}
        utt_features = []

        current_movie = []
        current_name = []
        for line in lines:
            if "FILE_NAME" in line:
                if current_movie:
                    movies[current_name] = current_movie
                current_name = line.strip()
                current_movie = []
            else:
                current_movie.append(line.strip())
        if current_movie:
            movies[current_name] = current_movie

        # shuffle movie here.
        shuffle_keys = movies.keys()
        np.random.shuffle(shuffle_keys)
        for key in shuffle_keys:
            utt_features.append("$$$")
            # we only add the line that have full 7 features
            utt_features.extend([l.split("|||") for l in movies[key] if len(l.split("|||")) == 7])

        total_size = len(utt_features)
        train_size = int(total_size * split_size[0] / 10)
        valid_size = int(total_size * split_size[1] / 10)

        # split the data
        self.train_data = utt_features[0: train_size]
        self.valid_data = utt_features[train_size: train_size + valid_size]
        self.test_data = utt_features[train_size + valid_size:]

        # begin dumpping data to file
        self.dump_data(os.path.join(self._data_dir, "train_data.txt"), self.train_data)
        self.dump_data(os.path.join(self._data_dir, "valid_data.txt"), self.valid_data)
        self.dump_data(os.path.join(self._data_dir, "test_data.txt"), self.test_data)

    def dump_data(self, file_name, lines):
        if os.path.exists(file_name):
            raise ValueError("File already exists. Abort dumping")
        print("Dumping to %s with %d lines" % (file_name, len(lines)))
        with open(file_name, "wb") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

    def load_data(self):
        if not os.path.exists(os.path.join(self._data_dir, "train_data.txt")):
            return False

        def load_file(file_name):
            with open(file_name, "rb") as f:
                lines = f.readlines()
                lines = [json.loads(l.strip()) for l in lines]
            return lines

        print("Loaded from cache")
        self.train_data = load_file(os.path.join(self._data_dir, "train_data.txt"))
        self.valid_data = load_file(os.path.join(self._data_dir, "valid_data.txt"))
        self.test_data = load_file(os.path.join(self._data_dir, "test_data.txt"))
        return True

    def get_corpus(self):
        """
        :return: the corpus in train/valid/test
        """
        return {"train": self.train_data,
                "valid": self.valid_data,
                "test": self.test_data}


class UttSeqCorpus(object):
    train_x = None
    train_y = None
    valid_x = None
    valid_y = None
    test_x = None
    test_y = None

    # feature names
    SPK_ID = 0
    TEXT_ID = 1
    DA_ID = 2
    SENTI_ID = 3
    OPINION_ID = 4
    EMPATH_ID = 5
    LIWC_ID = 6

    cache_name = "post_utt_features.txt"

    def __init__(self, data_dir, data_name, split_size, max_vocab_size, max_enc_utt_len, max_dec_word_len):
        """"
        :param line_thres: how many line will be merged as encoding sentensce
        :param split_size: size of training:valid:test

        """

        self._data_dir = data_dir
        self._data_name = data_name
        self._cache_dir = os.path.join(data_dir, "utt_seq_split")
        if not os.path.exists(self._cache_dir):
            os.mkdir(self._cache_dir)

        self.tokenizer = WordPunctTokenizer().tokenize
        self.split_size = split_size
        self.max_vocab_size = max_vocab_size
        self.max_enc_utt_len = max_enc_utt_len
        self.max_dec_word_len = max_dec_word_len

        utt_features = self.load_data()
        if utt_features is None:
            with open(os.path.join(data_dir, data_name), "rb") as f:
                utt_features = self._parse_file(f.readlines())

        self._create_corpus(utt_features, split_size)

        # clip train_y. Different word2seq, encoder don't need clipping, since it fixed history
        self.train_y = self.clip_to_max_len(self.train_y)
        self.valid_y = self.clip_to_max_len(self.valid_y)
        self.test_y = self.clip_to_max_len(self.test_y)

        # get vocabulary\
        self.vocab = self.get_vocab()

        self.print_stats("TRAIN", self.train_x, self.train_y)
        self.print_stats("VALID", self.valid_x, self.valid_y)
        self.print_stats("TEST", self.test_x, self.test_y)

    def get_vocab(self):
        # get vocabulary dictionary
        vocab_cnt = {}
        for spk, line in self.train_y:
            for tkn in line:
                cnt = vocab_cnt.get(tkn, 0)
                vocab_cnt[tkn] = cnt + 1
        vocab_cnt = [(cnt, key) for key, cnt in vocab_cnt.items()]
        vocab_cnt = sorted(vocab_cnt, reverse=True)
        vocab = [key for cnt, key in vocab_cnt]
        cnts = [cnt for cnt, key in vocab_cnt]
        total = float(np.sum(cnts))
        valid = float(np.sum(cnts[0:self.max_vocab_size]))
        print("Before cutting. Raw vocab size is %d with valid ratio %f" % (len(vocab), valid / total))
        return vocab[0:self.max_vocab_size]

    def clip_to_max_len(self, dec_data):
        new_dec_data = [(spk, x[0:self.max_dec_word_len]) for spk, x in dec_data]
        return new_dec_data

    def print_stats(self, name, enc_data, dec_data):
        enc_lens = [len(x) for x in enc_data]
        avg_len = float(np.mean(enc_lens))
        max_len = float(np.max(enc_lens))
        dec_lens = [len(x) for spk, x in dec_data]
        dec_avg_len = float(np.mean(dec_lens))
        dec_max_len = float(np.max(dec_lens))
        print ('%s encoder avg len %.2f max len %.2f of %d lines' % (name, avg_len, max_len, len(enc_data)))
        print ('%s decoder avg len %.2f max len %.2f of %d lines' % (name, dec_avg_len, dec_max_len, len(dec_data)))

    def _parse_file(self, lines):
        """
        :param lines: Each line is a line from the file
        """
        movies = {}
        utt_features = []

        current_movie = []
        current_name = []
        for line in lines:
            if "FILE_NAME" in line:
                if current_movie:
                    movies[current_name] = current_movie
                current_name = line.strip()
                current_movie = []
            else:
                current_movie.append(line.strip())
        if current_movie:
            movies[current_name] = current_movie

        # shuffle movie here.
        shuffle_keys = movies.keys()
        np.random.shuffle(shuffle_keys)
        for key in shuffle_keys:
            utt_features.append("$$$")
            # we only add the line that have full 7 features
            utt_features.extend([l.split("|||") for l in movies[key] if len(l.split("|||")) == 7])

        # tokenize the utterance text
        for feature in utt_features:
            if feature != "$$$":
                feature[self.TEXT_ID] = [tkn.lower() for tkn in self.tokenizer(feature[self.TEXT_ID])]

        # dump the file
        self.dump_data(utt_features)

        return utt_features

    def _create_corpus(self, utt_features, split_size):

        total_size = len(utt_features)
        train_size = int(total_size * split_size[0] / np.sum(split_size))
        valid_size = int(total_size * split_size[1] / np.sum(split_size))

        content_xs = []
        content_ys = []
        # Pointer for decoder input
        print("Begin creating data")
        cur_movie_start_idx = None
        for idx, features in enumerate(utt_features):
            # if we are at "a" in $$$ a b c, ignore the input
            if features == "$$$":
                cur_movie_start_idx = idx
                continue
            content_x = utt_features[max(cur_movie_start_idx+1, idx - self.max_enc_utt_len):idx]
            if len(content_x) <= 0:
                continue
            content_xs.append(content_x)
            content_ys.append((features[self.SPK_ID], features[self.TEXT_ID]))

        # split the data
        self.train_x = content_xs[0: train_size]
        self.train_y = content_ys[0: train_size]
        self.valid_x = content_xs[train_size: train_size + valid_size]
        self.valid_y = content_ys[train_size: train_size + valid_size]
        self.test_x = content_xs[train_size + valid_size:]
        self.test_y = content_ys[train_size + valid_size:]

    def get_corpus(self):
        """
        :return: the corpus in train/valid/test
        """
        return {"train": (self.train_x, self.train_y),
                "valid": (self.valid_x, self.valid_y),
                "test": (self.test_x, self.test_y)}

    def dump_data(self, utt_features):
        if os.path.exists(os.path.join(self._cache_dir, self.cache_name)):
            raise ValueError("File already exists. Abort dumping")
        print("Dumping to %s with %d lines" % (self.cache_name, len(utt_features)))
        with open(os.path.join(self._cache_dir, self.cache_name), "wb") as f:
            for line in utt_features:
                f.write(json.dumps(line) + "\n")

    def load_data(self):
        if not os.path.exists(os.path.join(self._cache_dir, self.cache_name)):
            return None

        with open(os.path.join(self._cache_dir, self.cache_name), "rb") as f:
            lines = f.readlines()
            lines = [json.loads(l.strip()) for l in lines]
        return lines
