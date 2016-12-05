import os
import numpy as np
import json
from nltk.tokenize.regexp import WordPunctTokenizer
from collections import Counter


class WordSeqCorpus(object):
    train_x = None
    train_y = None
    valid_x = None
    valid_y = None
    test_x = None
    test_y = None

    def __init__(self, data_dir, train_name, valid_name, test_name, vocab_name, max_enc_len, max_dec_len, line_thres):
        """"
        :param line_thres: how many line will be merged as encoding sentensce
        :param split_size: size of training:valid:test

        """

        self._data_dir = data_dir
        self._train_path = os.path.join(data_dir, train_name)
        self._valid_path = os.path.join(data_dir, valid_name)
        self._test_path = os.path.join(data_dir, test_name)
        self.vocab_path = os.path.join(self._data_dir, vocab_name)
        self._cache_dir = os.path.join(data_dir, train_name.replace(".txt", "_") + self.__class__.__name__)
        if not os.path.exists(self._cache_dir):
            os.mkdir(self._cache_dir)

        self.tokenizer = WordPunctTokenizer().tokenize
        self.line_threshold = line_thres
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        # try to load from existing file
        if not self.load_data():
            self._parse_file()

        # clip data
        self.train_x, self.train_y = self.clip_to_max_len(self.train_x, self.train_y)
        self.valid_x, self.valid_y = self.clip_to_max_len(self.valid_x, self.valid_y)
        self.test_x, self.test_y = self.clip_to_max_len(self.test_x, self.test_y)

        # get vocabulary\
        self.get_vocab()
        self.print_vocab_stats()

        self.print_stats("TRAIN", self.train_x, self.train_y)
        self.print_stats("VALID", self.valid_x, self.valid_y)
        self.print_stats("TEST", self.test_x, self.test_y)

    def print_vocab_stats(self):
        # get vocabulary dictionary
        all_words = []
        for line in self.train_x:
            all_words.extend(line.split())
        for line in self.valid_x:
            all_words.extend(line.split())
        for line in self.test_x:
            all_words.extend(line.split())

        vocab_cnt = Counter(all_words).most_common()
        total = float(np.sum([cnt for key, cnt in vocab_cnt]))
        valid = float(np.sum([cnt for key, cnt in vocab_cnt if key in self.vocab]))
        print("Raw vocab cnt %d with valid ratio %f" % (len(vocab_cnt), valid / total))

    def get_vocab(self):
        with open(self.vocab_path, "rb") as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
        self.vocab = lines

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

    def _parse_file(self):
        """
        :param lines: Each line is a line from the file
        """

        def read_data(name):
            with open(name, 'rb') as f:
                lines = f.readlines()
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
            for key in movies.keys():
                speakers.append("$$$")
                utterances.append("$$$")
                speakers.extend([l.split("|||")[0] for l in movies[key]])
                utterances.extend([" ".join(self.tokenizer(l.split("|||")[1])).lower() for l in movies[key]])

            content_xs = []
            content_ys = []
            # Pointer for decoder input
            print("Begin creating data")
            for idx, (spker, utt) in enumerate(zip(speakers, utterances)):
                # if we are at "a" in $$$ a b c, ignore the input
                if utt == "$$$" or "$$$" in utterances[max(0, idx - self.line_threshold): idx]:
                    continue

                content_x = " <t> ".join(utterances[max(0, idx - self.line_threshold):idx])
                if spker != speakers[idx - 1]:
                    content_x += " #"
                content_xs.append(content_x)
                content_ys.append(utt)
            return content_xs, content_ys

        # split the data
        self.train_x, self.train_y = read_data(self._train_path)
        self.valid_x, self.valid_y = read_data(self._valid_path)
        self.test_x, self.test_y = read_data(self._test_path)

        # begin dumpping data to file
        self.dump_data("train_x.txt", self.train_x)
        self.dump_data("train_y.txt", self.train_y)
        self.dump_data("valid_x.txt", self.valid_x)
        self.dump_data("valid_y.txt", self.valid_y)
        self.dump_data("test_x.txt", self.test_x)
        self.dump_data("test_y.txt", self.test_y)

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

        def load_file(file_name, use_cache_prefix=True):
            path = os.path.join(self._cache_dir, file_name) if use_cache_prefix else file_name
            with open(path, "rb") as f:
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


class HybridSeqCorpus(object):
    train_x = None
    train_y = None
    valid_x = None
    valid_y = None
    test_x = None
    test_y = None
    data_lines = None

    # class variable
    vocab = None
    movie_profile = None

    def __init__(self, data_dir, data_name, split_size, max_vocab_size, context_size):
        """"
        :param line_thres: how many line will be merged as encoding sentensce
        :param split_size: size of training:valid:test

        """

        self._data_dir = data_dir
        self._data_name = data_name
        self._cache_dir = os.path.join(data_dir, data_name.replace(".txt", "_") + self.__class__.__name__)
        if not os.path.exists(self._cache_dir):
            os.mkdir(self._cache_dir)

        self.tokenizer = WordPunctTokenizer().tokenize
        self.context_size = context_size
        self.split_size = split_size
        self.max_vocab_size = max_vocab_size
        # try to load from existing file
        if not self.load_data():
            with open(os.path.join(data_dir, data_name), "rb") as f:
                self._parse_file(f.readlines(), split_size)

        # get vocabulary\
        self.vocab = self.get_vocab()

        self.print_stats("TRAIN", self.train_x, self.train_y)
        self.print_stats("VALID", self.valid_x, self.valid_y)
        self.print_stats("TEST", self.test_x, self.test_y)

    def get_vocab(self):
        # get vocabulary dictionary
        all_words = []
        for _, _, utt in self.data_lines:
            all_words.extend(utt.split())

        vocab_cnt = Counter(all_words).most_common()
        vocab = [w for w, cnt in vocab_cnt]
        cnts = [cnt for w, cnt in vocab_cnt]
        total = float(np.sum(cnts))
        valid = float(np.sum(cnts[0:self.max_vocab_size]))
        print("Raw vocab cnt %d with valid ratio %f" % (len(cnts), valid/total))
        return vocab[0:self.max_vocab_size]

    def print_stats(self, name, enc_data, dec_data):
        print ('%s encoder %d lines' % (name, len(enc_data)))
        print ('%s decoder %d lines' % (name, len(dec_data)))

    def _parse_file(self, lines, split_size):
        """
        :param lines: Each line is a line from the file
        """
        data_lines = []
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
            for line in movies[key]:
                speaker = line.split("|||")[0]
                utt = " ".join(self.tokenizer(line.split("|||")[1])).lower()
                data_lines.append([key, speaker, utt])

        total_size = len(data_lines)
        train_size = int(total_size * split_size[0] / np.sum(split_size))
        valid_size = int(total_size * split_size[1] / np.sum(split_size))

        content_xs = []
        content_ys = []
        movie_word_cnt = {}
        # Pointer for decoder input
        print("Begin creating data")
        for idx, (movie_key, speaker, utt) in enumerate(data_lines):
            # collect all words of each movie as movie profile
            movie_words = movie_word_cnt.get(movie_key, [])
            movie_words.extend(utt.split(" "))
            movie_word_cnt[movie_key] = movie_words

            # x include: movie key, the ids of previous context_size utterance, the previous utterance id
            # y include: an id
            # find the previous utt. If this is the first/second utt, we ignore this line
            if idx < 2 or data_lines[idx-1][0] != movie_key or data_lines[idx-2][0] != movie_key:
                continue
            prev_utt = idx - 1

            context_utt = []
            for t_id in range(idx-2, np.maximum(-1, idx-self.context_size-2), -1):
                if data_lines[t_id][0] != movie_key:
                    break
                context_utt.append(t_id)
            context_utt = context_utt[::-1]
            content_xs.append((context_utt, prev_utt))
            content_ys.append(idx)

        # count the words
        self.movie_profile = {key: Counter(words).most_common() for key, words in movie_word_cnt.items()}
        self.data_lines = data_lines

        # split the data
        self.train_x = train_x = content_xs[0: train_size]
        self.train_y = train_y = content_ys[0: train_size]
        self.valid_x = valid_x = content_xs[train_size: train_size + valid_size]
        self.valid_y = valid_y = content_ys[train_size: train_size + valid_size]
        self.test_x = test_x = content_xs[train_size + valid_size:]
        self.test_y = test_y = content_ys[train_size + valid_size:]

        # begin dumpping data to file
        self.dump_data("movie_profile.dict", self.movie_profile)
        self.dump_data("lines.txt", self.data_lines)
        self.dump_data("train_x.txt", train_x)
        self.dump_data("train_y.txt", train_y)
        self.dump_data("valid_x.txt", valid_x)
        self.dump_data("valid_y.txt", valid_y)
        self.dump_data("test_x.txt", test_x)
        self.dump_data("test_y.txt", test_y)

    def dump_data(self, file_name, data):
        if os.path.exists(file_name):
            raise ValueError("File already exists. Abort dumping")
        print("Dumping to %s with %d lines" % (file_name, len(data)))
        with open(os.path.join(self._cache_dir, file_name), "wb") as f:
            if type(data) is list:
                for line in data:
                    f.write(json.dumps(line) + "\n")
            else:
                f.write(json.dumps(data))

    def load_data(self):
        if not os.path.exists(os.path.join(self._cache_dir, "train_x.txt")):
            return False

        def load_file(file_name):
            with open(os.path.join(self._cache_dir, file_name), "rb") as f:
                lines = f.readlines()
                lines = [json.loads(l.strip()) for l in lines]
            return lines

        def load_dict(file_name):
            with open(os.path.join(self._cache_dir, file_name), "rb") as f:
                lines = f.readlines()
            return json.loads(lines[0].strip())

        print("Loaded from cache")
        self.movie_profile = load_dict("movie_profile.dict")
        self.data_lines = load_file("lines.txt")
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


class FutureSeqCorpus(object):
    train_data = None
    valid_data = None
    test_data = None
    data_lines = None

    # class variable
    vocab = None
    movie_profile = None

    def __init__(self, data_dir, data_name, valid_name, test_name, vocab_name, context_size):

        self._data_dir = data_dir
        self._data_name = data_name
        self._train_path = os.path.join(self._data_dir, data_name)
        self._valid_path = os.path.join(self._data_dir, valid_name)
        self._test_path = os.path.join(self._data_dir, test_name)
        self._vocab_path = os.path.join(self._data_dir, vocab_name)

        self._cache_dir = os.path.join(data_dir, data_name.replace(".txt", "_") + self.__class__.__name__)
        if not os.path.exists(self._cache_dir):
            os.mkdir(self._cache_dir)

        self.tokenizer = WordPunctTokenizer().tokenize
        self.context_size = context_size
        # try to load from existing file
        if not self.load_data():
            self._parse_file()

        # get vocabulary\
        self.get_vocab()

        self.print_stats("TRAIN", self.train_data)
        self.print_stats("VALID", self.valid_data)
        self.print_stats("TEST", self.test_data)

    def get_vocab(self):
        # read vocab file
        with open(self._vocab_path, "rb") as f:
            lines = f.readlines()
        self.vocab = [l.strip() for l in lines]

        # get vocabulary dictionary
        all_words = []
        for _, _, utt in self.data_lines:
            all_words.extend(utt.split())

        vocab_set = set(self.vocab)
        vocab_cnt = Counter(all_words).most_common()
        total = float(np.sum([cnt for w, cnt in vocab_cnt]))
        valid = float(np.sum([cnt for w, cnt in vocab_cnt if w in vocab_set]))
        print("Raw vocab cnt %d with valid ratio %f" % (len(vocab_cnt), valid/total))

    def print_stats(self, name, data):
        print ('%s data %d lines' % (name, len(data)))
        x_len = [len(self.data_lines[x][2].split()) for ctx, x, y, z in data]
        print ('%s x avg len %.2f max len %.2f of %d lines' % (name, np.mean(x_len), np.max(x_len), len(x_len)))
        y_len = [len(self.data_lines[y][2].split()) for ctx, x, y, z in data]
        print ('%s y avg len %.2f max len %.2f of %d lines' % (name, np.mean(y_len), np.max(y_len), len(y_len)))

    def _parse_file(self):
        """
        :param lines: Each line is a line from the file
        """
        # read train_valid and test
        def read_file(name):
            with open(name, "rb") as f:
                lines = f.readlines()
                lines = [l.strip() for l in lines]
            return lines

        train_lines = read_file(self._train_path)
        valid_lines = read_file(self._valid_path)
        test_lines = read_file(self._test_path)

        lines = train_lines + valid_lines +test_lines

        data_lines = []
        movies = {}

        current_movie = []
        current_name = None
        movie_keys = []
        train_keys = set()
        valid_keys = set()
      
        for idx, line in enumerate(lines):
            if "FILE_NAME" in line:
                if current_movie:
                    movies[current_name] = current_movie
                current_name = line.strip()
                current_movie = []
                movie_keys.append(current_name)
                if idx < len(train_lines):
                    train_keys.add(current_name)
                elif idx < len(train_lines) + len(valid_lines):
                    valid_keys.add(current_name)
            else:
                current_movie.append(line.strip())
        if current_movie:
            movies[current_name] = current_movie

        # shuffle movie here.
        for key in movie_keys:
            for line in movies[key]:
                speaker = line.split("|||")[0]
                utt = " ".join(self.tokenizer(line.split("|||")[1])).lower()
                data_lines.append([key, speaker, utt])

        # Pointer for decoder input
        print("Begin creating data")
        idx = 0
        self.train_data = []
        self.valid_data = []
        self.test_data = []

        while True:
            if idx >= len(data_lines):
                break

            key, speaker, utt = data_lines[idx]
            # x include: the previous utterance id and context
            # y include: the current utterance id
            # z include: the next uttearnce id
            # find the previous utt. If this is the first/second utt, we ignore this line

            if idx < 2 or data_lines[idx-1][0] != key or data_lines[idx-2][0] != key:
                idx += 1
                continue
            # too late
            if idx == len(data_lines)-1 or data_lines[idx+1][0] != key:
                idx += 1
                continue

            context_utt = []
            for t_id in range(idx-2, np.maximum(-1, idx-self.context_size-2), -1):
                if data_lines[t_id][0] != key:
                    break
                context_utt.append(t_id)

            context_utt = context_utt[::-1]
            new_line = (context_utt, idx-1, idx, idx+1)
            if key in train_keys:
                self.train_data.append(new_line)
            elif key in valid_keys:
                self.valid_data.append(new_line)
            else:
                self.test_data.append(new_line)

            idx += 1

        # count the words
        self.data_lines = data_lines

        # begin dumpping data to file
        self.dump_data("lines.txt", self.data_lines)
        self.dump_data("train.txt", self.train_data)
        self.dump_data("valid.txt", self.valid_data)
        self.dump_data("test.txt", self.test_data)

    def dump_data(self, file_name, data):
        if os.path.exists(file_name):
            raise ValueError("File already exists. Abort dumping")
        print("Dumping to %s with %d lines" % (file_name, len(data)))
        with open(os.path.join(self._cache_dir, file_name), "wb") as f:
            if type(data) is list:
                for line in data:
                    f.write(json.dumps(line) + "\n")
            else:
                f.write(json.dumps(data))

    def load_data(self):
        if not os.path.exists(os.path.join(self._cache_dir, "train.txt")):
            return False

        def load_file(file_name):
            with open(os.path.join(self._cache_dir, file_name), "rb") as f:
                lines = f.readlines()
                lines = [json.loads(l.strip()) for l in lines]
            return lines

        def load_dict(file_name):
            with open(os.path.join(self._cache_dir, file_name), "rb") as f:
                lines = f.readlines()
            return json.loads(lines[0].strip())

        print("Loaded from cache")
        self.data_lines = load_file("lines.txt")
        self.train_data = load_file("train.txt")
        self.valid_data = load_file("valid.txt")
        self.test_data = load_file("test.txt")
        return True

    def get_corpus(self):
        """
        :return: the corpus in train/valid/test
        """
        return {"train": self.train_data,
                "valid": self.valid_data,
                "test": self.test_data}


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
