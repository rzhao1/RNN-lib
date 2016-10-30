import os
from nltk import WordPunctTokenizer
import numpy as np


class WordSeqCorpus(object):
    train_x = None
    train_y = None
    valid_x = None
    valid_y = None
    test_x = None
    test_y = None

    def __init__(self, data_dir, data_name, line_thres, split_size):
        """"
        :param line_thres: how many line will be merged as encoding sentensce
        :param split_size: size of training:valid:test

        """

        self._data_dir  = data_dir
        self._data_name = data_name
        self.tokenizer = WordPunctTokenizer().tokenize
        self.line_threshold = line_thres
        self.split_size = split_size

        # try to load from existing file
        if not self.load_data():
            with open(os.path.join(data_dir, data_name), "rb") as f:
                self._parse_file(f.readlines(), split_size)

        # get vocabulary\
        self.vocab = self.get_vocab()

        self.print_stats("train", self.train_x)
        self.print_stats("valid", self.valid_x)
        self.print_stats("test", self.test_x)

    def get_vocab(self):
        # get vocabulary dictionary
        vocab_cnt = {}
        for line in self.train_x:
            for tkn in line.split():
                cnt = vocab_cnt.get(tkn, 0)
                vocab_cnt[tkn] = cnt + 1
        vocab_cnt = [(cnt, key) for key, cnt in vocab_cnt.items()]
        vocab_cnt = sorted(vocab_cnt, reverse=True)
        return [key for cnt, key in vocab_cnt]

    def print_stats(self, name, data):
        avg_len = float(np.mean([len(x.split()) for x in data]))
        print ('%s avg encoder len %.2f of %d lines' % (name, avg_len, len(data)))

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
            utterances.extend([l.split("|||")[1] for l in movies[key]])

        total_size=len(utterances)
        train_size=int(total_size*split_size[0]/10)
        valid_size=int(total_size*split_size[1]/10)

        content_xs = []
        content_ys = []
        # Pointer for decoder input
        print("Begin creating data")
        for idx, (spker, utt) in enumerate(zip(speakers, utterances)):
            # if we are at "a" in $$$ a b c, ignore the input
            if utt == "$$$" or "$$$" in utterances[max(0, idx-self.line_threshold): idx]:
                continue

            content_x = " ".join(utterances[max(0, idx-self.line_threshold):idx])
            if spker == speakers[idx-1]:
                content_x += " #"
            content_xs.append(content_x)
            content_ys.append(utt)

        # split the data
        self.train_x = train_x = content_xs[0: train_size]
        self.train_y = train_y = content_ys[0: train_size]
        self.valid_x = valid_x = content_xs[train_size: train_size+valid_size]
        self.valid_y = valid_y = content_ys[train_size: train_size+valid_size]
        self.text_x = test_x = content_xs[train_size+valid_size:]
        self.test_y = test_y = content_ys[train_size+valid_size:]

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
        with open(file_name, "wb") as f:
            for line in lines:
                f.write(line + "\n")

    def load_data(self):
        if not os.path.exists("train_x.txt"):
            return False

        def load_file(file_name):
            with open(file_name, "rb") as f:
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

def main ():
    data_dir='../Data/'
    data_name='clean_data.txt'

    line_thres=2
    split_size=[7,1,2]
    WordSeqCorpus(data_dir, data_name, line_thres, split_size)

if __name__ == '__main__':
    main()





































