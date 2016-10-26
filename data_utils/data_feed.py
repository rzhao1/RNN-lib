import collections
import os
from nltk.tokenize.regexp import WordPunctTokenizer
import numpy as np
import re
import cPickle as pkl


class VsCorpus(object):
    # spk_id: (is_male, is_friend)
    spk_to_relation = {1: (False, False),
                       2: (True, True),
                       3: (True, True),
                       4: (True, True),
                       7: (False, False),
                       8: (False, False),
                       9: (True, False),
                       10: (False, True),
                       13: (False, True),
                       14: (True, False),
                       15: (False, True),
                       16: (True, False)}

    def __init__(self, data_dir):
        self._path = data_dir
        self.tokenizer = WordPunctTokenizer().tokenize
        self.nvb_size = None
        # read train data
        with open(os.path.join(data_dir, "train_x.txt"), "rb") as f:
            self.raw_train_x, self.train_list = self._parse_file(f.readlines())
        with open(os.path.join(data_dir, "train_y.txt"), "rb") as f:
            self.train_y, _ = self._parse_file(f.readlines(), is_y=True)
        # read valid data
        with open(os.path.join(data_dir, "cv_x.txt"), "rb") as f:
            self.raw_valid_x, self.valid_list = self._parse_file(f.readlines())
        with open(os.path.join(data_dir, "cv_y.txt"), "rb") as f:
            self.valid_y, _ = self._parse_file(f.readlines(), is_y=True)
        # read test data
        with open(os.path.join(data_dir, "test_x.txt"), "rb") as f:
            self.raw_test_x, self.test_list = self._parse_file(f.readlines())
        with open(os.path.join(data_dir, "test_y.txt"), "rb") as f:
            self.test_y, _ = self._parse_file(f.readlines(), is_y=True)

        # build vocabulary
        self._build_vocab(tokenizer=self.tokenizer)

        # convert input_x into ids
        self.train_x = self._transform_x_to_id(self.raw_train_x)
        self.valid_x = self._transform_x_to_id(self.raw_valid_x)
        self.test_x = self._transform_x_to_id(self.raw_test_x)

        self.train_y = self._transform_y_to_id(self.train_y)
        self.valid_y = self._transform_y_to_id(self.valid_y)
        self.test_y = self._transform_y_to_id(self.test_y)

        self.train_feat = None
        self.valid_feat = None
        self.test_feat = None
        self.feat_size = 3782

        # do a sanity check
        for key in self.train_x.keys():
            assert len(self.train_x[key]) == len(self.train_y[key])

        # report some stats
        self.print_stats("Train", self.train_x)
        self.print_stats("Valid", self.valid_x)
        self.print_stats("Test", self.test_x)

    def _parse_file(self, lines, is_y=False):
        """
        :param lines: Each line is a line from the file
        :return: A dictionary of list. Each entry in dictionary is dialog, second level is clause
        """
        result = {}
        curr_file = None
        cur_gender = None
        cur_relation= None
        file_list = []
        for idx, line in enumerate(lines):
            if "#" in line:
                curr_file = line.strip()
                # get gender and relation
                spk_id = curr_file[curr_file.find('D')+1:curr_file.find('S')]
                cur_gender, cur_relation = self.spk_to_relation[int(spk_id)]
                file_list.append(curr_file)
            else:
                if is_y:
                    features = line.strip()
                else:
                    features = re.split(r'\t+', line.strip())
                    # convert nvb to int
                    nvb = features[1].split()[1:]  # NO rapport
                    nvb = [int(n) for n in nvb]
                    # add gender and relation
                    nvb += [int(cur_gender), int(cur_relation)]
                    if self.nvb_size is None:
                        self.nvb_size = len(nvb)
                    features[1] = nvb

                dialog = result.get(curr_file, [])
                dialog.append(features)
                result[curr_file] = dialog
        return result, file_list

    def _build_vocab(self, tokenizer=None):
        all_words = []
        for dialog in self.raw_train_x.values():
            for line, nvb in dialog:
                all_words.extend(line.split() if tokenizer is None else tokenizer(line))

        counter = collections.Counter(all_words)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        # add EOD (end of dialog)
        words = ["EOD", "<unk>"] + list(words)
        self.vocabs = words
        self.word_to_id = dict(zip(words, range(len(words))))
        self.vocab_size = len(self.word_to_id)
        print("Vocabulary size is %d with EOD as %d" % (len(self.word_to_id), self.word_to_id["EOD"]))

    def _transform_x_to_id(self, x):
        id_x = {}
        for key, dialog in x.iteritems():
            id_dialog = []
            for line, nvb in dialog:
                id_line = [self.word_to_id.get(w, self.word_to_id["<unk>"]) for w in self.tokenizer(line)]
                id_dialog.append((id_line, nvb))
            id_x[key] = id_dialog
        return id_x

    def _transform_y_to_id(self, y):
        id_y = {}
        for key, dialog in y.iteritems():
            id_dialog = []
            for line in dialog:
                id_dialog.append(int(line))
            id_y[key] = id_dialog
        return id_y

    def print_stats(self, name, dialogs):
        print("Loaded %d %s dialogs" % (len(dialogs), name))
        all_lens = [len(d) for d in dialogs.values()]
        print("Avg dialog len is %.2f, Max dialog len is %d" % (np.average(all_lens), np.max(all_lens)))
        all_utts = []
        for d in dialogs.values():
            all_utts.extend(d)
        all_utt_len = [len(utt) for utt, nvb in all_utts]
        print("Avg utterance len is %.2f, Max utterance len is %d" % (np.average(all_utt_len), np.max(all_utt_len)))

    def read_extra_feature(self, feat_dir):
        # read the extra feature from the directory
        cache_dir = os.path.join(feat_dir, "feat.p")
        if os.path.exists(cache_dir):
            print("Read from cache")
            self.train_feat, self.valid_feat, self.test_feat = pkl.load(open(cache_dir, 'rb'))
        else:
            print("Reading extra feature")
            self.train_feat = self._parse_extra_feature(self.raw_train_x, self.train_list, os.path.join(feat_dir, 'train_x.csv'))
            self.valid_feat = self._parse_extra_feature(self.raw_valid_x, self.valid_list, os.path.join(feat_dir, 'cv_x.csv'))
            self.test_feat = self._parse_extra_feature(self.raw_test_x, self.test_list, os.path.join(feat_dir, 'test_x.csv'))
            pkl.dump((self.train_feat, self.valid_feat, self.test_feat), open(cache_dir, 'wb'))
        print("Done reading extra feature")

    def _parse_extra_feature(self, data_x, file_list, file_dir):
        with open(file_dir, 'rb') as f:
            lines = f.readlines()

        # strip down
        lines = [l.strip().split(',') for l in lines]
        lines = lines[1:] # remove header

        # collect features
        results = {}
        global_idx = 0
        for file in file_list:
            f_x_len = len(data_x[file])
            f_feat = []
            for _ in range(f_x_len):
                l = lines[global_idx]
                sent, feat = l[0], l[-1*self.feat_size:]
                try:
                    feat = [float(col) for col in feat]
                except ValueError:
                    print feat
                f_feat.append(feat)
                global_idx += 1

            results[file] = f_feat

        return results









