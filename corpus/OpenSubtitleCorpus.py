import os
import fnmatch
import re
import numpy as np
import nltk
from collections import OrderedDict


class OpenSubtitleCorpus(object):
    dump_name = "clean_data.txt"

    def __init__(self, corpus_path):
        """
        :param corpus_path: the folder that contains the data
        """
        self._path = corpus_path
        self.dump_path = os.path.join(self._path, self.dump_name)

        self.dialogs = self.load_dialogs()

        if self.dialogs is None:
            film_files = []
            for root, _, file_names in os.walk(self._path):
                for filename in fnmatch.filter(file_names, '*.txt'):
                    film_files.append(os.path.join(root, filename))
            print("Loaded %d files" % len(film_files))

            dialogs = OrderedDict()
            for f_idx, file in enumerate(film_files):
                if f_idx % 500 == 0:
                    print("Finished Processing files up to %d" % f_idx)
                with open(file, 'rb') as f:
                    file_key = os.path.basename(file)
                    # skip the film that is already processed before
                    if file_key in dialogs.keys():
                        continue
                    raw_lines = f.readlines()
                    speaker = "A"
                    lines = []
                    for line in raw_lines:
                        # remove stuff in [] or () and strip
                        line = line.strip()
                        if line:
                            lines.append((speaker, re.sub('\s+', ' ', line)))
                            if speaker == "A":
                                speaker = "B"
                            else:
                                speaker = "A"
                    if len(lines) > 0:
                        dialogs[file_key] = self._postprocess_lines(lines)
            self.dialogs = dialogs
            self.dump_dialogs()

        self.utt_cnt = np.sum([len(d) for d in self.dialogs.values()])
        print("Done parsing %d films with %d utts" % (len(self.dialogs.keys()), int(self.utt_cnt)))

    def remove_dupliate(self):
        temp = OrderedDict()
        for film_key, dialogs in self.dialogs.iteritems():
            new_film_key = os.path.basename(film_key)
            if new_film_key not in temp.keys() and len(dialogs) > 0:
                temp[new_film_key] = dialogs
        self.dialogs = temp
        self.utt_cnt = np.sum([len(d) for d in self.dialogs.values()])
        print("Done parsing %d films with %d utts" % (len(self.dialogs.keys()), int(self.utt_cnt)))

    def get_protagonists(self):
        def collect_character_info(dialog):
            char_word_cnt = {}
            for char, line in dialog:
                cnt = char_word_cnt.get(char, 0)
                char_word_cnt[char] = cnt + len(line.split())
            temp = [(cnt, char) for char, cnt in char_word_cnt.iteritems()]
            return sorted(temp, reverse=True)

        protagonist_dict = {}
        for file, d in self.dialogs.iteritems():
            char_cnt = collect_character_info(d)
            major = char_cnt[0]
            background = (np.sum([cnt for cnt, char in char_cnt[1:]]), "background")
            protagonist_dict[file] = (major, background)

        return protagonist_dict

    def _postprocess_lines(self, lines):
        merge_lines = []
        prev_speaker = None

        # merge same speaker speech into the same line
        for cur_speaker, l in lines:
            if cur_speaker == prev_speaker:
                merge_lines[-1] = (prev_speaker, merge_lines[-1][1] + " " + l)
            else:
                merge_lines.append((cur_speaker, l))
                prev_speaker = cur_speaker

        # split the same speaker speech into sentences (utterances)
        split_lines = []
        for speaker, l in merge_lines:
            for utt in nltk.sent_tokenize(l):
                split_lines.append((speaker, utt))

        return split_lines

    def pprint_dialog(self, idx):
        lines = self.dialogs.values()[idx]
        for l in lines:
            print("%s: %s" % l)

    def dump_dialogs(self):
        dest_f = open(self.dump_path, "wb")
        for file_name, lines in self.dialogs.iteritems():
            dest_f.write("FILE_NAME: %s\n" % file_name)
            for speaker, line in lines:
                dest_f.write("%s|||%s\n" %(speaker, line))
        dest_f.close()

    def load_dialogs(self):
        if os.path.exists(self.dump_path):
            dialogs = OrderedDict()
            with open(self.dump_path, "rb") as f:
                all_lines = f.readlines()
                file_name = None
                lines = None

                for l in all_lines:
                    l = l.strip()
                    if "FILE_NAME" in l:
                        if lines is not None:
                            dialogs[file_name] = lines

                        file_name = l.replace("FILE_NAME: ", "")
                        lines = []

                    elif file_name is not None:
                        tokens = l.split("|||")
                        speaker, line = tokens
                        lines.append((speaker, line))
                # save the last dialog
                if lines is not None and len(lines) > 0:
                    dialogs[file_name] = lines
            return dialogs
        else:
            return None

OpenSubtitleCorpus("../Data/open_subtitle")