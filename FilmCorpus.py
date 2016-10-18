import os
import fnmatch
import re

class FilmCorpus(object):
    def __init__(self, corpus_path):
        """
        :param corpus_path: the folder that contains the data
        """
        self._path = corpus_path

        film_files = []
        for root, _, file_names in os.walk(self._path):
            for filename in fnmatch.filter(file_names, '*.txt'):
                film_files.append(os.path.join(root, filename))
        print("Loaded %d films" % len(film_files))

        dialogs = {}
        utt_cnt = 0
        for f_idx, file in enumerate(film_files):
            if f_idx % 500 == 0:
                print("Finished Processing files up to %d" % f_idx)
            with open(file, 'rb') as f:
                raw_lines = f.readlines()
                speaker = None
                lines = []
                for line in raw_lines:
                    # remove stuff in [] or () and strip
                    line = re.sub("[\(\[].*?[\)\]]", "", line).strip()
                    if line:
                        if line.isupper():
                            speaker = line
                        elif speaker is not None:
                            lines.append((speaker, line))
                utt_cnt += len(lines)
                dialogs[file] = lines
        self.utt_cnt = utt_cnt
        self.dialogs = dialogs
        print("Done parsing all films with %d utts" % self.utt_cnt)

    def pprint_dialog(self, idx):
        lines = self.dialogs.values()[idx]
        for l in lines:
            print("%s: %s" % l)
