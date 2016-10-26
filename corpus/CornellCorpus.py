import os
import re
import numpy as np
import nltk


class CornellCorpus(object):
    delimiter = "+++$+++"
    dump_name = "cornell_data.txt"


    def __init__(self, data_dir):
        self.dump_path = os.path.join(data_dir, self.dump_name)

        # load the filename and meta info
        self.films = {}
        with open(os.path.join(data_dir, "movie_titles_metadata.txt")) as f:
            lines = f.readlines()
            for l in lines:
                tkns = l.strip().split(self.delimiter)
                self.films[tkns[0].strip()] = tkns[1].strip()

        # load character
        self.characters = {}
        with open(os.path.join(data_dir, "movie_characters_metadata.txt")) as f:
            lines = f.readlines()
            for l in lines:
                tkns = l.strip().split(self.delimiter)
                self.characters[tkns[0]] = {"name":re.sub(r'[^\x00-\x7F]+',' ', tkns[1].strip()),
                                            "movie_id": tkns[2].strip(), "gender": tkns[4].strip()}

        # load all the lines
        self.lines = {}
        self.dialogs = {}
        with open(os.path.join(data_dir, "movie_lines.txt")) as f:
            lines = f.readlines()
            for l in lines:
                tkns = l.strip().split(self.delimiter)
                l_meta = {"char_id": tkns[1].strip(), "movie_id": tkns[2].strip(), "text":tkns[4].strip()}
                self.lines[tkns[0].strip()] = l_meta
                dialog = self.dialogs.get(l_meta["movie_id"], [])
                dialog.append((tkns[0].strip(), l_meta))
                self.dialogs[l_meta["movie_id"]] = dialog
        # sort dialogs
        for movie_id, dialog in self.dialogs.iteritems():
            dialog = sorted(dialog)
            lines = [(meta["char_id"], meta["text"]) for l_id, meta in dialog]
            self.dialogs[movie_id] = self._postprocess_lines(lines)

        print("Loaded %d films dialogs with total %d utterances"
              %(len(self.dialogs), int(np.sum([len(d) for d in self.dialogs.values()]))))

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
            for utt in nltk.sent_tokenize(re.sub(r'[^\x00-\x7F]+',' ', l)):
                split_lines.append((speaker, utt))

        return split_lines

    def dump_dialogs(self, exclude_names):
        dest_f = open(self.dump_path, "wb")
        dump_cnt = 0

        for movie_id, lines in self.dialogs.iteritems():
            # check if in exclude_names
            movie_name = self.films[movie_id]
            if re.sub(r'[\W_]+', "", movie_name.lower()).replace("the", "") in exclude_names:
                continue

            dest_f.write("FILE_NAME: %s\n" % movie_id)
            dump_cnt += 1
            for speaker, line in lines:
                dest_f.write("%s|||%s\n" %(speaker, line))
        dest_f.close()
        print("Save %d films" % dump_cnt)

