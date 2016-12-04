import numpy as np
import nltk

# a helper script to create a test set given the raw data file for comparsion

def parse_file(lines, ctx_window):
    """
    :param lines: Each line is a line from the file
    """
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

    # get all conherent dialogs
    dialogs = []
    for key in shuffle_keys:
        speaker_queue = []
        movie = movies[key]

        for idx, line in enumerate(movie):
            spk, utt = line.strip().split("|||")

            if spk not in [name for name, s_id in speaker_queue] or idx == len(movie)-1:
                if len(speaker_queue) == 2:
                    dialog_len = idx - speaker_queue[0][1]
                    if dialog_len >= ctx_window:
                        dialogs.append((key.replace("FILE_NAME: ", ""), movie[speaker_queue[0][1]:idx-1]))
                        speaker_queue = speaker_queue[1:]

                speaker_queue.append((spk, idx))

    print("Get %d dialogs" % len(dialogs))

    # sort dialog by length and take the top 2000
    dialogs = [dialogs[idx] for idx in np.random.choice(range(len(dialogs)), 2000).tolist()]
    np.random.shuffle(dialogs)

    # write to file
    valid_dialogs = dialogs[0:500]
    test_dialogs = dialogs[500:]

    def dump_file(name, data):
        with open(name, "wb") as f:
            for d_id, dialog in enumerate(data):
                key, lines = dialog
                f.write("FILE_NAME: %d-%s\n" % (d_id, key))
                for l in lines:
                    f.write(l+"\n")

    dump_file("Data/valid.txt", valid_dialogs)
    dump_file("Data/test.txt", test_dialogs)


with open("Data/clean_data_ran.txt") as f:
    lines = f.readlines()

parse_file(lines, ctx_window=4)