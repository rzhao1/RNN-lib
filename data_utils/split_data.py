import os

from nltk import WordPunctTokenizer
import numpy as np
import matplotlib.pyplot as plt


class data_splitter(object):

    def __init__(self, data_dir, data_name,line_thres,split_size):
        """"
        :param line_thres: how many line will be merged as encoding sentensce
        :param split_size: size of training:valid:test

        """

        self._data_dir = data_dir
        self._data_name= data_name
        self.tokenizer = WordPunctTokenizer().tokenize
        self.line_thres=line_thres
        self.split_size=split_size

        with open(os.path.join(data_dir, data_name), "rb") as f:
            self._parse_file(f.readlines(),line_thres,split_size)


    def _parse_file(self, lines, line_thres, split_size):
        """
        :param lines: Each line is a line from the file
        """

        utterances = []
        speakers = []
        movies = {}

        train_x=open('train_x.txt','w')
        train_y=open('train_y.txt','w')
        valid_x=open('valid_x.txt','w')
        valid_y=open('valid_y.txt','w')
        test_x=open('test_x.txt','w')
        test_y=open('test_y.txt','w')

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
        train_size=total_size*split_size[0]/10
        valid_size=total_size*split_size[1]/10
        test_size=total_size*split_size[2]/10

        train_char=[]
        test_char=[]
        valid_char=[]

        # Pointer for decoder input
        ptr=0
        print("Begin creating data")
        while ptr+1<total_size:
            content_x = ""
            content_y = ""
            encoder_speaker = ""
            decoder_speaker = ""

            if (speakers[ptr]=="$$$"):
                ptr +=1
                for i in range(0,line_thres):
                    content_x += ' '+utterances[i+ptr]
                ptr += line_thres
                encoder_speaker=speakers[ptr-1]
                decoder_speaker=speakers[ptr]
                if encoder_speaker == decoder_speaker:
                    content_x += ' #'

                content_y=utterances[ptr]

                if(ptr<=train_size):
                    train_x.write("$$$\n")
                    train_y.write("$$$\n")
                    train_x.write(content_x+'\n')
                    train_y.write(content_y+'\n')
                    train_char.append(len(content_x.split(' ')))

                elif ptr>train_size and ptr<train_size+valid_size:
                    valid_x.write("$$$\n")
                    valid_y.write("$$$\n")
                    valid_x.write(content_x+'\n')
                    valid_y.write(content_y+'\n')
                    valid_char.append(len(content_x.split(' ')))
                else :
                    test_x.write("$$$\n")
                    test_y.write("$$$\n")
                    test_x.write(content_x+'\n')
                    test_y.write(content_y+'\n')
                    test_char.append(len(content_x.split(' ')))

            else:
                start = ptr-line_thres+1
                for i in range(0, line_thres):
                    content_x += ' '+utterances[i + start]

                ptr += 1
                encoder_speaker = speakers[ptr - 1]
                decoder_speaker = speakers[ptr]
                if encoder_speaker == decoder_speaker:
                    content_x+=' #'
                content_y = utterances[ptr]

                if ptr <= train_size:
                    train_x.write(content_x + '\n')
                    train_y.write(content_y + '\n')
                    train_char.append(len(content_x.split(' ')))

                elif ptr > train_size and ptr < train_size + valid_size:
                    valid_x.write(content_x + '\n')
                    valid_y.write(content_y + '\n')
                    valid_char.append(len(content_x.split(' ')))
                else:
                    test_x.write(content_x + '\n')
                    test_y.write(content_y + '\n')
                    test_char.append(len(content_x.split(' ')))

        train_x.close()
        train_y.close()
        valid_x.close()
        valid_y.close()
        test_y.close()
        print ('Average size of characters in training encoder sentence %.2f' % float(np.mean(train_char)))
        print ('Average size of characters in validation encoder sentence %.2f' % float(np.mean(valid_char)))
        print ('Average size of characters in testing encoder sentence %.2f'  % float(np.mean(test_char)))


def main ():
    data_dir='/home/ranzhao1/PycharmProjects/DeepLearningProject/data/'
    data_name='filmdata.txt'

    line_thres=2
    split_size=[7,1,2]
    m=data_splitter(data_dir,data_name,line_thres,split_size)

if __name__ == '__main__':
        main()





































