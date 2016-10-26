import os

from nltk import WordPunctTokenizer
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


    def _parse_file(self, lines,line_thres,split_size):
        """
        :param lines: Each line is a line from the file
        """

        utterance=[]
        speaker=[]

        train_x=open('train_x.txt','w')
        train_y=open('train_y.txt','w')
        valid_x=open('valid_x.txt','w')
        valid_y=open('valid_y.txt','w')
        test_x=open('test_x.txt','w')
        test_y=open('test_y.txt','w')
        for idx, line in enumerate(lines):
            if 'FILE_NAME' in line:
                speaker.append("$$$")
                utterance.append("$$$")
            else:
                line=line.replace('\n','')
                speaker_utterance=line.split("|||")
                utterance.append(speaker_utterance[1])
                speaker.append(speaker_utterance[0])

        total_size=len(utterance)
        train_size=total_size*split_size[0]/10
        valid_size=total_size*split_size[1]/10
        test_size=total_size*split_size[2]/10

        train_char=[]
        test_char=[]
        valid_char=[]
        #Pointer for decoder input
        ptr=0
        while  ptr+1<total_size:
                Content_x = ""
                Content_y = ""
                encoder_speaker = ""
                decoder_speaker = ""
                if(speaker[ptr]=="$$$"):
                    ptr +=1
                    for i in range(0,line_thres):
                        Content_x+= ' '+utterance[i+ptr]
                    ptr += line_thres
                    encoder_speaker=speaker[ptr-1]
                    decoder_speaker=speaker[ptr]
                    if encoder_speaker==decoder_speaker:
                        Content_x+=' #'
                    Content_y=utterance[ptr]

                    if(ptr<=train_size):
                        train_x.write("$$$\n")
                        train_y.write("$$$\n")
                        train_x.write(Content_x+'\n')
                        train_y.write(Content_y+'\n')
                        train_char.append(len(Content_x.split(' ')))

                    elif ptr>train_size and ptr<train_size+valid_size:
                        valid_x.write("$$$\n")
                        valid_y.write("$$$\n")
                        valid_x.write(Content_x+'\n')
                        valid_y.write(Content_y+'\n')
                        valid_char.append(len(Content_x.split(' ')))
                    else :
                        test_x.write("$$$\n")
                        test_y.write("$$$\n")
                        test_x.write(Content_x+'\n')
                        test_y.write(Content_y+'\n')
                        test_char.append(len(Content_x.split(' ')))


                else:
                     start=ptr-line_thres+1
                     for i in range(0, line_thres ):
                         Content_x += ' '+utterance[i + start]
                     ptr += 1
                     encoder_speaker = speaker[ptr - 1]
                     decoder_speaker = speaker[ptr]
                     if encoder_speaker==decoder_speaker:
                         Content_x+=' #'
                     Content_y = utterance[ptr]

                     if (ptr <= train_size):
                         train_x.write(Content_x + '\n')
                         train_y.write(Content_y + '\n')
                         train_char.append(len(Content_x.split(' ')))

                     elif ptr > train_size and ptr < train_size + valid_size:
                         valid_x.write(Content_x + '\n')
                         valid_y.write(Content_y + '\n')
                         valid_char.append(len(Content_x.split(' ')))
                     else:
                         test_x.write(Content_x + '\n')
                         test_y.write(Content_y + '\n')
                         test_char.append(len(Content_x.split(' ')))

        print ('Average size of characters in training encoder sentence ' + str(sum(train_char)/train_size))
        print ('Average size of characters in validation encoder sentence ' + str(sum(train_char) / valid_size))
        print ('Average size of characters in testing encoder sentence ' + str(sum(train_char) / (total_size-train_size-valid_size)))


def main ():

    data_dir='/home/ranzhao1/PycharmProjects/DeepLearningProject/data/'
    data_name='filmdata.txt'
    line_thres=2
    split_size=[7,1,2]
    m=data_splitter(data_dir,data_name,line_thres,split_size)

if __name__ == '__main__':
        main()





































