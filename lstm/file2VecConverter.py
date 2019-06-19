import torch
import numpy as np
from utilities import utilities
from tqdm import tqdm
from utilities.utilities import Vec
import sys

''' This class takes the original text files, converts each word into a vector specified by the word2vec dictionary
    and writes it to a new file location. This makes subsequent processing faster
'''

class File2VecConverter:

    def __init__(self, doc_file_paths, dict_file_path, unknown_vec):

        self.doc_file_paths = doc_file_paths
        self.dict_file_path = dict_file_path

        self.vector_dict, params = utilities.readVectorsDict(dict_file_path)

        self.unique_vectors = params[0]
        self.vector_size    = params[1]
        self.num_vec_req    = params[2]

        self.vec_replacement = self.unknownWordReplacement(unknown_vec)

        self.num_unknown_words = 0
        self.total_num_words = 0


    def zeroVector(self):
        string = "["
        for i in range(0, self.vector_size):
            string = string + "'0.0'"
            if i < self.vector_size-1:
                string = string + ", "

        return string + "]"



    # returns the vector in form of a parsed string, which is then used as the reverse ditionary key
    def getReverseDictKey(self,vector):

        count = 0
        for element in vector:
            if len(element) > 5 and count > 0:
                vector[count] = element[0:5]
            count += 1
        key = str(vector[1:]).replace("'", "").replace(",", "").replace(" ", "").replace("0", "")

        return key


    # converts original text documents into (pretrained) vector representations and writes them to file directory
    def convertDocuments(self,vec_files):

        count = 0

        for file in tqdm(self.doc_file_paths):

            vectors = self.vecToLine(file)

            with open(vec_files[count], 'w') as f:

                for i in range(0,len(vectors)):
                    f.write(str(vectors[i]).replace("'", "").replace(", ", " ").replace("[", "").replace("]", "")+'\n')

            count += 1

        percent_unknown_words = self.num_unknown_words*100/self.total_num_words
        sys.stderr.flush()
        print("\nTotal number of words {}, unknown words {}, percentage unknown words {}".format(self.total_num_words,
                                                                                               self.num_unknown_words,
                                                                                               percent_unknown_words))


    # looks vector up in word2vec dictionary and writes single line to file
    def vecToLine(self,file):

        vectors = list()
        for line in open(file, encoding="utf8"):
            line = utilities.parseLine(line).split()
            if len(line) > 1:
                for word in line:
                    if len(word) > 0:
                        if word in self.vector_dict:
                            vectors.append(self.vector_dict[word])
                            self.total_num_words = self.total_num_words + 1
                        else:
                            vectors.append(self.zero_vec)
                            self.num_unknown_words = self.num_unknown_words + 1

        return vectors


    # decide how to come up with a vector for unknown words
    def unknownWordReplacement(self, unknown_vec):
        if unknown_vec == vec.zeroVec:
            return self.zeroVector()
