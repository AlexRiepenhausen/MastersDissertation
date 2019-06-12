import torch
import numpy as np
from utilities import utilities

''' This class takes the original text files, converts each word into a vector specified by the word2vec dictionary
    and writes it to a new file location. This makes subsequent processing faster
'''

class File2VecConverter:

    def __init__(self, doc_file_paths, dict_file_path):

        self.doc_file_paths = doc_file_paths
        self.dict_file_path = dict_file_path

        self.unique_vectors = 0
        self.vector_size    = 0
        self.num_vec_req    = 0

        self.vector_dict = self.readVectorsDict()


    # read file containing mapping of words to (pretrained) vectors
    def readVectorsDict(self,reverse=False):

        lines = []
        for line in open(self.dict_file_path, encoding="utf8"):
            lines.append(line)

        self.unique_vectors = np.int_(lines[0].split()[0])
        self.vector_size    = np.int_(lines[0].split()[1])
        self.num_vec_req    = np.int_(lines[0].split()[2])

        vector_dict = dict()
        for i in range(1,self.unique_vectors+1):
            vector = lines[i].split()
            if not reverse:
                vector_dict[vector[0]] = vector[1:]
            if reverse:
                key = self.getReverseDictKey(vector)
                vector_dict[key] = vector[0]

        return vector_dict


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
        print("Writing vectors to files ...")
        for file in self.doc_file_paths:

            vectors = self.vecToLine(file)

            with open(vec_files[count], 'w') as f:

                for i in range(0,len(vectors)):
                    f.write(str(vectors[i]).replace("'", "").replace(", ", " ").replace("[", "").replace("]", "")+'\n')

                # padding at the end if number of vectors smaller than the maximum document size
                if len(vectors) < self.num_vec_req:
                    for i in range(len(vectors), self.num_vec_req):
                        f.write(str(np.zeros(self.vector_size)).replace("[", "").replace("]", "") + '\n')

            count += 1
        print("Done")


    # looks vector up in word2vec dictionary and writes single line to file
    def vecToLine(self,file):

        vectors = list()
        for line in open(file, encoding="utf8"):
            line = utilities.parseLine(line).split()
            if len(line) > 1:
                for word in line:
                    if len(word) > 0:
                        vectors.append(self.vector_dict[word])

        return vectors
