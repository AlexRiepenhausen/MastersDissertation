import torch
import numpy as np
from utilities import utilities
from tqdm import tqdm

''' This class takes the original text files, converts each word into a vector specified by the word2vec dictionary
    and writes it to a new file location. This makes subsequent processing faster
'''

class File2VecConverter:

    def __init__(self, doc_file_paths, dict_file_path):

        self.doc_file_paths = doc_file_paths
        self.dict_file_path = dict_file_path

        self.vector_dict, params = utilities.readVectorsDict(dict_file_path)

        self.unique_vectors = params[0]
        self.vector_size    = params[1]
        self.num_vec_req    = params[2]


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
        for file in tqdm(self.doc_file_paths):

            vectors = self.vecToLine(file)

            with open(vec_files[count], 'w') as f:

                for i in range(0,len(vectors)):
                    f.write(str(vectors[i]).replace("'", "").replace(", ", " ").replace("[", "").replace("]", "")+'\n')

            count += 1


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
