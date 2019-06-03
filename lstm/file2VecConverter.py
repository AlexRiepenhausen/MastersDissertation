import torch
import numpy as np
from utilities import utilities

class File2VecConverter:

    def __init__(self, doc_file_paths, vec_file_path):

        self.doc_file_paths = doc_file_paths
        self.vec_file_path = vec_file_path

        self.num_vectors = 0
        self.vector_size = 0

        self.vector_dict = self.readVectorsDict()

    def readVectorsDict(self):

        lines = []
        for line in open(self.vec_file_path, encoding="utf8"):
            lines.append(line)

        self.num_vectors = np.int_(lines[0].split()[0])
        self.vector_size = np.int_(lines[0].split()[1])

        vector_dict = dict()
        for i in range(1,self.num_vectors+1):
            vector = lines[i].split()
            vector_dict[vector[0]] = vector[1:]

        return vector_dict

    def convertDocuments(self):

        count = 0
        for file in self.doc_file_paths:
            file_name = '../data/vectors/vec_' + str(count) + '.vec'
            vec_count, vectors = self.vecToLine(file)

            with open(file_name, 'w') as f:
                for i in range(0,len(vectors)):
                    f.write(str(vectors[i]) + '\n')

            count += 1

    def vecToLine(self,file):

        vectors = list()
        line_count = 0
        for line in open(file, encoding="utf8"):
            line = utilities.parseLine(line).split()
            if len(line) > 1:
                for word in line:
                    if len(word) > 0:
                        vectors.append(self.vector_dict[word])
                        line_count += 1

        return line_count, vectors
