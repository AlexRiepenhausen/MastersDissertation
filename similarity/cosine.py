import numpy as np
import math
from numpy import dot
from numpy.linalg import norm
from utilities import utilities
import scipy.misc as smp
from tqdm import tqdm
import time

class CosineSimilarity():

    def __init__(self, dict_path, key_path):

        self.dictionary, _ = utilities.readVectorsDict(dict_path)
        self.keys          = self.getKeys(key_path)
         
        
        
    # generate all relevant keys
    def getKeys(self, key_path):
        keys = utilities.readKeyTable(key_path)
        for key in keys:
            if key not in self.dictionary:
                keys.remove(key)
        keys.remove('amb')
        return keys
        
        

    # cosine similarity -> angular distance
    def getDistance(self, row, col):
        if row == col:
            return 255
                        
        vec0, vec1 = self.itemToVec(row, col)
        cos_sim = abs(dot(vec0, vec1)) / (norm(vec0) * norm(vec1))

        return cos_sim



    # this parsing thing is a nightmare
    def itemToVec(self, key0, key1):
        item0 = self.dictionary[key0]
        item1 = self.dictionary[key1]
        vec0  = np.array(item0).astype(np.float)
        vec1  = np.array(item1).astype(np.float)
        return vec0, vec1



    # writing the dictionary/matrix to file
    def angularDistancesToFile(self, path):

        matrix_header = self.getMatrixHeader()

        width  = len(matrix_header) + 1
        height = len(matrix_header) + 1

        image = np.zeros((height+1, width+1), dtype=np.uint8)

        self.defineImageBorders(image, height, width, matrix_header)

        for i in tqdm(range(1, height)):
            for j in range(1, width):
                row = matrix_header[i-1]
                col = matrix_header[j-1]
                image[i][j] = self.getDistance(row, col)*255

        img = smp.toimage(image)
        smp.imsave(path, img)



    def defineImageBorders(self,image, height, width, matrix_header):

        image[0][0] = 210
        image[height][width] = 210

        for i in range(0, height):
            image[i][height] = 210
            image[height][i] = 210

        for i in range(1, height):
            if matrix_header[i-1] in self.keys:
                image[i][0] = 10
                image[0][i] = 10
            else:
                image[i][0] = 210
                image[0][i] = 210



    def getMatrixHeader(self):

        matrix_header = list()
        key_len       = len(self.keys)        
        dict_len      = len(self.dictionary)

        count = 0

        for key in self.dictionary:
        
            if key not in self.keys:
                matrix_header.append(key)
            count += 1

            # put the keys (e.g. the colours that we trained on a different corpus) in the middle
            if count == int(dict_len/2):
                matrix_header = matrix_header + self.keys

        return matrix_header





