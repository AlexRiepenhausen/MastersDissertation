import numpy as np
import math
from numpy import dot
from numpy.linalg import norm
from utilities import utilities
import scipy.misc as smp
from tqdm import tqdm

class CosineSimilarity():

    def __init__(self, doc_path, dict_path, key_path=None):

        self.doc_path  = doc_path
        self.dict_path = dict_path
        self.key_path  = None

        self.all_dictionary, _ = utilities.readVectorsDict(dict_path)
        self.subset_dictionary = self.subsetDictFromDocs()
        self.key_dictionary    = dict()

        if key_path:
            self.key_path       = key_path
            self.key_dictionary = self.getKeyDictionary()


        self.distance_matrix   = self.getDistances()


    # looks vector up in word2vec dictionary and writes single line to file
    def subsetDictFromDocs(self):

        vectors = dict()
        for file in self.doc_path:
            for line in open(file, encoding="utf8"):
                line = utilities.parseLine(line).split()
                if len(line) > 1:
                    for word in line:
                        if len(word) > 0 and word not in vectors:
                            vectors[word] = self.all_dictionary[word]

        return vectors


    # get those keys that exist within the main dictionary, e.g. all colours in the document dictionary
    def getKeyDictionary(self):

        key_dict = dict()
        all_keys = utilities.readKeyTable(self.key_path)

        for item in self.subset_dictionary:
            if item in all_keys:
                key_dict[item] = self.subset_dictionary[item]

        return key_dict


    # cosine similarity -> angular distance
    def getDistances(self):

        distance_matrix = dict()

        # compare item against all other items in the key dictionary
        for item0 in self.subset_dictionary:
            distances_row = dict()
            for item1 in self.subset_dictionary:
                if item0 == item1:
                    distances_row[item1] = 1.0
                else:
                    vec0, vec1 = self.itemToVec(item0, item1)
                    cos_sim = abs(dot(vec0, vec1)) / (norm(vec0) * norm(vec1))
                    dist = 1 - math.acos(cos_sim)/math.pi
                    distances_row[item1] = dist
            distance_matrix[item0] = distances_row

        return distance_matrix


    # this parsing thing is a nightmare
    def itemToVec(self, key0, key1):
        item0 = self.subset_dictionary[key0]
        item1 = self.subset_dictionary[key1]
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
                image[i][j] = int(self.distance_matrix[row][col]*255)

        img = smp.toimage(image)
        smp.imsave(path, img)


    def defineImageBorders(self,image, height, width, matrix_header):

        image[0][0] = 210
        image[height][width] = 210

        for i in range(0, height):
            image[i][height] = 210
            image[height][i] = 210

        for i in range(1, height):
            if matrix_header[i-1] in self.key_dictionary:
                image[i][0] = 10
                image[0][i] = 10
            else:
                image[i][0] = 210
                image[0][i] = 210

    def getMatrixHeader(self):

        matrix_header = list()
        subset_dict_len = len(self.subset_dictionary)
        key_dict_len    = len(list(self.key_dictionary.keys()))

        count = 0

        for key in self.subset_dictionary:
            if key not in self.key_dictionary:
                matrix_header.append(key)
            count += 1

            # put the keys (e.g. the colours that we trained on a different corpus) in the middle
            if count == int((subset_dict_len - key_dict_len)/2):
                matrix_header = matrix_header + list(self.key_dictionary.keys())

        return matrix_header





