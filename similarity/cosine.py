import numpy as np
from numpy import dot
from numpy.linalg import norm
from utilities import utilities

class CosineSimilarity():

    def __init__(self, doc_path, dict_path, key_path):

        self.doc_path  = doc_path
        self.dict_path = dict_path
        self.key_path  = key_path

        self.all_dictionary, _ = utilities.readVectorsDict(dict_path)
        self.subset_dictionary = self.subsetDictFromDocs()
        self.key_dictionary    = self.getKeyDictionary()
        self.key_similarities  = self.getCosineSimilarities()

        #print(self.key_dictionary.keys())


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


    # cosine similarity
    def getCosineSimilarities(self):

        key_similarities = dict()

        # compare key against all other keys in the key dictionary
        for key0 in self.key_dictionary:
            similarities = list()
            for key1 in self.key_dictionary:
                if key0 == key1:
                    similarities.append((key1, 1.0))
                else:
                    vec0, vec1 = self.itemToVec(key0, key1)
                    cos_sim = dot(vec0, vec1) / (norm(vec0) * norm(vec1))
                    similarities.append((key1, cos_sim))
            key_similarities[key0] = similarities

        return key_similarities


    # this parsing thing is a nightmare
    def itemToVec(self, key0, key1):
        item0 = self.key_dictionary[key0]
        item1 = self.key_dictionary[key1]
        vec0  = np.array(item0).astype(np.float)
        vec1  = np.array(item1).astype(np.float)
        return vec0, vec1


