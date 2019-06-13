import torch
import numpy as np
from utilities.utilities import *

''' This class takes both primary and secondary dictionary files (e.g. colours and documents)
    and replaces specific vectors in the primary dictionary with their equivalents in the secondary dictionary.
    The vectors to be replaced have to be specified in a third file, e.g. colour_table
'''

class DictionaryMerger:

    def __init__(self, primary_dict_file_path,
                 secondary_dict_file_path,
                 replacement_table_file_path,
                 output_file_path):

        self.output_file_path = output_file_path

        self.dict1, params = readVectorsDict(primary_dict_file_path)
        self.dict2, params = readVectorsDict(secondary_dict_file_path)
        self.new_dict      = self.dict1

        self.unique_vectors = params[0]
        self.vector_size    = params[1]
        self.max_file_size  = params[2]

        self.replacement_table = readKeyTable(replacement_table_file_path)


    # create a new dictionary by replacing the vectors
    def replaceVectors(self):
        for item in self.replacement_table:
            if item in self.new_dict:
                self.new_dict[item] = self.dict2[item]


    # write new dictionary to file
    def writeVectors(self):
        with open(self.output_file_path, 'w') as f:
            f.write('%d %d %d\n' % (self.unique_vectors, self.vector_size, self.max_file_size))
            for item in self.new_dict:
                vec = ' '.join(map(lambda x: str(x), self.new_dict[item]))
                f.write('%s %s\n' % (item, vec))