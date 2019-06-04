import torch
from utilities import utilities
from word2vec.trainer import Word2VecTrainer
from lstm.file2VecConverter import File2VecConverter
from lstm.dataReaderVec import VectorDataset
from lstm.lstm import LSTM

from enum import IntEnum

class Mode(IntEnum):
    word2vec   = 0
    conversion = 1
    lstm       = 2

# convert documents into vector representations and write them to files
def documentVectorisation(doc_files, vec_files, dict_file, debug=False):

    converter = File2VecConverter(doc_files, dict_file)
    converter.convertDocuments(vec_files)

    if debug:
        dataReader = VectorDataset(vec_files)
        reverse_dict = converter.readVectorsDict(reverse=True)

        for vector_doc in dataReader:
            utilities.printVecToWords(reverse_dict, vector_doc)
            exit(0)


if __name__ == '__main__':

    # set mode of operation
    mode = Mode.lstm

    # file locations
    dict_file = './data/dictionary/dict.vec'
    doc_files = utilities.generateFilePaths('./data/documents/test_', 3, '.txt')
    vec_files = utilities.generateFilePaths('./data/vectors/vec_',    3, '.vec')


    if mode == Mode.word2vec:

        # word2vec training parameters
        w2v = Word2VecTrainer(input_files=doc_files,
                              output_file=dict_file,
                              emb_dimension=10,
                              batch_size=32,
                              window_size=5,
                              iterations=300,
                              initial_lr=0.001,
                              min_count=1)

        # train word2vec
        w2v.train()


    if mode == Mode.conversion:

        # convert documents into vector representation and save to different file location
        documentVectorisation(doc_files, vec_files, dict_file, debug=False)


    if mode == Mode.lstm:

        dataReader = VectorDataset(vec_files)
        model = LSTM(308, 32)

        for vector_doc in dataReader:
            print(vector_doc.shape)
            hs, _ = model(vector_doc)
            exit(0)

