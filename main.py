import torch
from utilities import utilities, graphs
from word2vec.trainer import Word2VecTrainer
from lstm.trainer import LSTMTrainer

from enum import IntEnum

class Mode(IntEnum):
    word2vec   = 0
    conversion = 1
    lstm       = 2
    accuracy   = 3

if __name__ == '__main__':

    # set mode of operation
    mode = Mode.accuracy

    # file locations
    label_file   = './data/labels/labels.txt'
    dict_file    = './data/dictionary/dict.vec'
    doc_files    = utilities.generateFilePaths('./data/documents/test_', 3, '.txt')
    vec_files    = utilities.generateFilePaths('./data/vectors/vec_',    3, '.vec')

    # directories
    accuracy_dir = './data/accuracies/csv/'
    graphs_dir   = './data/accuracies/graphs/'
    merged_dir   = './data/accuracies/graphs_merged/'


    if mode == Mode.word2vec:

        # word2vec training parameters
        w2v = Word2VecTrainer(input_files=doc_files,
                              output_file=dict_file,
                              emb_dimension=10,
                              batch_size=32,
                              window_size=5,
                              iterations=50000,
                              initial_lr=0.1,
                              min_count=1)

        # train word2vec
        w2v.train()


    if mode == Mode.conversion:

        # convert documents into vector representation and save to different file location
        utilities.documentVectorisation(doc_files, vec_files, dict_file, debug=False)


    if mode == Mode.lstm:

        max_doc_len = utilities.getMaxDocumentLength(dict_file)

        # lstm training parameters
        lstm = LSTMTrainer(vec_files,
                           label_file,
                           learning_rate=0.001,
                           input_dim=max_doc_len,
                           hidden_dim=10,
                           layer_dim=1,
                           output_dim=3)

        # train lstm
        accuracies = lstm.train(num_epochs=100, seq_dim=10)

        # write accuracies
        csv_file = accuracy_dir + lstm.to_string + '_date_' + utilities.timeStampedFileName() + '.csv'
        utilities.writeAccuraciesToCSV(accuracies, csv_file)


    if mode == Mode.accuracy:
        #graphs.convertCsvToGraphs(accuracy_dir, graphs_dir)
        graphs.mergeAllGraphs(accuracy_dir, merged_dir)