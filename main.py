import torch
from utilities import utilities, plotgraphs
from word2vec.trainer import Word2VecTrainer
from lstm.trainer import LSTMTrainer

from enum import IntEnum

class Mode(IntEnum):
    word2vec   = 0
    conversion = 1
    lstm       = 2
    plot       = 3

if __name__ == '__main__':

    # set mode of operation
    mode = Mode.plot

    # file locations
    label_file = './data/labels/labels.txt'
    dict_file  = './data/dictionary/dict.vec'
    doc_files  = utilities.generateFilePaths('./data/documents/test_', 3, '.txt')
    vec_files  = utilities.generateFilePaths('./data/vectors/vec_',    3, '.vec')

    # directories
    csv_acc_dir   = './data/performance/csv_accuracies/'
    csv_lss_dir   = './data/performance/csv_losses/'
    graph_acc_dir = './data/performance/graph_accuracies/'
    graph_lss_dir = './data/performance/graph_losses/'
    merged_dir    = './data/performance/graphs_merged/'


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

        #max_doc_len = utilities.getMaxDocumentLength(dict_file)

        # lstm training parameters
        lstm = LSTMTrainer(vec_files,
                           label_file,
                           learning_rate=0.001,
                           iterations_per_epoch=100,
                           input_dim=10,
                           seq_dim=6,
                           hidden_dim=30,
                           layer_dim=1,
                           output_dim=3)

        # train lstm and write results to csv
        parcel = lstm.train(num_epochs=100, compute_accuracies=True)
        utilities.resultsToCSV(parcel, lstm.to_string, csv_lss_dir, csv_acc_dir)


    if mode == Mode.plot:

        lss_y_range = [ 0.0, 2.0]
        acc_y_range = [-0.1, 1.1]

        plotgraphs.convertCsvToGraphs(csv_lss_dir, graph_lss_dir, lss_y_range, 'Cross Entropy Loss')
        plotgraphs.convertCsvToGraphs(csv_acc_dir, graph_acc_dir, acc_y_range, 'Accuracy in Percent')
