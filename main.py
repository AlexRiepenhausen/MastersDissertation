import torch
from utilities import utilities, plotgraphs
from word2vec.trainer import Word2VecTrainer
from word2vec.dictionaryMerger import DictionaryMerger
from lstm.trainer import LSTMTrainer

from enum import IntEnum

class Mode(IntEnum):
    word2vec   = 0
    conversion = 1
    lstm       = 2
    plot       = 3


class VecTrain(IntEnum):
    combined  = 0
    primary   = 1
    secondary = 2

if __name__ == '__main__':

    # set mode of operation
    mode = Mode.plot

    # lstm training data
    label_file = './data/lstm/training/labels/labels.txt'
    vec_files  = utilities.generateFilePaths('./data/lstm/training/vectors/vec_', 3, '.vec')

    # lstm training results
    lstm_csv_acc_dir   = './data/lstm/performance/csv_accuracies/'
    lstm_csv_lss_dir   = './data/lstm/performance/csv_losses/'
    lstm_graph_acc_dir = './data/lstm/performance/graph_accuracies/'
    lstm_graph_lss_dir = './data/lstm/performance/graph_losses/'
    lstm_merged_dir    = './data/lstm/performance/graphs_merged/'

    # word2vec training data dictionaries
    docs_dict    = './data/w2v/training/dictionary/docs_dict.vec'
    colour_dict  = './data/w2v/training/dictionary/colour_dict.vec'
    repl_file    = './data/w2v/training/dictionary/colour_table.txt'
    dict_file    = './data/w2v/training/dictionary/dict.vec'

    # word2vec training raw documents
    doc_files    = utilities.generateFilePaths('./data/w2v/training/documents/test_',  3, '.txt')
    colour_files = utilities.generateFilePaths('./data/w2v/training/colours/colours_', 2, '.txt')

    # word2vec training results
    w2v_csv_lss_dir   = './data/w2v/performance/csv_losses/'
    w2v_graph_lss_dir = './data/w2v/performance/graph_losses/'
    w2v_merged_dir    = './data/w2v/performance/graphs_merged/'


    if mode == Mode.word2vec:

        # word2vec training parameters
        w2v0 = Word2VecTrainer(primary_files=doc_files,
                               emb_dimension=10,
                               batch_size=32,
                               window_size=5,
                               initial_lr=0.1,
                               min_count=1,
                               supporting_files=colour_files)

        w2v1 = Word2VecTrainer(primary_files=doc_files,
                               emb_dimension=10,
                               batch_size=32,
                               window_size=5,
                               initial_lr=0.1,
                               min_count=1,
                               supporting_files=colour_files)

        # train standard word2vec
        parcel_0 = w2v0.train(VecTrain.primary, docs_dict, num_epochs=100)
        parcel_1 = w2v1.train(VecTrain.secondary, colour_dict, num_epochs=100)

        utilities.resultsToCSV(parcel_0, w2v0.toString(), w2v_csv_lss_dir)
        utilities.resultsToCSV(parcel_1, w2v1.toString(), w2v_csv_lss_dir)

        mergeDictionaries = DictionaryMerger(docs_dict, colour_dict, repl_file, dict_file)
        mergeDictionaries.replaceVectors()
        mergeDictionaries.writeVectors()


    if mode == Mode.conversion:

        # convert documents into vector representation and save to different file location
        utilities.documentVectorisation(doc_files, vec_files, dict_file, debug=False)


    if mode == Mode.lstm:

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
        utilities.resultsToCSV(parcel, lstm.to_string, lstm_csv_lss_dir, lstm_csv_acc_dir)


    if mode == Mode.plot:

        w2v_lss_y_range  = [ 0.0, 4.0]
        lstm_lss_y_range = [ 0.0, 2.0]
        lstm_acc_y_range = [-0.1, 1.1]

        plotgraphs.convertCsvToGraphs(w2v_csv_lss_dir,   w2v_graph_lss_dir,  w2v_lss_y_range, 'Log-sigmoid loss')
        plotgraphs.convertCsvToGraphs(lstm_csv_lss_dir, lstm_graph_lss_dir, lstm_lss_y_range, 'Cross-entropy loss')
        plotgraphs.convertCsvToGraphs(lstm_csv_acc_dir, lstm_graph_acc_dir, lstm_acc_y_range, 'Accuracy in percent')
