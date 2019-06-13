import torch
from utilities import utilities, plotgraphs, paths
from word2vec.trainer import Word2VecTrainer
from word2vec.dictionaryMerger import DictionaryMerger
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

    #init paths
    p = paths.Paths()

    if mode == Mode.word2vec:

        # word2vec training parameters
        w2v = Word2VecTrainer(primary_files=p.doc_files,
                               emb_dimension=10,
                               batch_size=32,
                               window_size=5,
                               initial_lr=0.1,
                               min_count=1,
                               supporting_files=p.colour_files)

        # train standard word2vec
        parcel_0 = w2v.train(p.doc_files, p.docs_dict, num_epochs=100, retrain=True)
        parcel_1 = w2v.train(p.colour_files, p.colour_dict, num_epochs=100, retrain=True)

        utilities.resultsToCSV(parcel_0, w2v.toString(), p.w2v_csv_lss_dir)
        utilities.resultsToCSV(parcel_1, w2v.toString(), p.w2v_csv_lss_dir)

        mergeDictionaries = DictionaryMerger(p.docs_dict, p.colour_dict, p.repl_file, p.dict_file)
        mergeDictionaries.replaceVectors()
        mergeDictionaries.writeVectors()


    if mode == Mode.conversion:

        # convert documents into vector representation and save to different file location
        utilities.documentVectorisation(p.doc_files, p.vec_files, p.dict_file, debug=False)


    if mode == Mode.lstm:

        # lstm training parameters
        lstm = LSTMTrainer(p.vec_files,
                           p.label_file,
                           learning_rate=0.001,
                           iterations_per_epoch=100,
                           input_dim=10,
                           seq_dim=6,
                           hidden_dim=30,
                           layer_dim=1,
                           output_dim=3)

        # train lstm and write results to csv
        parcel = lstm.train(num_epochs=100, compute_accuracies=True)
        utilities.resultsToCSV(parcel, lstm.to_string, p.lstm_csv_lss_dir, p.lstm_csv_acc_dir)


    if mode == Mode.plot:

        w2v_lss_y_range  = [ 0.0, 4.0]
        lstm_lss_y_range = [ 0.0, 2.0]
        lstm_acc_y_range = [-0.1, 1.1]

        plotgraphs.convertCsvToGraphs(p.w2v_csv_lss_dir,   p.w2v_graph_lss_dir,  w2v_lss_y_range, 'Log-sigmoid loss')
        plotgraphs.convertCsvToGraphs(p.lstm_csv_lss_dir, p.lstm_graph_lss_dir, lstm_lss_y_range, 'Cross-entropy loss')
        plotgraphs.convertCsvToGraphs(p.lstm_csv_acc_dir, p.lstm_graph_acc_dir, lstm_acc_y_range, 'Accuracy in percent')
