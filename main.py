import torch
from utilities import utilities, plotgraphs, paths
from utilities.utilities import Mode, weightInit
from word2vec.trainer import Word2VecTrainer
from word2vec.dictionaryMerger import DictionaryMerger
from lstm.trainer import LSTMTrainer
from similarity.cosine import CosineSimilarity

if __name__ == '__main__':

    #init paths
    p = paths.Paths()

    # set mode of operation
    mode       = Mode.conversion
    save_model = True

    # models to be loaded
    w2v_model  = p.w2v_model_param  + 'lr_0.1_bs_32_ipe_288_embs_459_embd_10_win_5_date_2019_06_13_15_06_14'
    lstm_model = p.lstm_model_param + 'lr_0.001_ipe_100_in_10_sq_6_hd_30_ly_1_out_3_date_2019_06_13_15_25_25'

    if mode == Mode.word2vec:

        # word2vec training parameters
        w2v = Word2VecTrainer(primary_files=p.doc_files,
                              emb_dimension=10,
                              batch_size=32,
                              window_size=5,
                              initial_lr=0.1,
                              min_count=1)

        # train standard word2vec -> train function outputs dictionary at the end
        parcel_0 = w2v.train(p.doc_files, p.docs_dict, num_epochs=100)

        # write training results (learning curve) to csv
        utilities.resultsToCSV(parcel_0, w2v.toString(), p.w2v_csv_lss_dir)

        # save model if specified
        if save_model:
            path = p.w2v_model_param + w2v.toString() + '_date_' + utilities.timeStampedFileName()
            torch.save(w2v.skip_gram_model.state_dict(), path)


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

        # train lstm
        parcel = lstm.train(num_epochs=100, compute_accuracies=False,
                            init=weightInit.load, model_path=lstm_model)

        # write results to csv
        utilities.resultsToCSV(parcel, lstm.to_string, p.lstm_csv_lss_dir, p.lstm_csv_acc_dir)

        # save model if specified
        if save_model:
            path = p.lstm_model_param + lstm.to_string + '_date_' + utilities.timeStampedFileName()
            torch.save(lstm.model.state_dict(), path)


    if mode == Mode.similarity:
        path = p.sim_img_dir + utilities.timeStampedFileName() + '.bmp'
        measure_similarity = CosineSimilarity(p.doc_files, p.dict_file, p.repl_file)
        measure_similarity.angularDistancesToFile(path)
        #p.sim_csv_dir


    if mode == Mode.plot:

        w2v_lss_y_range  = [ 0.0, 4.0]
        lstm_lss_y_range = [ 0.0, 2.0]
        lstm_acc_y_range = [-0.1, 1.1]

        plotgraphs.convertCsvToGraphs(p.w2v_csv_lss_dir,   p.w2v_graph_lss_dir,  w2v_lss_y_range, 'Log-sigmoid loss')
        plotgraphs.convertCsvToGraphs(p.lstm_csv_lss_dir, p.lstm_graph_lss_dir, lstm_lss_y_range, 'Cross-entropy loss')
        plotgraphs.convertCsvToGraphs(p.lstm_csv_acc_dir, p.lstm_graph_acc_dir, lstm_acc_y_range, 'Accuracy in percent')
