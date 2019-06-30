import torch
import time
from utilities.utilities import Mode, weightInit, Vec
from utilities import utilities, plotgraphs, paths, display
from word2vec.trainer import Word2VecTrainer
from lstm.trainer import LSTMTrainer
from similarity.cosine import CosineSimilarity


if __name__ == '__main__':

    start   = time.time()
    loading = start

    #init paths
    ros = paths.RosDataPaths()

    # set mode of operation
    mode       = Mode.display
    save_model = True
    confusion  = True

    
    if mode == Mode.display:
        display = display.Display(ros.docpath, ros.docfile, 3)
        display.run()
        exit(0)

    if mode == Mode.word2vec:

        # word2vec training parameters
        w2v = Word2VecTrainer(primary_files=p.all_files,
                              emb_dimension=300,
                              batch_size=32,
                              window_size=5,
                              initial_lr=0.01,
                              min_count=1)

        # train standard word2vec -> train function outputs dictionary at the end
        loading = time.time()
        parcel_0 = w2v.train(p.all_files, p.dict_file, num_epochs=300)

        # write training results (learning curve) to csv
        utilities.resultsToCSV(parcel_0, w2v.toString(), p.w2v_csv_lss_dir)

        # save model if specified
        if save_model:
            path = p.w2v_model_param + w2v.toString() + '_date_' + utilities.timeStampedFileName()
            torch.save(w2v.skip_gram_model.state_dict(), path)


    if mode == Mode.conversion:
        # convert documents into vector representation and save to different file location

        train_files = p.imdb_files_pos_train + p.imdb_files_neg_train
        test_files  = p.imdb_files_pos_test  + p.imdb_files_neg_test

        utilities.documentVectorisation(train_files, p.vec_files_train, p.dict_file, unknown_vec=Vec.skipVec)
        utilities.documentVectorisation(test_files, p.vec_files_test, p.dict_file, unknown_vec=Vec.skipVec)


    if mode == Mode.lstm:

        # lstm training parameters
        lstm = LSTMTrainer(p.vec_files_train,
                           p.vec_lbls_train,
                           p.vec_files_test,
                           p.vec_lbls_test,
                           learning_rate=0.002,
                           iterations_per_epoch=1000,
                           input_dim=100,
                           seq_dim=6,
                           hidden_dim=30,
                           layer_dim=1,
                           output_dim=10)

        # train lstm
        loading = time.time()
        parcel = lstm.train(num_epochs=200, compute_accuracies=True)
        #parcel = lstm.train(num_epochs=1, compute_accuracies=False, test_samples=100)

        # save model if specified
        if save_model:
            path = p.lstm_model_param + lstm.to_string + '_date_' + utilities.timeStampedFileName()
            torch.save(lstm.model.state_dict(), path)

        # write results to csv
        utilities.resultsToCSV(parcel, lstm.to_string, p.lstm_csv_lss_dir, p.lstm_csv_acc_dir)

        # write confusion matrix as image to output
        if confusion:

            # test set
            labels, accuracy = lstm.evaluateModel(test_samples=1000, test=True)
            print("Accuracy Test Set: {}".format(accuracy))
            class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            plotgraphs.plot_confusion_matrix(labels[0], labels[1], p.confusion_matrix, classes=class_names,
                                                title='Confusion matrix, without normalization')

            # training set
            labels, accuracy = lstm.evaluateModel(test_samples=1000, test=False)
            print("Accuracy Training Set: {}".format(accuracy))
            plotgraphs.plot_confusion_matrix(labels[0], labels[1], p.confusion_matrix, classes=class_names,
                                                title='Confusion matrix, without normalization')



    if mode == Mode.similarity:
        path = p.sim_img_dir + utilities.timeStampedFileName() + '.bmp'
        measure_similarity = CosineSimilarity(p.imdb_files_neg_train, p.dict_file)
        measure_similarity.angularDistancesToFile(path)


    if mode == Mode.plot:

        w2v_lss_y_range  = [ 0.0, 4.0]
        lstm_lss_y_range = [-0.2, 2.0]
        lstm_acc_y_range = [-0.1, 1.1]

        plotgraphs.convertCsvToGraphs(p.w2v_csv_lss_dir,   p.w2v_graph_lss_dir,  w2v_lss_y_range, 'Log-sigmoid loss')
        plotgraphs.convertCsvToGraphs(p.lstm_csv_lss_dir, p.lstm_graph_lss_dir, lstm_lss_y_range, 'Cross-entropy loss')
        plotgraphs.convertCsvToGraphs(p.lstm_csv_acc_dir, p.lstm_graph_acc_dir, lstm_acc_y_range, 'Accuracy in percent')



    end = time.time()

    print("Time needed for loading data {} seconds".format(round(loading - start)))
    print("Time needed for processing {} seconds".format(round(end - loading)))
    print("Total time {} seconds".format(round(end - start)))
