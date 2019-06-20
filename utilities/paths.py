from utilities.utilities import generateFilePaths, readSpecifiedNumberOfFiles, copyFileNamesToDifferentPath
from utilities.utilities import getLabelsFromFiles

class Paths():
    def __init__(self, training_samples, test_samples):

        # word2vec training data dictionaries
        self.docs_dict   = './data/w2v/training/dictionary/docs_dict.vec'
        self.colour_dict = './data/w2v/training/dictionary/colour_dict.vec'
        self.repl_file   = './data/w2v/training/dictionary/colour_table.txt'
        self.dict_file   = './data/w2v/training/dictionary/dict.vec'

        # word2vec model weights
        self.w2v_model_param = './data/w2v/training/models/'

        # word2vec training raw documents
        self.doc_files    = generateFilePaths('./data/w2v/training/documents/test_', 3, '.txt')
        self.colour_files = generateFilePaths('./data/w2v/training/colours/colours_', 2, '.txt')

        # word2vec training imdb
        _, all_negatives_train = readSpecifiedNumberOfFiles(2500, './data/w2v/training/aclImdb/train/neg/')
        _, all_positives_train = readSpecifiedNumberOfFiles(2500, './data/w2v/training/aclImdb/train/pos/')
        self.all_files = all_negatives_train + all_positives_train

        # lstm training set
        self.imdb_lbl_neg_train, self.imdb_files_neg_train = readSpecifiedNumberOfFiles(training_samples/2, './data/w2v/training/aclImdb/train/neg/')
        self.imdb_lbl_pos_train, self.imdb_files_pos_train = readSpecifiedNumberOfFiles(training_samples/2, './data/w2v/training/aclImdb/train/pos/')
        self.imdb_lbl_neg_test,  self.imdb_files_neg_test  = readSpecifiedNumberOfFiles(test_samples/2, './data/w2v/training/aclImdb/test/neg/')
        self.imdb_lbl_pos_test,  self.imdb_files_pos_test  = readSpecifiedNumberOfFiles(test_samples/2, './data/w2v/training/aclImdb/test/pos/')

        # word2vec training results
        self.w2v_csv_lss_dir   = './data/w2v/performance/csv_losses/'
        self.w2v_graph_lss_dir = './data/w2v/performance/graph_losses/'
        self.w2v_merged_dir    = './data/w2v/performance/graphs_merged/'

        # lstm model weights
        self.lstm_model_param = './data/lstm/training/models/'

        # lstm training data
        self.vec_files_train = copyFileNamesToDifferentPath('./data/lstm/training/vectors_train/',
                                                             self.imdb_files_pos_train + self.imdb_files_neg_train,
                                                             '.vec')

        self.vec_lbls_train = getLabelsFromFiles(self.vec_files_train, '.vec')

        self.vec_files_test = copyFileNamesToDifferentPath('./data/lstm/training/vectors_test/',
                                                            self.imdb_files_pos_test + self.imdb_files_neg_test,
                                                            '.vec')

        self.vec_lbls_test = getLabelsFromFiles(self.vec_files_test, '.vec')


        # lstm training results
        self.lstm_csv_acc_dir   = './data/lstm/performance/csv_accuracies/'
        self.lstm_csv_lss_dir   = './data/lstm/performance/csv_losses/'
        self.lstm_graph_acc_dir = './data/lstm/performance/graph_accuracies/'
        self.lstm_graph_lss_dir = './data/lstm/performance/graph_losses/'
        self.lstm_merged_dir    = './data/lstm/performance/graphs_merged/'

        # similarity
        self.sim_csv_dir = './data/w2v/similarity/csv/'
        self.sim_img_dir = './data/w2v/similarity/img/'


