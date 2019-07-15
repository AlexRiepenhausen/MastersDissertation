from utilities.utilities import generateFilePaths, readSpecifiedNumberOfFiles, copyFileNamesToDifferentPath, getFilesInDirectory
from utilities.utilities import getLabelsFromFiles
import ndjson

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

        self.vec_lbls_test  = getLabelsFromFiles(self.vec_files_test, '.vec')


        # lstm training results
        self.lstm_csv_acc_dir   = './data/lstm/performance/csv_accuracies/'
        self.lstm_csv_lss_dir   = './data/lstm/performance/csv_losses/'
        self.lstm_graph_acc_dir = './data/lstm/performance/graph_accuracies/'
        self.lstm_graph_lss_dir = './data/lstm/performance/graph_losses/'
        self.lstm_merged_dir    = './data/lstm/performance/graphs_merged/'
        self.confusion_matrix   = './data/lstm/performance/confusion_matrix/'

        # similarity
        self.sim_csv_dir = './data/w2v/similarity/csv/'
        self.sim_img_dir = './data/w2v/similarity/img/'


class RosDataPaths():

    def __init__(self,num_train,num_test):

        # word2vec training raw documents
        self.docpath = './data/w2v/training/documents/'
        self.docfile_flats = './data/w2v/training/documents/flatted_examples.ndjson'
        self.docfile_house = './data/w2v/training/documents/examples.ndjson'
        self.colours = './data/w2v/training/documents/colours.txt'

        # word2vec training data dictionaries
        self.dict_file = './data/w2v/training/dictionary/dict.vec'

        # word2vec model weights
        self.w2v_model_param = './data/w2v/training/models/'
        
        # word2vec training results
        self.w2v_csv_lss_dir   = './data/w2v/performance/csv_losses/'
        self.w2v_graph_lss_dir = './data/w2v/performance/graph_losses/'
        self.w2v_merged_dir    = './data/w2v/performance/graphs_merged/'
        
        # word2vec model weights
        self.w2v_model_param = './data/w2v/training/models/' 
        
        # lstm training data
        house = None  
        flat  = None
        with open(self.docfile_house) as f:
            house = ndjson.load(f)   
        with open(self.docfile_flats) as f:
            flat  = ndjson.load(f)   
                      
        self.training = house[0:num_train] + flat[0:num_train]
        
        self.testing  = house[num_train:(num_train+num_test)] + flat[num_train:(num_train+num_test)]
        
        self.vec_files_train = generateFilePaths('./data/lstm/training/vectors/trainset/train_', num_train, '.txt')
        self.vec_files_test  = generateFilePaths('./data/lstm/training/vectors/testset/test_', num_test, '.txt')
        
        self.vec_files_train_labels = generateFilePaths('./data/lstm/training/vectors/trainsetlabels/labels_', num_train, '.txt')
        self.vec_files_test_labels  = generateFilePaths('./data/lstm/training/vectors/testsetlabels/labels_', num_test, '.txt')
        
        
        # lstm training results
        self.lstm_csv_acc_dir   = './data/lstm/performance/csv_accuracies/'
        self.lstm_csv_lss_dir   = './data/lstm/performance/csv_losses/'
        self.lstm_graph_acc_dir = './data/lstm/performance/graph_accuracies/'
        self.lstm_graph_lss_dir = './data/lstm/performance/graph_losses/'  
        self.confusion_matrix   = './data/lstm/performance/confusion_matrix/'  
        
        # similarity
        self.sim_csv_dir = './data/w2v/similarity/csv/'
        self.sim_img_dir = './data/w2v/similarity/img/'  
        
        # lstm model weights
        self.lstm_model_param = './data/lstm/training/models/'
        
        #keywords
        self.keyword_path = './data/w2v/training/dictionary/keywords.txt'
        
        # similarity
        self.sim_csv_dir = './data/w2v/similarity/csv/'
        self.sim_img_dir = './data/w2v/similarity/img/'
        
        


