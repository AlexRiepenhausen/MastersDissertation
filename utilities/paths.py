from utilities.utilities import generateFilePaths, readSpecifiedNumberOfFiles, copyFileNamesToDifferentPath, getFilesInDirectory
from utilities.utilities import getLabelsFromFiles
import ndjson

class RosDataPaths():

    def __init__(self,num_train,num_test):

        # word2vec training raw documents
        self.docpath           = './data/w2v/training/documents/'
        self.docfile_flats     = './data/w2v/training/documents/flatted_examples.ndjson'
        self.docfile_house     = './data/w2v/training/documents/examples.ndjson'
        self.colours           = './data/w2v/training/documents/colours.txt'

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
        data = None
        with open(self.docfile_house) as f:
            data = ndjson.load(f)   
        
        self.training = data[0:num_train]
        self.testing  = data[num_train:(num_train+num_test)]
        
        self.vec_files_train_path = './data/lstm/training/vectors/trainset/train_'
        self.vec_files_test_path  = './data/lstm/training/vectors/testset/test_'
        
        self.vec_files_train_labels_path = './data/lstm/training/vectors/trainsetlabels/labels_'
        self.vec_files_test_labels_path  = './data/lstm/training/vectors/testsetlabels/labels_'
        
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
        self.lstm_merged_dir    = './data/lstm/performance/graphs_merged/'
        
        # similarity
        self.sim_csv_dir = './data/w2v/similarity/csv/'
        self.sim_img_dir = './data/w2v/similarity/img/'  
        
        # lstm model weights
        self.lstm_model_param = './data/lstm/training/models/'
        
        #keywords
        self.keyword_path = './data/w2v/training/dictionary/keywords.txt'
        
        


