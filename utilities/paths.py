from utilities.utilities import generateFilePaths

class Paths():
    def __init__(self):

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

        # word2vec training results
        self.w2v_csv_lss_dir   = './data/w2v/performance/csv_losses/'
        self.w2v_graph_lss_dir = './data/w2v/performance/graph_losses/'
        self.w2v_merged_dir    = './data/w2v/performance/graphs_merged/'

        # lstm model weights
        self.lstm_model_param = './data/lstm/training/models/'

        # lstm training data
        self.label_file = './data/lstm/training/labels/labels.txt'
        self.vec_files  = generateFilePaths('./data/lstm/training/vectors/vec_', 3, '.vec')

        # lstm training results
        self.lstm_csv_acc_dir   = './data/lstm/performance/csv_accuracies/'
        self.lstm_csv_lss_dir   = './data/lstm/performance/csv_losses/'
        self.lstm_graph_acc_dir = './data/lstm/performance/graph_accuracies/'
        self.lstm_graph_lss_dir = './data/lstm/performance/graph_losses/'
        self.lstm_merged_dir    = './data/lstm/performance/graphs_merged/'

        # similarity
        self.sim_csv_dir   = './data/w2v/similarity/csv/'
        self.sim_graph_dir = './data/w2v/similarity/graphs/'

