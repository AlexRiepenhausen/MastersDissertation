from file2VecConverter import File2VecConverter
from dataReaderVec import VectorDataset
from utilities import utilities

if __name__ == '__main__':

    # convert documents into vector representations
    files = utilities.generateFilePaths('../data/documents/test_', 3, '.txt')
    vectors = '../data/dictionary/dict.vec'
    converter = File2VecConverter(files, vectors)
    #converter.convertDocuments()

    reverse_dict = converter.readVectorsDict(reverse=True)

    #https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/lstm_from_scratch.ipynb
    files = utilities.generateFilePaths('../data/vectors/vec_', 3,'.vec')
    dataReader = VectorDataset(files)

    for i in dataReader:
        utilities.printVecToWords(reverse_dict,i)
        exit(0)

