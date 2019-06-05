import re
from string import punctuation
from lstm.file2VecConverter import File2VecConverter
from lstm.dataReaderVec import VectorDataset

# cleans a line of text from punctuation and other special characters before processing
def parseLine(line):
    line = line.lower()
    line = re.sub(r'\(.*?\)', '', line)
    line = ''.join([c for c in line if c not in punctuation])
    return line


# returns a list of file paths from the same directory to avoid manual initialisation
def generateFilePaths(path, number_of_docs, extension):
    paths = list()
    for i in range(0, number_of_docs):
        paths.append(path + str(i) + extension)
    return paths


# convert documents into vector representations and write them to files
def documentVectorisation(doc_files, vec_files, dict_file, debug=False):
    converter = File2VecConverter(doc_files, dict_file)
    converter.convertDocuments(vec_files)

    if debug:
        dataReader = VectorDataset(vec_files)
        reverse_dict = converter.readVectorsDict(reverse=True)

        for vector_doc in dataReader:
            printVecToWords(reverse_dict, vector_doc)
            exit(0)


# reconverts vectors to words for debugging/checking purposes
def printVecToWords(reverse_dict,vectors):
    for vector in vectors:
        count = 0
        for element in vector:
            if len(str(element)) > 5:
                vector[count] = str(element)[0:5]
            count += 1
        key = str(vector).replace("\n", "").replace(" ", "").replace("0","")
        if key in reverse_dict:
            print(reverse_dict[key])
        else:
            print(key)


# returns the maximum number of words observed in a single document
def getMaxDocumentLength(dict_file):
    with open(dict_file,'r') as f:
        header = f.readline()
        return int(header.split()[2])