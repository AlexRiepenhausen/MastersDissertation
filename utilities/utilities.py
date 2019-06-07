import re
import csv
import datetime
import numpy as np
import os
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


# saves accuracy measures obtained during training to csv file
def writeDataToCSV(data, filename):
    x_axis = np.arange(len(data))
    with open(filename, mode='w', newline='') as csv_file:
        accuracy_writer = csv.writer(csv_file, delimiter=',')
        for i in range(0, len(data)):
            accuracy_writer.writerow([x_axis[i],data[i]])


# reads accuracy value from csv; needed for plotting a graph with matplotlib
def readAccuraciesFromCSV(filename):
    x_axis = list()
    y_axis = list()
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            x_axis.append(int(line[0]))
            y_axis.append(float(line[1]))

    return x_axis, y_axis


# generates time stamp
def timeStampedFileName():
    fmt = '%Y_%m_%d_%H_%M_%S'
    return datetime.datetime.now().strftime(fmt)


# returns names of all files in given directory
def getFilesInDirectory(directory):
    return os.listdir(directory)


# write accuracies and losses
def resultsToCSV(parcel, lstm_info, csv_losses_dir, csv_accuracies_dir):

    timestamp = timeStampedFileName()

    losses = parcel[0]
    lss_csv_file = csv_losses_dir + 'lss_' + lstm_info + '_date_' + timestamp + '.csv'
    writeDataToCSV(losses, lss_csv_file)

    if len(parcel) == 2:
        accuracies   = parcel[1]
        acc_csv_file = csv_accuracies_dir + 'acc_' + lstm_info + '_date_' + timestamp + '.csv'
        writeDataToCSV(accuracies, acc_csv_file)
