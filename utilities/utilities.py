import torch
import re
import sys
import csv
import time
import datetime
import numpy as np
import os
from string import punctuation
from lstm.file2VecConverter import File2VecConverter
from lstm.dataReaderVec import VectorDataset

from enum import IntEnum
                    
class labelType(IntEnum):
    property_type     = 0
    tenement_steading = 1 
    exclusive_strata  = 2                       
    exclusive_solum   = 3
    common_strata     = 4
    common_solum      = 5
    additional_info   = 6
    char_count        = 7              
    index             = 8
    identifier        = 9



class Mode(IntEnum):
    word2vec   = 0
    conversion = 1
    lstm       = 2
    similarity = 3
    plot       = 4
    display    = 5



class weightInit(IntEnum):
    fromScratch = 0
    load        = 1
    inherit     = 2



class Vec(IntEnum):
    zeroVec = 0
    skipVec = 1



# cleans a line of text from punctuation and other special characters before processing
def parseLine(line):
    line = line.lower()
    line = re.sub(r'\(.*?\)', '', line) #remove special characters
    line = ''.join([c for c in line if c not in punctuation])
    line = re.sub(r'[^\x00-\x7f]', r'', line) #remove hex characters
    return line



# returns a list of file paths from the same directory to avoid manual initialisation
def generateFilePaths(path, number_of_docs, extension):
    paths = list()
    for i in range(0, number_of_docs):      
        paths.append(path + '{0:05d}'.format(i) + extension)
    return paths
    
    

def getTextNdJson(data, index):

    address    = parseLine(data[index]['address'][0]['prettyPrint']) 
    text_array = parseLine(data[index]['text']).replace(address, 'address').split(' ') 

    result = list()
    
    for item in text_array:
        if len(item) > 0:
            result.append(item)
            
    return result  



def ndjsonVectorisation(data, vec_files, labels, dict_file, unknown_vec):

    num_unknown_words = 0
    total_num_words   = 0

    vector_dict, params = readVectorsDict(dict_file)

    for i in range(0, len(vec_files)):
        
        text = getTextNdJson(data, i)
            
        with open(vec_files[i], 'w') as f:
        
            # replace words with vectors
            for word in text:
            
                if word in vector_dict:
                    f.write(str(vector_dict[word]).replace("'", "").replace(", ", " ").replace("[", "").replace("]", "")+'\n')
                else:
                    num_unknown_words += 1
                    
                total_num_words += 1
                
        with open(labels[i], 'w') as f:
            f.write(data[i]['property_type'] + '\n')
            f.write(data[i]['exclusive_solum']+ '\n')                   
            f.write(data[i]['common_solum']+ '\n')  
            f.write(data[i]['additional_info']+ '\n')
        
            
    percent_unknown_words = num_unknown_words*100/total_num_words
    sys.stderr.flush()
    print("\nTotal number of words {}, unknown words {}, percentage unknown words {}".format(total_num_words,
                                                                                              num_unknown_words,
                                                                                              percent_unknown_words))            



# convert documents into vector representations and write them to files
def documentVectorisation(doc_files, vec_files, dict_file, unknown_vec, debug=False):

    converter = File2VecConverter(doc_files, dict_file, unknown_vec)
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
    time.sleep(1.1) # -> make sure no equivalent time stamp is given out
    return datetime.datetime.now().strftime(fmt)



# returns names of all files in given directory
def getFilesInDirectory(directory):
    return os.listdir(directory)



# write lstm or word2vec accuracies and losses to csv
def resultsToCSV(parcel, lstm_info, csv_losses_dir, csv_accuracies_dir=None):

    timestamp = timeStampedFileName()

    losses = parcel[0] if csv_accuracies_dir else parcel
    lss_csv_file = csv_losses_dir + 'lss_' + lstm_info + '_date_' + timestamp + '.csv'
    writeDataToCSV(losses, lss_csv_file)

    if csv_accuracies_dir:

        if len(parcel) == 1:
            return

        accuracies       = parcel[1]
 
        acc_csv_file_neg = csv_accuracies_dir + 'acc_neg' + lstm_info + '_date_' + timestamp + '.csv'        
        acc_csv_file_pos = csv_accuracies_dir + 'acc_pos' + lstm_info + '_date_' + timestamp + '.csv'
        
        #get positive accuracies
        acc_neg = list()
        acc_pos = list()
        
        for item in accuracies:
            acc_neg.append(item["negative"])
            acc_pos.append(item["positive"])
        
        writeDataToCSV(acc_neg, acc_csv_file_neg)
        writeDataToCSV(acc_pos, acc_csv_file_pos)
        
        

# read file containing mapping of words to (pretrained) vectors
def readVectorsDict(dict_file_path, reverse=False):

    lines = []
    num_lines = 0
    for line in open(dict_file_path, encoding="utf8"):
        lines.append(line)
        num_lines = num_lines + 1

    num_vectors = np.int_(lines[0].split()[0])
    vector_size = np.int_(lines[0].split()[1])
    num_vec_req = np.int_(lines[0].split()[2])

    vector_dict = dict()
    for i in range(1,num_lines):
        vector = lines[i].split()
        if not reverse:
            cross_hatched = vector[0] + ' ' + vector[1]
            if cross_hatched == 'cross hatched':
                vector_dict['cross hatched'] = vector[2:]
            else:
                vector_dict[vector[0]] = vector[1:]
        if reverse:
            key = getReverseDictKey(vector)
            vector_dict[key] = vector[0]

    return vector_dict, (num_vectors, vector_size, num_vec_req)



# returns the vector in form of a parsed string, which is then used as the reverse ditionary key
def getReverseDictKey(vector):

    count = 0
    for element in vector:
        if len(element) > 5 and count > 0:
            vector[count] = element[0:5]
        count += 1
    key = str(vector[1:]).replace("'", "").replace(",", "").replace(" ", "").replace("0", "")

    return key



# read the words to be replaced in the primary vector file
def readKeyTable(replacement_table_file_path):
    table = list()
    for line in open(replacement_table_file_path, encoding="utf8"):
        table.append(line.replace("\n",""))
    return table



# read the first x files in a directory
def readSpecifiedNumberOfFiles(numFiles,path):

    # numFiles has to be an integer value
    numFiles = int(numFiles)

    files  = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    labels = list()

    for i in range(0,numFiles):
        labels.append(int(files[i].split('_',1)[1].replace('.txt','')))
        files[i] = path + files[i]

    return labels[0:numFiles], files[0:numFiles]



def getLabelsFromFiles(files, extension):

    labels = list()

    for f in files:
        chunks = f.split('/')
        label  = chunks[len(chunks)-1].split('.')[0].split('_')[1]
        labels.append(int(label))

    return labels



def copyFileNamesToDifferentPath(path, filenames, extension):

    new_file_names = list()
    for file in filenames:
        chunks = file.split('/')
        file_name = path + chunks[len(chunks)-1].split('.')[0] + extension
        new_file_names.append(file_name)
        
    return new_file_names



# save labels of every document so that they are readily accessible
def initLabels(self, label_file):

    labels = list()
    file = open(label_file, 'r', encoding='utf8')
    for line in file:
        item = line.replace("\n","")
        labels.append(int(item))

    return np.asarray(labels)


