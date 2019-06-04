import re
from string import punctuation

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