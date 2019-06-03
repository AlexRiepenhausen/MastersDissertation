import re
from string import punctuation

def parseLine(line):
    line = line.lower()
    line = re.sub(r'\(.*?\)', '', line)
    line = ''.join([c for c in line if c not in punctuation])
    return line

def generateFilePaths(path, number_of_docs, extension):
    paths = list()
    for i in range(0, number_of_docs):
        paths.append(path + str(i) + extension)
    return paths

