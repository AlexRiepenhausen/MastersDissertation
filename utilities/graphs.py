from utilities import utilities
import matplotlib.pyplot as plt
import numpy as np


# populate graph with information in accuracy data
def populateGraph(accuracy_data):
    num_elements = len(accuracy_data[0])
    plt.figure(num='None',figsize=(10.5,5))
    line = plt.plot(np.array(accuracy_data[0]), np.array(accuracy_data[1]))
    plt.grid(True)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy in Percent')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, num_elements-1])
    plt.locator_params(axis='y', nbins=20,tight=None)
    plt.minorticks_on()
    plt.grid(linewidth='1')


# take csv files with accuracy data, convert to graphical representation and save to specified directory
def convertCsvToGraphs(accuracy_dir, graphs_dir):

    all_csv_files   = utilities.getFilesInDirectory(accuracy_dir)
    existing_graphs = utilities.getFilesInDirectory(graphs_dir)
    num_csv_files   = len(all_csv_files)

    remaining_targets = list()

    # check if csv had already been converted in the past
    for i in range(0, num_csv_files):
        csv_file = all_csv_files[i]
        if csv_file.replace('csv','jpg') not in existing_graphs:
            remaining_targets.append(csv_file.replace('.csv',''))

    # convert targets, i.e. those file that have not been converted yet
    for target in remaining_targets:
        csv_file   = accuracy_dir + target + '.csv'
        graph_file = graphs_dir   + target + '.jpg'

        accuracy_data = utilities.readAccuraciesFromCSV(csv_file)
        populateGraph(accuracy_data)
        plt.savefig(graph_file)
        plt.clf()

        print("Created graph from csv file {}".format(target))

    if not remaining_targets:
        print("No need to create new graphs. Stop.")


# merge all graphs in directory
def mergeAllGraphs(accuracy_dir, merge_dir):

    all_csv_files = utilities.getFilesInDirectory(accuracy_dir)
    num_csv_files = len(all_csv_files)

    for i in range(0, num_csv_files):
        accuracy_data = utilities.readAccuraciesFromCSV(accuracy_dir + all_csv_files[i])
        populateGraph(accuracy_data)

    merge_file = merge_dir + utilities.timeStampedFileName() + '.jpg'
    plt.savefig(merge_file)
    print("Created new graph '{}' from {} csv files".format(merge_file, num_csv_files))