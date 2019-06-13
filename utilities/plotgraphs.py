from utilities import utilities
import matplotlib.pyplot as plt
import numpy as np
import re

# populate graph with information in accuracy data
def populateGraph(data, y_axis_range, y_axis_name, iter_per_epoch=1):
    num_elements = len(data[0])
    plt.figure(num='None',figsize=(10.5,5))
    plt.plot(np.array(data[0]), np.array(data[1]))
    plt.grid(True)
    plt.xlabel('Number of Epochs (epoch={} iterations)'.format(iter_per_epoch))
    plt.ylabel(y_axis_name)
    plt.ylim(y_axis_range)
    plt.xlim([0.0, num_elements])
    plt.locator_params(axis='y', nbins=22,tight=None)
    plt.minorticks_on()
    plt.grid(linewidth='1')


# take csv files with accuracy data, convert to graphical representation and save to specified directory
def convertCsvToGraphs(csv_dir, graphs_dir, y_axis_range, y_axis_name):

    all_csv_files   = utilities.getFilesInDirectory(csv_dir)
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
        csv_file   = csv_dir    + target + '.csv'
        graph_file = graphs_dir + target + '.jpg'
        accuracy_data = utilities.readAccuraciesFromCSV(csv_file)
        populateGraph(accuracy_data, y_axis_range, y_axis_name, int(iterPerEpoch(target)))
        plt.savefig(graph_file)
        plt.clf()

        print("Created graph from csv file {}".format(target))

    if not remaining_targets:
        print("No need to create new graphs. Stop.")


# merge all graphs in directory
def mergeAllGraphs(accuracy_dir, merge_dir, y_axis_range, y_axis_name):

    all_csv_files = utilities.getFilesInDirectory(accuracy_dir)
    num_csv_files = len(all_csv_files)

    for i in range(0, num_csv_files):
        accuracy_data = utilities.readAccuraciesFromCSV(accuracy_dir + all_csv_files[i])
        populateGraph(accuracy_data, y_axis_range, y_axis_name)

    merge_file = merge_dir + utilities.timeStampedFileName() + '.jpg'
    plt.savefig(merge_file)
    print("Created new graph '{}' from {} csv files".format(merge_file, num_csv_files))


# find _ipe_ and get the number after (ipe = iterations per epoch)
def iterPerEpoch(string):
    substr = re.search('_ipe_(.+?)_', string).group(0)
    return substr.replace("_ipe_","").replace("_","")