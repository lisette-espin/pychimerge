__author__ = 'lisette.espin'

######################################################################################################################
# SYSTEM
######################################################################################################################
import sys
import numpy as np
import json

######################################################################################################################
# LOCAL DEPENDENCES
######################################################################################################################
from chimerge import ChiMerge
from chi2 import Chi2
import utils

######################################################################################################################
# FUNCTIONS IRIS DB
# http://archive.ics.uci.edu/ml/machine-learning-databases/iris/
######################################################################################################################
def example_chimerge_irisdb(attribute_column, min_expected_value, max_number_intervals, threshold):
    chi = ChiMerge(min_expected_value, max_number_intervals, threshold)
    data = _readIrisDataset(attribute_column)
    chi.loadData(data, False)
    chi.generateFrequencyMatrix()
    chi.chimerge()
    chi.printFinalSummary()

def example_chi2_irisdb(alpha, delta, min_expected_value):
    chi = Chi2(alpha, delta, min_expected_value)
    data = _readIrisDataset()
    chi.loadData(data)
    chi.printInitialSummary()
    chi.chi2()
    # chi.printFinalSummary()

def _readIrisDataset(attribute_column=-1):
    '''
    Reference: http://archive.ics.uci.edu/ml/machine-learning-databases/iris/
    e.g.: 5.1,3.5,1.4,0.2,Iris-setosa
        1. sepal length in cm   (index 0) a
        2. sepal width in cm    (index 1) a
        3. petal length in cm   (index 2) a
        4. petal width in cm    (index 3) a
        5. class:               (index 4) c
        -- Iris Setosa
        -- Iris Versicolour
        -- Iris Virginica
    :return:
    '''

    if attribute_column < -1 or attribute_column > 3:
        utils.printf('ERROR: index {} is not valid in this dataset!'.format(attribute_column))
        return
    if attribute_column == -1:
        attribute_columns = [0,1,2,3]
        utils.printf('INFO: You are about to load the complete dataset, including all attribute columns.')
    else:
        attribute_columns = [attribute_column]

    #pathfn = "data/bezdekIris.data"
    pathfn = "data/iris.data"
    data = []
    vocab = {}
    counter = 0
    with open(pathfn, 'r') as f:
        for line in f:
            tmp = line.split(',')
            class_label = tmp[4].strip().replace('\n','')
            if class_label not in vocab:
                vocab[class_label] = counter
                counter += 1
            data.append('{} {}'.format(' '.join(['{}'.format(float(tmp[x])) for x in attribute_columns]), vocab[class_label]))

    m =  np.matrix(';'.join([x for x in data]))
    utils.printf('Data: matrix {}x{}'.format(m.shape[0],m.shape[1]))
    return m

######################################################################################################################
# FUNCTIONS TOI EXAMPLE
# https://alitarhini.files.wordpress.com/2010/11/hw2.ppt
######################################################################################################################
def toi_example(min_expected_value=0.5, max_number_intervals=6, threshold=2.71):
    chi = ChiMerge(min_expected_value, max_number_intervals, threshold)
    data = _readToiExample()
    chi.loadData(data, True)
    chi.generateFrequencyMatrix()
    chi.chimerge()
    chi.printFinalSummary()

def _readToiExample():
    '''
    Reference: https://alitarhini.files.wordpress.com/2010/11/hw2.ppt
    :return:
    '''

    m =  np.matrix('1 1;3 2;7 1;8 1;9 1;11 2;23 2;37 1;39 2;45 1;46 1;59 1')
    utils.printf('Data: matrix {}x{}'.format(m.shape[0],m.shape[1]))
    return m


######################################################################################################################
# INIT
# ChiMerge paper: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf
######################################################################################################################
if __name__ == '__main__':
    example_chimerge_irisdb(attribute_column=0, min_expected_value=0.5, max_number_intervals=6, threshold=4.61)
    # example_chimerge_irisdb(attribute_column=1, min_expected_value=0.5, max_number_intervals=6, threshold=4.61)
    # example_chimerge_irisdb(attribute_column=2, min_expected_value=0., max_number_intervals=6, threshold=4.61)
    # example_chimerge_irisdb(attribute_column=3, min_expected_value=0., max_number_intervals=6, threshold=4.61)
    # toi_example(min_expected_value=0.0, max_number_intervals=6, threshold=2.71)
    # example_chi2_irisdb(alpha=0.5, delta=0.05, min_expected_value=0.1)