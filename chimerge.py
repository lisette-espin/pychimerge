__author__ = 'lisette.espin'

######################################################################################################################
# SYSTEM DEPENDENCES
######################################################################################################################
import numpy as np
import math

######################################################################################################################
# LOCAL DEPENDENCES
######################################################################################################################
import utils

######################################################################################################################
# CONSTANTS
# Threshold reference: (http://sites.stat.psu.edu/~mga/401/tables/Chi-square-table.pdf)
######################################################################################################################
MIN_NUMBER_INTERVALS = 2

######################################################################################################################
# ChiMerge CLASS
######################################################################################################################
class ChiMerge():
    '''
    1992 by R. Kerber
    Reference: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf
    '''

    def __init__(self, min_expected_value, max_number_intervals, threshold):
        '''
        chi-square distribution table: http://sites.stat.psu.edu/~mga/401/tables/Chi-square-table.pdf
        :param min_expected_value:
        :param max_number_intervals:
        :param threshold:
        :return:
        '''
        self.data = None
        self.sorted_data = None
        self.frequency_matrix = None
        self.frequency_matrix_intervals = None
        self.nclasses = -1
        self.nattrinutes = -1
        self.degrees_freedom = -1
        self.min_expected_value = min_expected_value
        self.min_number_intervals = MIN_NUMBER_INTERVALS
        self.max_number_intervals = max_number_intervals
        self.threshold = threshold

    def loadData(self, data, issorted=False):
        '''
        :param data: numpy matrix
        :param issorted: boolean, if data is already sorted, no need to sort again (based on attribute_column)
        :return:
        '''
        if type(data) != np.matrix and type(data) != np.array:
            utils.printf('ERROR: data must be a numpy.matrix or numpy.array')
            return

        self.data = data # numpy.matrix (x,2). column index 0 refers to attributes column and index 1 classes
        if not issorted:
            self.sorted_data = np.array(np.sort(data.view('i8,i8'), order=['f0'], axis=0).view(np.float))   #always sorting column 0 (attribute column)
        else:
            self.sorted_data = np.array(data)
        utils.printf('Sorted data: matrix {}x{}'.format(self.sorted_data.shape[0],self.sorted_data.shape[1]))

    def loadFrequencyMatrix(self, frequency_matrix, unique_attribute_values):
        '''
        :param frequency_matrix: numpy array
        :return: void
        '''
        if type(frequency_matrix) != np.array:
            utils.printf('ERROR: data must be a numpy.array')
            return
        self.frequency_matrix = frequency_matrix
        self.frequency_matrix_intervals = unique_attribute_values
        self.nclasses = self.frequency_matrix.shape[1]
        self.degrees_freedom = self.nclasses - 1
        self.printInitialSummary()

    def generateFrequencyMatrix(self):

        if self.sorted_data is None:
            utils.printf('ERROR: Your (sorted) data should be loaded!')
            return
        if self.sorted_data.shape[1] != 2:
            utils.printf('ERROR: Your (sorted) matrix should have 2 columns only (attribute, class)')
            return

        unique_attribute_values, indices = np.unique(self.sorted_data[:,0], return_inverse=True)    # first intervals: unique attribute values
        unique_class_values = np.unique(self.sorted_data[:,1])                                      # classes (column index 1)
        self.frequency_matrix = np.zeros((len(unique_attribute_values), len(unique_class_values)))  # init frequency_matrix
        self.frequency_matrix_intervals = unique_attribute_values                                   # init intervals (unique attribute values)
        self.nclasses = len(unique_class_values)                                                    # number of classes
        self.degrees_freedom = self.nclasses - 1                                                    # degress of freedom (look at table)

        # Generating first frequency values (contingency table), number of instances found in data: attribute-class
        for row in np.unique(indices):
            for col, clase in enumerate(unique_class_values):
                self.frequency_matrix[row,col] += np.where(self.sorted_data[np.where(indices == row)][:,1] == clase)[0].shape[0]
        self.printInitialSummary()

    def chisqrtest(self, array):
        '''
        :param array: np.array 2 consecutive rows from frequeny attribute/class matrix, e.g.,: a = np.matrix('16 0 0; 4 1 1')
        :return chisqr value of distribution of 2 rows
        '''

        shape = array.shape
        N = float(array.sum())  # total number of observations
        r = self._getTotalsPerRow(array)
        c = self._getTotalsPerColumn(array)

        chisqr = 0
        for row in range(shape[0]):
            for col in range(shape[1]):
                e = r[row]*c[col] / N   # expected value
                o = array[row,col]      # observed value
                e = self.min_expected_value if e < self.min_expected_value else e
                chisqr += 0. if e == 0. else math.pow((o - e),2) / float(e)

        return chisqr

    def chimerge(self):
        if self.frequency_matrix is None:
            utils.printf('ERROR: Your frequency matrix should be loaded!')
            return

        chitest = {}
        counter = 0
        smallest = -1

        while self._continue() or self._merge(smallest):

            ###
            # CHI2 TEST
            ###
            chitest = {}
            shape = self.frequency_matrix.shape
            for r in range(shape[0] - 1):
                interval = r,r+1
                chi2 = self.chisqrtest(self.frequency_matrix[[interval],:][0])
                if chi2 not in chitest:
                    chitest[chi2] = []
                chitest[chi2].append( (interval) )
            smallest = min(chitest.keys())
            biggest = max(chitest.keys())

            ###
            # SUMMARY
            ###
            counter += 1
            utils.printf('')
            utils.printf('ROUND {}: {} intervals. Chi min:{}, Chi max:{}'.format(counter, self.frequency_matrix.shape[0], smallest, biggest))

            ###
            # MERGE ?
            ###
            if self._merge(smallest):
                utils.printf('MERGING INTERVALS: chi {} -> {}'.format(smallest, chitest[smallest]))
                for (lower,upper) in list(reversed(chitest[smallest])):                                     # reversed, to be able to remove rows on the fly
                    for col in range(shape[1]):                                                             # checking columns (to append values from row i+1 ---to be removed--- to row i)
                        self.frequency_matrix[lower,col] += self.frequency_matrix[upper,col]                # appending frequencies to the remaining interval
                    self.frequency_matrix = np.delete(self.frequency_matrix, upper, 0)                      # removing interval (because we merged it in the previous step)
                    self.frequency_matrix_intervals = np.delete(self.frequency_matrix_intervals, upper, 0)  # also removing the corresponding interval (real values)
                utils.printf('NEW INTERVALS: ({}):{}'.format(len(self.frequency_matrix_intervals),self.frequency_matrix_intervals))

        self.chitestvalues = chitest
        utils.printf('END (chi {} > {})\n'.format(smallest, self.threshold))

    ##############################################################
    # Printing (output)
    ##############################################################

    def printInitialSummary(self):
        utils.printf('')
        utils.printf('ROUND 0: Initial values:')
        utils.printf('- Number of classes: {}'.format(self.nclasses))
        utils.printf('- Degrees of Freedom: {} (deprecated)'.format(self.degrees_freedom))
        utils.printf('- Threshold: {}'.format(self.threshold))
        utils.printf('- Max number of intervals: {}'.format(self.max_number_intervals))
        utils.printf('- Number of (unique) intervals: {}'.format(len(self.frequency_matrix_intervals)))
        utils.printf('- Frequency matrix: {}x{} (sum {})'.format(self.frequency_matrix.shape[0], self.frequency_matrix.shape[1], self.frequency_matrix.sum()))
        utils.printf('- Intervals: {}'.format(self.frequency_matrix_intervals))

    def printFinalSummary(self):
        utils.printf('FINAL SUMMARY')
        utils.printf('{}{}'.format('Intervals: ',self.frequency_matrix_intervals))
        utils.printf('{}{}'.format('Chi2: ',', '.join(['[{}-{}):{:5.1f}'.format(v[0][0],v[0][1],k) for k,v in utils.sortDictByValue(self.chitestvalues,False)])))
        utils.printf('{} ({}x{})\n{}'.format('Interval-Class Frequencies',self.frequency_matrix.shape[0],self.frequency_matrix.shape[1],self.frequency_matrix))

    ##############################################################
    # Handlers
    ##############################################################

    def _merge(self, smallestchi2):
        return smallestchi2 < self.threshold

    def _continue(self):
        return self.frequency_matrix.shape[0] >= self.max_number_intervals

    def _getTotalsPerRow(self, narray):
        '''
        :param narray: numpy.array 2 consecutive rows from frequeny attribute/class
        :return: dictionary with total number of observations per i_th row
        '''
        shape = narray.shape
        r = {}
        for i in range(shape[0]):
            r[i] = narray[i].sum()
        return r

    def _getTotalsPerColumn(self, narray):
        '''
        :param narray: numpy.array 2 consecutive rows from frequeny attribute/class matrix
        :return: dictionary with total number of observations per j_th column
        '''
        shape = narray.shape
        c = {}
        for j in range(shape[1]):
            c[j] = narray[:,j].sum()
        return c