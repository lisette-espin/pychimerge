__author__ = 'lisette.espin'

######################################################################################################################
# SYSTEM DEPENDENCES
######################################################################################################################
import numpy as np
import math
import json

######################################################################################################################
# LOCAL DEPENDENCES
######################################################################################################################
import utils
from chimerge import ChiMerge

######################################################################################################################
# CONSTANTS
######################################################################################################################
SIGLEVELMINUS = 0.1

######################################################################################################################
# Chi2 CLASS
######################################################################################################################
class Chi2():
    '''
    1995 by Liu et al.
    Reference: http://sci2s.ugr.es/keel/pdf/specific/congreso/liu1995.pdf
    '''

    def __init__(self, alpha, delta, min_expected_value):
        '''
        :param alpha: siglevel
        :param delta: consistency test
        :return:
        '''
        self.data = None
        # self.sorted_data = None
        # self.frequency_matrix = None
        # self.frequency_matrix_intervals = None
        self.chimerge_per_column = None
        self.alpha_per_column = None
        self.attribute_can_be_merged = None
        self.nclasses = -1
        self.nattributes = -1
        self.degrees_freedom = -1
        self.alpha = alpha
        self.delta = delta
        self.chidistribution = None
        self.min_expected_value = min_expected_value

    def loadData(self, data):
        '''
        :param data: numpy matrix
        :return:
        '''
        if type(data) != np.matrix and type(data) != np.array:
            utils.printf('ERROR: data must be a numpy.matrix or numpy.array')
            return
        self.data = np.array(data) # no need to sort at this point
        self.nattributes = self.data.shape[1]-1 # last column refers to class label
        self.nclasses = np.unique(self.data[:,self.nattributes]).shape[0]
        self.degrees_freedom = self.nclasses - 1
        self.chimerge_per_column = {colid:None for colid in range(self.nattributes)}
        self.alpha_per_column = {colid:None for colid in range(self.nattributes)}
        self.attribute_can_be_merged = {colid:True for colid in range(self.nattributes)}
        utils.printf('Data: matrix {}x{} ({} numeric attributes)'.format(self.data.shape[0],self.data.shape[1], self.nattributes))
        self._loadChiDistribution()

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

    def chi2(self):
        if self.data is None:
            utils.printf('ERROR: Your data matrix should be loaded!')
            return

        ### Phase1: defining sigLevel values for every numeric attribute, and chimerge for every attribute-column
        sigLevel0 = self._phase1()

        ### Phase2: merging attrinutes if needed (vertical-wise)
        self._phase2(sigLevel0)

    def _phase1(self):
        '''
        Perfomrs phase_1 of the Chi2 algorithm (runs chimerge over each attribute-column)
        :return: the smallest sigLevel value
        '''
        sigLevel0 = self.alpha
        while self._inConsistency() < self.delta:
            for attribute_column in range(self.nattributes):    # chimerge for all attribute-columns
                chimerge = ChiMerge(self.min_expected_value,self.data.shape[0],self.chidistribution[self.alpha])
                chimerge.loadData(self.data[:,[attribute_column,self.nattributes]],False) # 1 attribute-column and class column (last column)
                chimerge.generateFrequencyMatrix()
                chimerge.chimerge()
                self.chimerge_per_column[attribute_column] = chimerge

                sigLevel0 = self.alpha
                self.alpha -= self._decreaseSigLevel()
        return sigLevel0

    def _phase2(self, sigLevel0):
        self.alpha_per_column = {colid:sigLevel0 for colid in self.alpha_per_column.keys()}
        while self._attributeColumnsCanBeMerged():
            for colid,canbemerge in self.attribute_can_be_merged.items():
                if canbemerge:
                    chimerge = ChiMerge(self.min_expected_value,self.data.shape[0],self.chidistribution[self.alpha_per_column[colid]])
                    chimerge.loadData(self.data[:,[colid,self.nattributes]],False) # 1 attribute-column and class column (last column)
                    chimerge.generateFrequencyMatrix()
                    chimerge.chimerge()

                    if self._inConsistency() < self.delta:
                        self.alpha_per_column[colid] -= self._decreaseSigLevel()
                    else:
                        self.attribute_can_be_merged[colid] = False

    def _decreaseSigLevel(self):
        return SIGLEVELMINUS

    def _inConsistency(self):
        #1. matrix with all attribute-columns (except class-column)
        #2. find duplicates (register indexes)
        #3. for every duplicated instance do:
        #   3.1. calculate inconsistency_count = (n-ck) where n is the number of time such instance is duplicated and ck the largest number of duplicates of such instance among all classes
        #4. incosistency rate sum all inconsistency_count and divide by the number of instances (total instances)

        #
        # IT SHOULD NOT BE OVER RAW DATA, BUT OVER THE MERGED DATA!!!
        # To be fixed!
        #
        if self.data is None:
            utils.printf('ERROR: Your data matrix should be loaded!')
            return

        # 1. matrix with only attribute values
        # 2. identify duplicates
        unique_values, unique_indexes = np.unique(self.data[:,:self.nattributes-1], return_inverse=True)
        unique_counts = np.bincount(unique_indexes)
        matching_instances = unique_values[unique_counts>1]
        sum_inconsistencies = 0
        total_instances = unique_indexes.shape[0]

        # 3. calculating inconsistency_count for every instance
        for matching_instance in matching_instances:
            c = {}
            for colid in range(self.nclasses):
                c[colid] = (self.data[self.data[:,self.nattributes]==colid] == matching_instance).sum()
            n = sum(c.values())
            cmax = max(c.values())
            inconsistency_count = n - cmax
            sum_inconsistencies += inconsistency_count

        # 4. inconsistency rate
        inconsistency_rate = sum_inconsistencies / float(total_instances)
        return inconsistency_rate

    def _attributeColumnsCanBeMerged(self):
        return not all([flag == False for flag in self.attribute_can_be_merged.values()])

    ##############################################################
    # Printing (output)
    ##############################################################

    def printInitialSummary(self):
        utils.printf('')
        utils.printf('ROUND 0: Initial values:')
        utils.printf('- Number of attributes: {}'.format(self.nattributes))
        utils.printf('- Number of classes: {}'.format(self.nclasses))
        utils.printf('- Degrees of Freedom: {} (deprecated)'.format(self.degrees_freedom))
        utils.printf('- alpha (initial value of sigLevel): {}'.format(self.alpha))
        utils.printf('- delta (inConsistency level): {}'.format(self.delta))

    ##############################################################
    # Handlers
    ##############################################################

    def _loadChiDistribution(self):
        with open('data/chisquare_distribution.data','r') as f:
            data = json.load(f)
        self.chidistribution = {float(k):v for k,v in data.items()}
        utils.printf('ChiSquare distribution table loaded. {} sigLevel and {} degrees of freedom.'.format(len(self.chidistribution.keys()),len(self.chidistribution.values()[0])-1))