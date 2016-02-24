__author__ = 'lisette.espin'

######################################################################################################################
# SYSTEM DEPENDENCES
######################################################################################################################
from datetime import datetime
import operator

######################################################################################################################
# FUNCTIONS
######################################################################################################################
def printf(msg):
    strtowrite = "[{}] {}".format(datetime.now(), msg)
    print(strtowrite)

def sortDictByValue(x,desc):
    sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=desc)
    return sorted_x

def sortDictByKey(x,desc):
    sorted_x = sorted(x.items(), key=operator.itemgetter(0),reverse=desc)
    return sorted_x