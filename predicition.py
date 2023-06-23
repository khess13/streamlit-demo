''' Provides files funct '''
import pickle
import os

ROOT = os.getcwd()

def predict(data):
    ''' Opens model returns result'''
    if len(data) > 0:
        clf = pickle.load(open(ROOT + '\\data\\class.pickle', 'rb'))
        # because accepts [] and not strings
        pred = clf.predict([data])
        return pred
    return -1