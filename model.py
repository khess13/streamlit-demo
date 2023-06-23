""" Module providing function for saving model """
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score# , recall_score

ROOT = os.getcwd()
FILE_LOC = ROOT + '\\data\\TrainingDatav2.csv'
PICKLE_LOC = ROOT + '\\data\\'


'''classifier settings'''
extra = ['inc', 'lp', '&', 'llc', 'sc', ',', '-']
stop = extra
SEED_NO = 309

# read file, dtype specs for faster read
acct_trans = pd.read_csv(FILE_LOC,
                         header=0,
                         usecols=['GLKey',
                                  'Ven_LD_Header',
                                  'VenTxt',
                                  'Long Description',
                                  'IT?',
                                  'RE?'],
                         dtype={'GLKey':str,
                                'Ven_LD_Header': str,
                                'VenTxt': str,
                                'Long Description': str,
                                'IT?': str,
                                'RE?':str},
                         encoding='iso-8859-1')  # because fu excel

# to fix 'Integer value of NA in column x'
acct_trans.dropna(how='all', inplace=True)
acct_trans['IT?'] = acct_trans['IT?'].apply(lambda x: int(x))
acct_trans['RE?'] = acct_trans['RE?'].apply(lambda x: int(x))
acct_trans = acct_trans.replace(np.nan, '', regex=True)

# large C value will choose smaller-margin hyperplane ---> strives to label data more finely
# small C value will choose larger-margin hyperplane ---> strives to label data more broadly
text_clf = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1, 3), stop_words=stop)),
    ('clf', LinearSVC(C=0.5))  # 0.3
])

# create train/set sets
d_train, d_test, l_train, l_test = train_test_split(acct_trans['Ven_LD_Header'], # data
                                                    acct_trans['GLKey'], # label
                                                    random_state=SEED_NO)

# train
features = text_clf.fit(d_train, l_train)
# predict
predicted = text_clf.predict(d_test)

# model metrics
#how often the model is right against labels
accuracy = np.round(accuracy_score(l_test, predicted) * 100, decimals = 2)
#quality of labels
# recall = np.round(recall_score(l_test, predicted) * 100, decimals = 2)
print(f'Accuracy {accuracy}%')
# print(f'Precision of guess {recall}%')

# pickle saves python objects, wb = write in bytes for >py2
# to save

with open(PICKLE_LOC+'class.pickle', 'wb') as save:
    pickle.dump(text_clf, save)