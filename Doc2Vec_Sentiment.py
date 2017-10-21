# -*- coding: utf-8 -*-
"""
Created on Sat Sep 02 00:34:54 2017

@author: shyam
"""

import pandas as pd
import re

from nltk.corpus import stopwords
from nltk import FreqDist

from sklearn.model_selection import train_test_split

import numpy as np

import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras import optimizers

from gensim.models import Doc2Vec
#%%
def polishData(df):
    
    data = []

    for i in xrange(0, df.shape[0]):
        no_links = re.sub("https?:\/\/.*[\r\n]*", " ", df.iloc[:,0][i], flags=re.MULTILINE)

        letters_only = re.sub("[^a-zA-Z]", " ", no_links)

        lower_case = letters_only.lower()
        words = lower_case.split()
    
        words = [w for w in words if not w in stopwords.words("english")]
    
        data.append(words)
    
    return data
#%%
def labelsOneHot(df):
    sentiOneHot = pd.get_dummies(df.iloc[:,1])

    labels = np.empty((sentiOneHot.shape), dtype = int)

    for i in xrange(0,sentiOneHot.shape[1]):
        numbers = np.array(sentiOneHot.iloc[:,i])
        labels[:,i] = numbers

    return labels
#%%
def polishDataSet(df):
    data = polishData(df)
    labels = labelsOneHot(df)
    
    return data, labels
#%%
def vocabBuilder (data, unknown = True, min_no_of_words = 1):
    tokens = []
    
    for i in xrange(0, len(data)):
        for j in xrange(0,len(data[i])):
            tokens.append(data[i][j])
            
    freqdist = FreqDist(tokens)
    
    vocab = []
    
    for key in freqdist:
        if freqdist[key] >= min_no_of_words:
            vocab.append(key)
    
    if unknown:
        vocab.append('UNKNOWN')
    
    return vocab
#%%
def fit_unknown_token(data, vocab):
    
    data_unknown = data
    
    for i in xrange(0, len(data)):
        for j in xrange(0, len(data[i])):
            if data[i][j] not in vocab:
                data_unknown[i][j] = 'UNKNOWN'
                
    return data_unknown
#%%
def doc2vec(train_data, path, size, min_count, epochs):
    model = Doc2Vec(train_data, min_count = min_count, size = size, epochs = epochs)
    return model
#%%
def makeModel(train_data_reduced, train_labels, hidden_layer = 30, 
              activation = 'relu', optimizer = 'adam', loss = 'categorical_crossentropy'):
    
    adamOptimizer = keras.optimizers.Adam(lr=0.0008, decay = 0.00001)
    model = Sequential()
    model.add(Dense(hidden_layer,input_dim = train_data_reduced.shape[1], activation = activation))
    model.add(Dropout(0.4))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(train_labels.shape[1],activation = 'softmax'))
    model.compile(optimizer = adamOptimizer, loss = loss, metrics = ['accuracy'])
    
    return model
#%%
def evaluateModel(test_data, test_labels):   
    scores = model.evaluate(test_data, test_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))   
#%%
class LabeledLineSentence(object):
    def __init__(self, filename, training_labels):
        self.filename = filename
        self.training_labels = training_labels
        self.trained_labels = np.where(training_labels == 1)
        
    def __iter__(self):
        with gensim.utils.smart_open(self.filename) as fin:
            for uid, line in enumerate(fin):
                print uid, line
                yield gensim.models.doc2vec.LabeledSentence(words = line.split(), tags = ['SENT_%s' % uid])
            
    def to_array(self):
        self.sentences = []
        for uid, line in enumerate(open(self.filename)):
            print uid, line
            self.sentences.append(gensim.models.doc2vec.LabeledSentence(words = line.split(), tags = ['SENT_%s' % self.trained_labels[uid-1]]))
        return self.sentences
#%%
path = "./train_data.csv"
df = pd.read_csv(path)
#%%
data, labels = polishDataSet(df)
#%%
test_split = 0.20
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = test_split, random_state = 42)
#%%
vocab = vocabBuilder(train_data, unknown = True, min_no_of_words = 2)
#%%
train_data_fit = fit_unknown_token(train_data, vocab)
test_data_fit = fit_unknown_token(test_data, vocab)
#%%
np.savetxt('./train.txt', train_data_fit, fmt = "%s")

with open("./train.txt", "a") as myfile:
    np.savetxt(myfile, test_data_fit, fmt = "%s")

#%%
path = './mymodel'
size = 100
sentences = LabeledLineSentence('./train.txt', train_labels)
wordModel = Doc2Vec(sentences, min_count=1, window = 2, size = size, epochs = 10)
#wordModel.build_vocab(sentences)
#%%
#wordModel.train(sentences, total_examples = wordModel.corpus_count, epochs = 10)
#%%
train_data_reduced = np.empty((len(train_data),size))

for i in xrange(0, len(train_data)):
    train_data_reduced[i] = wordModel.docvecs[i]

#%%
test_data_reduced = np.empty((len(test_data),size))

for i in xrange(0, len(test_data)):
    test_data_reduced[i] = wordModel.docvecs[24000+i]

#%%
model = makeModel(train_data_reduced, train_labels)
#%%
epochs = 3
batch_size = 128
model.fit(train_data_reduced, train_labels, epochs = epochs, batch_size = batch_size)
#%%
model.evaluateModel(test_data_reduced, test_labels)

