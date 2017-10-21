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

from gensim.models import Word2Vec
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
def word2vec(data, window, min_count, size, iterations):
    model = Word2Vec(train_data, window = window, min_count = min_count, 
                            size = size, iter = iterations)
    return model.wv
#%%
def data_reduce(data, size, wordVectorsModel):
    
    data_reduced = np.empty((len(data),size))

    for i in xrange(0, len(data)):

        row_average = np.zeros((1,size))
    
        for x in xrange(0, len(data[i])):
            row_average += wordVectorsModel[data[i][x]]
    
        if len(data[i]) != 0:
            row_average /= len(data[i])
        
        data_reduced[i] = row_average
    
    return data_reduced
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
path = "./train_data.csv"
df = pd.read_csv(path)
df
#%%
data, labels = polishDataSet(df)
#%%
test_split = 0.20
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, 
                                                                    test_size = test_split, 
                                                                    random_state = 10)
#%%
vocab = vocabBuilder(train_data, unknown = True, min_no_of_words = 2)
#%%
train_data_fit = fit_unknown_token(train_data, vocab)
test_data_fit = fit_unknown_token(test_data, vocab)
#%%
size = 100
wordVectorsModel = word2vec(train_data, 4, 2, size, 5).wv
#%%
train_data_reduced = data_reduce(train_data, size, wordVectorsModel)
test_data_reduced = data_reduce(test_data, size, wordVectorsModel)    
#%%
model = makeModel(train_data_reduced, train_labels)
#%%
epochs = 3
batch_size = 128
model.fit(train_data_padded, train_labels, epochs = epochs, batch_size = batch_size)
#%%
model.evaluateModel(test_data_reduced, test_labels)
