# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:42:05 2019

@author: manojkotheda
"""

import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;
import matplotlib.pyplot as plt


# Importing the trainDataSet
reviewDataFrame = pd.read_csv('reviewDataFrame.csv');



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X_counts = cv.fit_transform(reviewDataFrame['question'].values.astype('U')).toarray()

transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(X_counts);

xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

num_topics = 10;
#obtain a NMF model.
model = NMF(n_components=num_topics, init='nndsvd');
#fit the model
model.fit(xtfidf_norm)

def get_nmf_topics(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = cv.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);

opdataframe = get_nmf_topics(model, 20)

