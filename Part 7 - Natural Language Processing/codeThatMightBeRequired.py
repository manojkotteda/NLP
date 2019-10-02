# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:55:50 2019

@author: manojkotheda
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the trainDataSet
trainDataSet = pd.read_csv('Train.csv');
testDataset = pd.read_csv('Test.csv');
resultDataset = pd.read_csv('Results.csv');
queAns = trainDataSet.iloc[: , :2];
distractor = trainDataSet.iloc[: , 2:];

#######################################################################################################
splitter = re.compile(r",[^a-zA-Z]")

data = splitter.split(trainDataSet['distractor'][0]);
count = 0;
excessData = []
for x in range(0,31499):
    data = splitter.split(trainDataSet['distractor'][x]);
    if(len(data)>3):
        excessData.append(data) 
        count=count+1
    
            
distractor = pd.DataFrame(data, columns = ['distractor1', 'distractor2','distractor3']); 

#############################################################################################################

import re
from nltk.tokenize import sent_tokenize,word_tokenize 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

print(re.sub(r',[^a-zA-Z]', '||', trainDataSet['distractor'][0]))
print(re.sub('[^a-zA-Z]', ',', trainDataSet['distractor'][0]))
string = trainDataSet['distractor'][0];
print(sent_tokenize(string,'english'))




https://nexuscimgmt.sgp.dbs.com:8443/nexus/repository/dbsrepo

nltk.set_proxy('https://nexuscimgmt.sgp.dbs.com:8443/nexus/repository/dbsrepo', ('manojkotheda', 'Newpassword123'))
