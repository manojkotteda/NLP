# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:55:50 2019

@author: manojkotheda
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Train.csv');

import re
from nltk.tokenize import sent_tokenize,word_tokenize 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
exp_text='hello there? how are you';
print(sent_tokenize(dataset['distractor'][0]));

https://nexuscimgmt.sgp.dbs.com:8443/nexus/repository/dbsrepo

nltk.set_proxy('https://nexuscimgmt.sgp.dbs.com:8443/nexus/repository/dbsrepo', ('manojkotheda', 'Newpassword123'))
