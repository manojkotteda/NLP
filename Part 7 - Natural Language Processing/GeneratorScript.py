


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the trainDataSet
trainDataSet = pd.read_csv('Train.csv');
testDataset = pd.read_csv('Test.csv');
resultDataset = pd.read_csv('Results.csv');
queAns = trainDataSet.iloc[: , :2];
distractor = trainDataSet.iloc[: , 2:];

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

queCorpus = []
distractorCorpus = []
ansCorpus = []
ps = PorterStemmer()
for i in range(0, 31499):
    review1 = re.sub('[^a-zA-Z]', ' ', distractor['distractor'][i])
    review2 = re.sub('[^a-zA-Z]', ' ', queAns['question'][i]) 
    review3 = re.sub('[^a-zA-Z]', ' ', queAns['answer_text'][i]) 
    
    review1 = review1.split()
    review2 = review2.split()
    review3 = review3.split()
    
    review1 = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
    review2 = [ps.stem(word) for word in review2 if not word in set(stopwords.words('english'))]
    review3 = [ps.stem(word) for word in review3 if not word in set(stopwords.words('english'))]
    
    review1 = ' '.join(review1)
    review2 = ' '.join(review2)
    review3 = ' '.join(review3)
    
    queCorpus.append(review2)
    ansCorpus.append(review3)
    distractorCorpus.append(review1)

reviewDataFrame =  pd.DataFrame(
    {'question': queCorpus,
     'answer_text':ansCorpus,
     'distractor': distractorCorpus
    })
    
reviewDataFrame.to_csv('reviewDataFrame.csv');
    
    
    