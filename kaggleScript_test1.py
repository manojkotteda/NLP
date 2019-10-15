# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.
# %% [code]

from gensim.models import KeyedVectors
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
from numba import jit, cuda
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
# %% [code]
reviewDataFrame = pd.read_csv('/kaggle/input/nlp-datasets/Train.csv');
# %% [code]
distractors = []
questions = []
answers = []
lemmatizer = WordNetLemmatizer() 
for i in range(0, 31499):
    distractor = re.sub(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", ' ', reviewDataFrame['distractor'][i])
    answer = re.sub(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", ' ', reviewDataFrame['answer_text'][i])
    question = re.sub(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", ' ', reviewDataFrame['question'][i])
    distractor = re.sub('   +','',distractor)
    distractor = re.split('  ',distractor)
    for sents in distractor:
        sents = sents.split()
        sents = [lemmatizer.lemmatize(word) for word in sents]
        sents = ' '.join(sents)
        distractors.append(sents)
    questions.append(question)
    answers.append(answer)
# %% [code]
filename = '/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

# %% [code]
model.most_similar('man',topn = 10)

# %% [code]
reviewDataFrame = pd.read_csv('/kaggle/input/nlp-datasets/Test.csv');
distractors = []
questions = []
answers = []
lemmatizer = WordNetLemmatizer() 
for i in range(0, 13500):
    review = re.sub(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", ' ', reviewDataFrame['answer_text'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    answers.append(review)
    review = re.sub(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", ' ', reviewDataFrame['question'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    questions.append(review)   


# %% [code]
answers
