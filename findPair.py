import pandas as pd
import numpy as np
from gensim import corpora
from nltk.corpus import stopwords
import re

trainDF = pd.read_csv("Data/train.csv")
testDF = pd.read_csv("Data/test.csv")

trainDF['data'] = 'train'
testDF['data'] = 'test'
fullDF = pd.concat([trainDF, testDF], ignore_index=True)

def cleanQuestion(question):
    """Functions to clean question pairs
    """
    import ipdb; ipdb.set_trace()
    # convert to lower case
    question = question.lower()
    # remove extra spaces
    question = re.sub(r'[\s]+', ' ', question, flags=re.MULTILINE)
    # remove all punctuations
    question = re.sub(r'[^a-zA-Z]', ' ', question, flags=re.MULTILINE)
    

fullDF.question1 = fullDF.question1.apply(cleanQuestion)
