import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from gensim import corpora
from nltk.corpus import stopwords
import re

# http://radimrehurek.com/gensim/tut1.html
# http://radimrehurek.com/gensim/tut2.html

trainDF=pd.read_csv("Data/train.csv")
testDF=pd.read_csv("Data/test.csv")

trainDF.question1.fillna('', inplace=True)
trainDF.question2.fillna('', inplace=True)
testDF.question1.fillna('', inplace=True)
testDF.question2.fillna('', inplace=True)


trainDF['data']='train'
testDF['data']='test'
frame = [trainDF,testDF]
df = pd.concat(frame,ignore_index = True)

frame = [df.question1,df.question2]
# stop = stopwords.words('english')
stop=['my','i','the','for']
questions = pd.concat(frame,ignore_index = True)
questions=questions.str.lower()
questions=questions.replace('?','')

# questions=questions.apply(lambda x: [item for item in x if item not in stop])

def cleanQuestions(question):
    question=re.sub(r'[\s]+', ' ', question,flags=re.MULTILINE)
    return question

questions=questions.apply(cleanQuestions)

questions=questions.str.split()
import ipdb; ipdb.set_trace()
questions=list(questions)

dictionary=corpora.Dictionary(questions)
dictionary.save('/Models/dict.dict')