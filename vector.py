import pandas as pd
from gensim import models
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import *

print "Reading Dataframe..."
trainDF = pd.read_csv("Data/train.csv")
testDF = pd.read_csv("Data/test.csv")

trainDF.question1.fillna('abc', inplace=True)
trainDF.question2.fillna('abc', inplace=True)

def cleanQuestion(question):
    """Functions to clean question pairs
    """
    # convert to lower case
    question = question.lower()
#     question = question.replace()
    question = question.decode('utf-8')
    stop = stopwords.words('english')
    stemmer = PorterStemmer()
    import ipdb; ipdb.set_trace()
    questions = re.sub(r'[^\x00-\x7F]+', ' ', question, flags=re.MULTILINE)
    question = ' '.join([word for word in question.split() if word not in stop])
    question = ' '.join([stemmer.stem(word) for word in question.split()])
    # remove extra spaces
    question = re.sub(r'[\s]+', ' ', question, flags=re.MULTILINE)
    # remove all punctuations
    question = re.sub(r'[^a-zA-Z]', ' ', question, flags=re.MULTILINE)
    return question

def getLabeledSentence(questions, label):
    labeledQuestions=[]
    for uid, line in enumerate(questions):
        labeledQuestions.append( models.doc2vec.LabeledSentence(words=line.split(), tags=[label+str(uid)]))
    
    return labeledQuestions
    
trainDF.question1 = trainDF.question1.apply(cleanQuestion)
trainDF.question2 = trainDF.question2.apply(cleanQuestion)



def getLabeledSentence(questions, label):
    labeledQuestions=[]
    for uid, line in enumerate(questions):
        labeledQuestions.append( models.doc2vec.LabeledSentence(words=line.split(), tags=[label+str(uid)]))
    
    return labeledQuestions

#retaining indexes to retrieve vectors for questions
train.reset_index(inplace=True)
train['index']=train.index

print "Cleaning Dataframe..."
train.question1 = train.question1.apply(cleanQuestion)
train.question2 = train.question2.apply(cleanQuestion)

labeledQuestions1 = getLabeledSentence(trainDF.question1.tolist(), 'question1')
labeledQuestions2 = getLabeledSentence(trainDF.question2.tolist(), 'question2')
model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1)
labeledQuestions = labeledQuestions1 + labeledQuestions2

print "Building Vocab..."
model.build_vocab(labeledQuestions)

print "Training model..."
model.train(labeledQuestions)

print "Saving model..."
model.save('Models/4_23model')

model_loaded = Doc2Vec.load('Models/4_8model')

label = []
for i in range(len(labeledQuestions1)):
    label.append(model.docvecs.similarity('question1%d'%i, 'question2%d'%i))
import ipdb; ipdb.set_trace()
similarityDF = pd.DataFrame(label)
similarityDF['y'] = train.is_duplicate

vectors1 = []
vectors2 = []
print "Fetching Vectors..."
for i in range(len(labeledQuestions1)*2):
    if i< len(labeledQuestions1):
        vectors1.append(model.docvecs[i])
    else:
        vectors2.append(model.docvecs[i])
import ipdb; ipdb.set_trace()

vectors1 = np.asarray(vectors1)
vectors2 = np.asarray(vectors2)

labels = np.asarray(train.is_duplicate)


