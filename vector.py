import re
import pandas as pd
from gensim import models
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import *
import cPickle as pickle
import time
"""
print "Reading Dataframe..."
trainDF = pd.read_csv("Data/train.csv")
testDF = pd.read_csv("Data/test.csv")

trainDF.question1.fillna('abc', inplace=True)
trainDF.question2.fillna('abc', inplace=True)
testDF.question1.fillna('abc', inplace=True)
testDF.question2.fillna('abc', inplace=True)
"""
def cleanQuestion(question):
    """Functions to clean question pairs
    """
    # convert to lower case
    question = question.lower()
#     question = question.replace()
    question = question.decode('utf-8')
    stop = stopwords.words('english')
    #stemmer = PorterStemmer()
    question = re.sub(r'[^\x00-\x7F]+', ' ', question, flags=re.MULTILINE)
    question = ' '.join([word for word in question.split() if word not in stop])
    #question = ' '.join([stemmer.stem(word) for word in question.split()])
    # remove extra spaces
    question = re.sub(r'[\s]+', ' ', question, flags=re.MULTILINE)
    # remove all punctuations
    question = re.sub(r'[^a-zA-Z]', ' ', question, flags=re.MULTILINE)
    return question

def getLabeledSentence(questions, label):
    labeledQuestions = []
    for uid, line in enumerate(questions):
        labeledQuestions.append( models.doc2vec.LabeledSentence(words=line.split(), tags=[label+str(uid)]))
    
    return labeledQuestions

#time for appending to filename
#timestr = time.strftime("%Y%m%d-%H%M%S")
"""
print "Cleaning Train Dataframe..."
trainDF.question1 = trainDF.question1.apply(cleanQuestion)
trainDF.question2 = trainDF.question2.apply(cleanQuestion)
print "Saving Train pickle file..."
pickle.dump (trainDF, open("Data/cleaned_nonStemmed_train.pickle","wb"))
"""
print "loading Train data from pickle file"
trainDF = pickle.load( open( "Data/cleaned_nonStemmed_train.pickle", "rb" ))

"""
print "Cleaning Test Dataframe..."
testDF.question1 = testDF.question1.apply(cleanQuestion)
testDF.question2 = testDF.question2.apply(cleanQuestion)
print "Saving Test pickle file..."
pickle.dump (testDF, open("Data/cleaned__nonStemmed_test.pickle","wb"))
"""
print "loading Test data from pickle file"
testDF = pickle.load( open( "Data/cleaned__nonStemmed_test.pickle", "rb" ))

print "Labeling Questions..."
labeledQuestions1 = getLabeledSentence(trainDF.question1.tolist(), 'question1')
labeledQuestions2 = getLabeledSentence(trainDF.question2.tolist(), 'question2')
labeledQuestionstest1 = getLabeledSentence(testDF.question1.tolist(), 'testquestion1')
labeledQuestionstest2 = getLabeledSentence(testDF.question2.tolist(), 'testquestion2')

model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1)
labeledQuestions = labeledQuestions1 + labeledQuestions2 + labeledQuestionstest1 + labeledQuestionstest2
"""
print "Building Vocab..."
model.build_vocab(labeledQuestions)

print "Training model..."
model.train(labeledQuestions)

print "Saving model..."
model.save('Models/4_23model_unstemmed_test_train')
"""

print 'loading Model'
model = models.Doc2Vec.load('Models/4_23model_unstemmed_test_train')

"""print "finding similarity..."
similarity = []
for i in range(len(labeledQuestions1)):
    similarity.append(model.docvecs.similarity('question1%d'%i, 'question2%d'%i))
"""
"""
similarityTest = []
for i in range(len(labeledQuestionstest1)):
    similarityTest.append(model.docvecs.similarity('testquestion1%d'%i, 'testquestion2%d'%i))
similarityTest = np.asarray(similarityTest)
similarityTest[similarityTest<0] = 0

import ipdb; ipdb.set_trace()
similarityDF = pd.DataFrame()
similarityDF['test_id'] = testDF.test_id
similarityDF['is_duplicate'] = pd.Series(similarityTest)

similarityDF.to_csv('cosineSimilarity.csv', index=False)
"""
vectors1 = []
vectors2 = []
print "Fetching Train Vectors..."
for i in range(len(labeledQuestions1)*2):
    if i< len(labeledQuestions1):
        vectors1.append(model.docvecs[i])
    else:
        vectors2.append(model.docvecs[i])
#0.913 to 0.443
testvectors1 = []
testvectors2 = []
print "Fetching Test Vectors..."
for i in range(len(labeledQuestions1)*2, len(labeledQuestions)):
    if i < (len(labeledQuestions1)*2) + len(labeledQuestionstest1):
        testvectors1.append(model.docvecs[i])
    else:
        testvectors2.append(model.docvecs[i])

vectors1 = np.asarray(vectors1)
vectors2 = np.asarray(vectors2)

testvectors1 = np.asarray(testvectors1)
testvectors2 = np.asarray(testvectors2)

vectorDiff = np.subtract(vectors1, vectors2)
testvectorDiff = np.subtract(testvectors1, testvectors2)

labels = np.asarray(trainDF.is_duplicate)

from sklearn import metrics
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier

#training Neural Network
clf = MLPClassifier(hidden_layer_sizes=(600,300,40,10,10),max_iter=150, alpha=1e-4, solver='sgd',verbose=True, \
	tol=0.0001, learning_rate_init=0.0005)
print "training classifier ..."
#clf.fit(vectorDiff, labels)
#pickle.dump(clf, open('Models/nn_150_unstemmed.pickle', 'wb'))
clf = pickle.load( open('Models/nn_150_unstemmed.pickle', "rb" ))
import ipdb; ipdb.set_trace()
Y_test = clf.predict_proba(testvectorDiff)

sub = pd.DataFrame()
sub['test_id'] = testDF.test_id
sub['is_duplicate'] = pd.Series(zip(*Y_test)[1])
sub.to_csv('simple_nn.csv', index=False)

