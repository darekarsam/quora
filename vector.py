import pandas as pd



trainDF = pd.read_csv("Data/train.csv")
testDF = pd.read_csv("Data/test.csv")
import ipdb; ipdb.set_trace()
train = trainDF.sample(frac=0.8,random_state=200)
test = trainDF.drop(train.index)

train, test = train_test_split(trainDF, test_size=0.2)
