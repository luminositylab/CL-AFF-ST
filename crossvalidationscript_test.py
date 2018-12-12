import pandas as pd
import datetime
import numpy
#import multiprocessing as mp
from dcec_xval_test import model_evaluator
from sklearn.utils import shuffle



class Dataset():
    def __init__(self, train, test, number):
        self.train = train
        self.test = test
        self.number = number
def runmodel(dataset, number):
    evaluator = model_evaluator(dataset.train, dataset.test)
    metrics = evaluator.train()
    evaluator.predict(number)
    print(metrics)
    return metrics#return accuracy
df = pd.read_csv('csv/labeled_10k.csv', header=0)
df = shuffle(df)
df = df.reset_index(drop=True)
df.to_csv('csv/rundata/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv')
dftest = df.head(1056)
dftrain=df.iloc[1056:]
df=dftrain
df=df.append(dftest)
datasets = []
outputs = []
for x in range(10):
    dftest = df.head(1056)
    #print(dftest)
    dftrain=df.iloc[1056:]
    df=dftrain
    df=df.append(dftest)
    this_dataset = Dataset(dftrain, dftest, x)
    this_output = runmodel(this_dataset, x)
    datasets.append(this_dataset)
    outputs.append(this_output)
#pool = mp.Pool(processes=12)
print(outputs)
print(numpy.mean(numpy.vstack(outputs),axis=0))
