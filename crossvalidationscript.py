import pandas as pd
import datetime
import numpy
import multiprocessing as mp
from sklearn.utils import shuffle



class Dataset():
    def __init__(self, train, test, number):
        self.train = train
        self.test = test
        self.number = number
def runmodel(dataset):
    #SAXON PUT CODE HERE
    return dataset.number#return accuracy
df = pd.read_csv('csv/labeled_10k.csv', header=0)
df = shuffle(df)
df = df.reset_index(drop=True)
df.to_csv('csv/rundata/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv')
dftest = df.head(1056)
dftrain=df.iloc[1056:]
df=dftrain
df=df.append(dftest)
datasets = []
for x in range(10):
    dftest = df.head(1056)
    print(dftest)
    dftrain=df.iloc[1056:]
    df=dftrain
    df=df.append(dftest)
    datasets.append(Dataset(dftrain, dftest, x))
pool = mp.Pool(processes=12)
results = [pool.apply_async(runmodel, args=(x,)) for x in datasets]
output = [p.get() for p in results]
print(output)
print(numpy.mean(output))
