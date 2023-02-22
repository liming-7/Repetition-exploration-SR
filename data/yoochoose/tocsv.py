import numpy as np
import pandas as pd
import datetime as dt

PATH_TO_ORIGINAL_DATA = '../input/recsys-challenge-2015/'
data_file = "yoochoose-clicks.dat"
data = pd.read_csv(data_file, sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
data.columns = ['user_id', 'time_old', 'item_id']
print(data.head(10))
data['timestamp'] = data.time_old.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #This is not UTC. It does not really matter.
print(data.head(10))
del(data['time_old'])
print(data.head(10))
data.to_csv('yoochoose-all.csv', index=False)