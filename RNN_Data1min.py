from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
import pandas as pd
#from keras.optimizers import SGD
import csv
from numpy import *
import numpy as np


input = []
output = []
readin=[]
i=0

with open('/Users/yuwang/Downloads/cubist_project/data.1_min.csv','r') as f:
    f.readline()
    for line in f.readlines():
        readin.append(float(line.strip().split(',')[1]))

        i=i+1
size=65535
trainSize=int(0.8*size)
testSize=size-trainSize
#print (readin)
input=readin[0:size]
inarray= np.array(input).reshape(size,1)


docX, docY = [], []

#adjust memory size here
memorySize=20

for i in range(len(inarray) - memorySize):
    docX.append(np.asmatrix(inarray[i:i + memorySize]))
    docY.append(inarray[i + memorySize])
inarray = np.array(docX)
outarray = np.array(docY)


train=inarray[:trainSize]
trainTargets=outarray[:trainSize]

test=inarray[trainSize:]
testTargets=outarray[trainSize:]

print (testTargets.shape)
#print (testTargets)

batch_size = 32

#start building models
model = Sequential()
model.add(SimpleRNN(input_dim=1, output_dim=32,return_sequences=False))
model.add(Dense(32))

model.add(Activation("linear"))
model.add(Dropout(0.3))
model.add(Dense(32))
model.add(Activation("linear"))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation("linear"))

#allow to using difference optimizer here
""" optimizer 1: model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="binary")"""

# optimizer 2:sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#optimizer 3:
model.compile(loss="mse", optimizer="rmsprop")

model.fit(train, trainTargets, batch_size=32, nb_epoch=30, validation_split=0.1)

#valid_preds = model.predict_proba(test, verbose=0)
valid_preds = model.predict(test)
#print (valid_preds.shape)
print(valid_preds)
print(valid_preds.shape)


rmse = np.sqrt(((valid_preds - testTargets) ** 2).mean(axis=0))#root mean square error
stderror2=100*(1-np.absolute(np.sum((valid_preds - testTargets)/testTargets)/13097))
print('the accuracy percentage is:', stderror2,'%')
pd.DataFrame(valid_preds[:100]).plot()
pd.DataFrame(testTargets[:100]).plot()


