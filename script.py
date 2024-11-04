import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("parkinsons.csv")
df.head()

x = df.drop(["status","name"],axis=1)
y = df["status"]

scaler=MinMaxScaler((-1,1))
x = scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

model=XGBClassifier()
model.fit(x_train,y_train)

#single prediction
ip = (162.56800,198.34600,77.63000,0.00502,0.00003,0.00280,0.00253,0.00841,0.01791,0.16800,0.00793,0.01057,0.01799,0.02380,0.01170,25.67800,0.427785,0.723797,-6.635729,0.209866,1.957961,0.135242)
#ip = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)
print('Prediction: ',model.predict([ip]))


y_pred=model.predict(x_test)
print('Accuracy: ',accuracy_score(y_test, y_pred)*100)