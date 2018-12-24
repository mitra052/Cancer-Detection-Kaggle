# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:35:45 2018

@author: doddi
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df= pd.read_csv('data_set_ALL_AML_independent.csv')
df.head()
df1 = [col for col in df.columns if "call" not in col]
df = df[df1]
df.head()
df.T.head()
df = df.T
df2 = df.drop(['Gene Description','Gene Accession Number'],axis=0)
df2.index = pd.to_numeric(df2.index)
df2.sort_index(inplace=True)
df2.head()
df2['cat'] = list(pd.read_csv('actual.csv')[38:73]['cancer'])
dic = {'ALL':-1,'AML':1}
df2.replace(dic,inplace=True)
df2.head(3)


X = np.matrix(df2.iloc[:, 0:-1].values, dtype = 'float')
y = np.transpose(np.array(df2.iloc[:, -1].values, dtype = 'float'))

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X)
testSize = len(X_test[:,1])
X_test = np.append(X_test,np.ones([testSize,1]),axis = 1)


x = np.load('EN_sol.npy')
holder = np.matmul(X_test,x)
temp = np.sign(holder)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, temp)
