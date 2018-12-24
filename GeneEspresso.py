# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 01:21:39 2018

@author: doddi
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df= pd.read_csv('data_set_ALL_AML_train.csv')
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
df2['cat'] = list(pd.read_csv('actual.csv')[:38]['cancer'])
dic = {'ALL':-1,'AML':1}
df2.replace(dic,inplace=True)
df2.head(3)


X = np.matrix(df2.iloc[:, 0:-1].values, dtype = 'float')
y = np.transpose(np.array(df2.iloc[:, -1].values, dtype = 'float'))

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.02, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)












#L2  Gradient Descent
from Switchregularizers import svmL2
[x, cm ]= svmL2(X_train, y_train, X_test, y_test, L = 0.4, max_count = 18000, stepSize = 0.00001,algorithm='GD')
np.save('L2_sol',x)
#L2  SGD
from Switchregularizers import svmL2
[x, cm ]= svmL2(X_train, y_train, X_test, y_test, L = 0.4, max_count = 5000, stepSize = 0.0002,algorithm='SGD')

#L2  CGD
from Switchregularizers import svmL2
[x, cm ]= svmL2(X_train, y_train, X_test, y_test, L = 0.4, max_count = 100, stepSize = 0.1,algorithm='CGD')





#L1  Gradient Descent
from Switchregularizers import svmL1
[x, cm ]= svmL1(X_train, y_train, X_test, y_test, L = 0.4, max_count = 5000, stepSize = 0.01,algorithm='GD')

#L1  SGD  
from Switchregularizers import svmL1
[x, cm ]= svmL1(X_train, y_train, X_test, y_test, L = 0.4, max_count = 3500, stepSize = 0.002,algorithm='SGD')

#L1  CGD
from Switchregularizers import svmL1
[x, cm ]= svmL1(X_train, y_train, X_test, y_test, L = 0.4, max_count = 100, stepSize = 0.1,algorithm='CGD')



#ElasticNet  Gradient Descent
from Switchregularizers import svm_ElasticNet
[x, cm ]= svm_ElasticNet(X_train, y_train, X_test, y_test, L = 0.4, max_count = 5000, stepSize = 0.01,algorithm='GD')


