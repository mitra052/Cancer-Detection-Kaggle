# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:07:58 2018

@author: doddi
"""
import numpy as np


def svmL2(X_train, y_train, X_test, y_test, L, max_count, stepSize, algorithm):
  
    SS = len(X_train[:,1])
    testSize = len(X_test[:,1])
    X_train = np.append(X_train,np.ones([SS,1]),axis = 1)
    X_test = np.append(X_test,np.ones([testSize,1]),axis = 1)
    y_train[np.where(y_train==0)]=-1
    y_test[np.where(y_test==0)]=-1
    Features = len(X_train[1,:])
    count = 0
    x = np.random.rand(Features)
    
    def fx(A, B, x):
        f = np.sum([max(0,1-b*(np.matmul(a,x))) for b in B for a in A])
        f = f/SS+  L*np.linalg.norm(x,2)*np.linalg.norm(x,2)
        return f    
    if algorithm == 'GD':
    	def grad(A, B, x):
    		gd = np.zeros(Features)
    		for cnt in range(0,SS):
    			if B[cnt]*(np.matmul(A[cnt,:],x))<1:
    				gd = gd -B[cnt]*np.transpose(A[cnt,:]) 
    		gd = gd/SS + 2.0*L*x 
    		return [gd,x]
    	print('GD')   	
    elif algorithm == 'SGD':
    	def grad(A, B, x):
    		for cnt in range(0,SS):
    			gd = np.zeros(Features)
    			if B[cnt]*(np.matmul(A[cnt,:],x))<1:
    				gd = -B[cnt]*np.transpose(A[cnt,:]) 
    			gd = gd + 2.0*L*x
    			x = x - stepSize*gd
    		return [np.zeros(Features),x]
    	print('SGD')    
    elif algorithm == 'CGD':
        def grad(A, B, x):
            gd = np.zeros(Features)
            for f in range(0,Features):
                for cnt in range(0,SS):
                    if B[cnt]*(np.matmul(A[cnt,:],x))<1:
                        gd[f] = gd[f] -B[cnt]*np.transpose(A[cnt,f]) 
                gd[f] = gd[f]/SS + 2.0*L*x[f]
                x[f]= x[f]-stepSize*gd[f]
            return [np.zeros(Features),x]
        print('CGD')   	 
    else:
    	import sys
    	sys.exit('Only GD and SGD is available for this regularizer')  
    print(x)
    while count<=max_count:
#        stepSize = 1.5/(1.5*count+0.5)
        [gd,x]= grad(X_train,y_train,x)
        x = x - stepSize*gd
        print(fx(X_train,y_train,x))
        count = count+1
    holder = np.matmul(X_test,x)
    temp = np.sign(holder)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, temp)
    return [x, cm]


def svmL1(X_train, y_train, X_test, y_test, L, max_count, stepSize, algorithm ):
  
    SS = len(X_train[:,1])
    testSize = len(X_test[:,1])
    X_train = np.append(X_train,np.ones([SS,1]),axis = 1)
    X_test = np.append(X_test,np.ones([testSize,1]),axis = 1)
    y_train[np.where(y_train==0)]=-1
    y_test[np.where(y_test==0)]=-1
    Features = len(X_train[1,:])
    count = 0
    x = np.random.rand(Features)

    def fx(A, B, x):
        f = np.sum([max(0,1-b*(np.matmul(a,x))) for b in B for a in A])
        f = f/SS+  L*np.linalg.norm(x,1)
        return f    

    if algorithm == 'GD':
    	def grad(A, B, x):
    		gd = np.zeros(Features)
    		for cnt in range(0,SS):
    			if B[cnt]*(np.matmul(A[cnt,:],x))<1:
    				gd = gd -B[cnt]*np.transpose(A[cnt,:]) 
    		gd = gd/SS + L*np.sign(x) 
    		return [gd,x]
    	print('GD')
    	
    	
    	
    elif algorithm == 'SGD':
    	def grad(A, B, x):
    		for cnt in range(0,SS):
    			gd = np.zeros(Features)
    			if B[cnt]*(np.matmul(A[cnt,:],x))<1:
    				gd = -B[cnt]*np.transpose(A[cnt,:]) 
    			gd = gd + L*np.sign(x)
    			x = x - stepSize*gd
    		return [np.zeros(Features),x]
    	print('SGD')
    elif algorithm == 'CGD':
        def grad(A, B, x):
            gd = np.zeros(Features)
            for f in range(0,Features):
                for cnt in range(0,SS):
                    if B[cnt]*(np.matmul(A[cnt,:],x))<1:
                        gd[f] = gd[f] -B[cnt]*np.transpose(A[cnt,f]) 
                gd[f] = gd[f]/SS + L*np.sign(x[f]) 
                x[f]= x[f]-stepSize*gd[f]
            return [np.zeros(Features),x]
        print('CGD')   	    	
    else:
    	import sys
    	sys.exit('Only GD and SGD is available for this regularizer')

   
    while count<=max_count:
        #Gradient Descent
        stepSize = 1.5/(1.5*count+0.5)        
        [gd,x]= grad(X_train,y_train,x) 
        x = x - stepSize*gd
        print(fx(X_train,y_train,x))
        count = count+1
    holder = np.matmul(X_test,x)
    temp = np.sign(holder)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, temp)
    return [x, cm]


def svm_ElasticNet(X_train, y_train, X_test, y_test, L, max_count, stepSize, algorithm):
  
    SS = len(X_train[:,1])
    testSize = len(X_test[:,1])
    X_train = np.append(X_train,np.ones([SS,1]),axis = 1)
    X_test = np.append(X_test,np.ones([testSize,1]),axis = 1)
    y_train[np.where(y_train==0)]=-1
    y_test[np.where(y_test==0)]=-1
    Features = len(X_train[1,:])
#    C = 0.00001
    count = 0
    x = np.random.rand(Features)
    def fx(A, B, x):
        f = np.sum([max(0,1-b*(np.matmul(a,x))) for b in B for a in A])
        f = f/SS+  L*np.linalg.norm(x,2)*np.linalg.norm(x,2)+L*np.linalg.norm(x,1)
        return f    
    if algorithm == 'GD':
    	def grad(A, B, x):
    		gd = np.zeros(Features)
    		for cnt in range(0,SS):
    			if B[cnt]*(np.matmul(A[cnt,:],x))<1:
    				gd = gd -B[cnt]*np.transpose(A[cnt,:]) 
    		gd = gd/SS + 2.0*L*x+L*np.sign(x)  
    		return [gd,x]
    	print('GD')   	
    elif algorithm == 'SGD':
    	def grad(A, B, x):
    		for cnt in range(0,SS):
    			gd = np.zeros(Features)
    			if B[cnt]*(np.matmul(A[cnt,:],x))<1:
    				gd = -B[cnt]*np.transpose(A[cnt,:]) 
    			gd = gd + 2.0*L*x+L*np.sign(x)
    			x = x - stepSize*gd
    		return [np.zeros(Features),x]
    	print('SGD')    
    elif algorithm == 'CGD':
        def grad(A, B, x):
            gd = np.zeros(Features)
            for f in range(0,Features):
                for cnt in range(0,SS):
                    if B[cnt]*(np.matmul(A[cnt,:],x))<1:
                        gd[f] = gd[f] -B[cnt]*np.transpose(A[cnt,f]) 
                gd[f] = gd[f]/SS + 2.0*L*x[f]+L*np.sign(x[f]) 
                x[f]= x[f]-stepSize*gd[f]
            return [np.zeros(Features),x]
        print('CGD')   	 
    else:
    	import sys
    	sys.exit('Only GD and SGD is available for this regularizer')  
    while count<=max_count:
        #Gradient Descent
        stepSize = 1.5/(1.5*count+0.5)
        [gd,x]= grad(X_train,y_train,x)
        x = x - stepSize*gd
        print(fx(X_train,y_train,x))
        count = count+1
        
    holder = np.matmul(X_test,x)
    temp = np.sign(holder)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, temp)
    return [x, cm]




    