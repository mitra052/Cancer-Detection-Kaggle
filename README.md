# Cancer-Detection-Kaggle
Molecular Classification of Cancer by Gene Expression Monitoring-A Kaggle Challenge

**About**

```
This repository contains Training Data and Testing Data for Binary Classification of Cancer into AML(Acute myeloid Leukemia) and  
ALL(Acute Lymphoblastic Leukemia). These datasets contain measurementscorresponding to ALL and AML samples from Bone Marrow and  
Peripheral Blood. Intensity values have been re-scaled such that overall intensities for each chip are equivalent.

```

**Technicalities**
```
A linear SVM classifier with non-separable formulation, has been developed to predict the class AML or ALL based on a set of.  
features. A unique characteristic of this dataset is that the feature space is high dimensional, i.e there are 7129 features,  
whereas the total sample size is just 38 (much less than the dimension of the feature space). It is not uncommon to observe  
this in bio-medical data because DNA sequence is very long and the number of experimental samples is often small as it is  
expensive to conduct experiments. Therefore, its very useful to select the features which are contributing highly to the class  
prediction. However, the standard linear SVM classifier does a poor job at feature selection, although it might give good accuracy  
in predicting the correct class. In this project, the issue of feature or variable selection problem has been addressed by adding  
a regularizer to the objective function to select the features that contribute most to the class prediction. To this end, different  
regularization methods such as Lasso, Ridge and ElasticNethas been implemented on this data set and the codes are programmed in  
python. No python built-in modules such as sciklearn for SVM has been used. The outcomes of the programs are compared with python  
built-in module for linear SVM classification. 

```
**Libraries required**
```
Numpy
Pandas
```
**How to run-Steps**
```
1) Run the 'GeneEspresso.py' file untill line# 43
2) Run the appropriate lines  for L2 regularizer 
      from Switchregularizers import svmL2
      [x, cm ]= svmL2(X_train, y_train, X_test, y_test, L = 0.4, max_count = 18000, stepSize = 0.00001,algorithm='GD')
      np.save('L2_sol',x)

3) Save the variable 'x' as L2.sol.npy using 'np.save' command
4) Run the testData.py file which loads the test data and uses the optimal solution to compute the accuracy

```
