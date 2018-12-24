# Cancer-Detection-Kaggle
Cancer Cell Detection using Gene Expression Monitoring- A Kaggle Challenge

First run the 'GeneEspresso.py' file untill line# 43

Then run the appropriate lines  for L2 regularizer
from Switchregularizers import svmL2
[x, cm ]= svmL2(X_train, y_train, X_test, y_test, L = 0.4, max_count = 18000, stepSize = 0.00001,algorithm='GD')
np.save('L2_sol',x)

save the variable 'x' as L2.sol.npy using 'np.save' command

Then run the testData.py file which loads the test data and uses the optimal solution to compute the accuracy
