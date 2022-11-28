This file is a simple implementation of Stochastic Gradient Descent (SGD) on the SKlearn Breast Cancer dataset.
The format of this learning problem is binary classification using the logistic loss. SGD will minimize the logistic loss
to varying degrees across different size training sets. We compare the loss and error on each size of training set and
output a graph showing the differences. For a training set size of n=350 we get a bit over 90% accuracy on the dataset.

Stochastic Gradient Descent is a rather amazing technique that can minimuze the population risk rather than utilize empirical 
risk minimization. This algorithm is such a case in which it minimizes true risk. The various statistics evaluated in this 
algorithm are the excess risk, error rate, and both those prior two's standard deviations across multiple trials. Excess risk 
is calculated by finding the difference between the expected true risk and minimum possible risk for any parameter in the paramete 
space. This can be estimated empirically, which is done using the following estimates. We estimate the excess risk and error by 
running 30 trials for each setting of the training set size. Then, those 50 trials are averaged to give specific data points 
that are then plotted over the training set size. Excess risk in particular is estimated by taking the difference between the average 
of the 30 risks and the minimum of the 50, while our error estimate is simply the average of the errors of the 50 trials. As well, we 
show the standard deviation of the 50 excess risks and 50 errors.
