import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets as ds
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

'''This file is a simple implementation of Stochastic Gradient Descent (SGD) on the SKlearn Breast Cancer dataset.
The format of this learning problem is binary classification using the logistic loss. SGD will minimize the logistic loss
to varying degrees across different size training sets. We compare the loss and error on each size of training set and
output a graph showing the differences. For a training set size of n=350 we get a bit over 90% accuracy on the dataset.
'''
	
def inner_prod(w, x):  # calculate the inner product times label using the w parameter vector and an extended data point from a given point x
	
	vals = np.array([w[i] * x[0][i] for i in range(len(x[0]))]) # pre-calculate values used in inner product
	
	prod = (x[1]) * (np.sum(vals) + w[len(x[0])]) # inner product plus bias times label
	
	return prod
	
def grad(w, x): # function that calculates gradients of logistic loss

	length = len(x[0])
	
	gradient = np.array([(-x[1]) * x[0][i] * (np.exp(-inner_prod(w,x)) / 
	(1 + np.exp(-inner_prod(w,x)))) for i in range(length)]) # partial derivative at each w component that isnt a bias term

	gradient = np.append(gradient, (-x[1]) * (np.exp(-inner_prod(w,x)) / 
	(1 + np.exp(-inner_prod(w,x))))) # partial derivative for the bias term

	return gradient
	
def error(w, S): # function that calculates binary classification error
	
	err = 0
	
	sign_prod = np.array([inner_prod(w, S[i]) 
	for i in range(len(S))]) # calculate the label times the inner product for classification
		
	err = np.where(sign_prod 
	< 0, 1, 0) # if the inner product times label is less than 0 encode its entry as 1 for an error, 0 otherwise
	
	err = np.sum(err) / len(S) # normalize the error
	
	return err
	
def loss(w, S): # function that calculates logistic loss on the test set
	
	losses = np.array([np.log(1 + np.exp(-inner_prod(w, S[i]))) for i in range(len(S))]) # loss calculations
		
	mean_loss = np.sum(losses) / len(S)
	
	return mean_loss
	
def dev(mean, S): # function that calculates a finite set empirical standard deviation
	
	square_dif = np.array([(S[i]-mean) * (S[i]-mean) for i in range(len(S))]) # calculate the quantity (value-mean)^2
	
	variance = np.sum(square_dif) / len(square_dif) # normalize to variance
	
	deviation = np.sqrt(variance) # take square root for std
	
	return deviation
		
def SGD(n, data): # stochastic gradient descent algorithm that uses a data oracle

	T = n + 1
	paras = np.empty(shape=(T, len(data[0][0]) + 1)) # initialize parameter set

	w = np.zeros(len(data[0][0]) + 1) # initialize as the all zeros vector
	
	paras[0] = w

	for t in range(n):
	
		pick = np.random.choice(np.array(range(300))) # new point chosen at random
		point = data[pick]
		
		w = w - grad(w, point) # gradient descent with fixed learning rate of 1
		
		paras[t + 1] = w # save parameter

	w_hat = np.sum(paras, axis=0) / T # averaged parameter
	
	return w_hat
	
n = [50, 100, 200, 350] # different sizes of training sets, which translates to n iterations of SGD

cache = np.empty(shape=(4,4)) # where we will store various statistics

points = ds.load_breast_cancer() # import breast cancer dataset from sklearn

X = points.data
Y = points.target

clf = ExtraTreesClassifier(n_estimators=500) # here we find the best features of the data via a tree classifier impurity test
clf = clf.fit(X, Y)
model = SelectFromModel(clf, prefit=True)
X = model.transform(X)

max_norm = np.max(np.array([np.linalg.norm(X[i]) for i in range(len(X))])) # get max norm of all data to normalize

Y = np.array([int(1) if x==1 else int(-1) for x in Y]) # re-encode classes to 1, -1

data = np.array([(X[i] / max_norm, Y[i]) for i in range(300)], dtype=object) # put data together into (features, label) form with normalized features

test_set = np.array([data[i] for i in range(len(data)-200, len(data))], dtype=object) # generate test set, last 200 data points

for l in range(len(n)):

	weights = np.array([SGD(n[l], data) for i in range(30)]) # generate 30 mean parameters via SGD
	losses = np.array([loss(weights[i], test_set) for i in range(len(weights))]) # generate 30 empirical losses on the test set
	errors = np.array([error(weights[i], test_set) for i in range(len(weights))]) # generate 30 empirical binary error values on the test set

	min_loss = np.min(losses) # find minimum loss

	mean_loss = np.sum(losses) / len(losses)
	mean_error = np.sum(errors) / len(errors)
	dif_loss = mean_loss - min_loss # difference between mean and min losses, aka estimated excess loss
	
	dev_loss = dev(mean_loss, losses)
	dev_error = dev(mean_error, errors)
	
	cache[l] = np.array([dif_loss, mean_error, dev_loss, dev_error]) # estimated excess loss, mean loss, deviation of risks, deviation of errors
	print("estimated excess loss for n="+str(n[l])+":",dif_loss)
	print("estimated mean loss for n="+str(n[l])+":",mean_loss)
	print("estimated min loss for n="+str(n[l])+":",min_loss)
	print("std of estimated loss for n="+str(n[l])+":",dev_loss)
	print("mean error for n="+str(n[l])+":",mean_error)
	print("std of error for n="+str(n[l])+":",dev_error,"\n")

names = ['50', '100', '200', '350'] # x-axis labels showing number of iterations to obtain averaged parameter

plt.subplot(1, 2, 1)

plt.plot(names, cache[0:4, 0], 'rs', label='std=.2')

plt.errorbar(names, cache[0:4, 0], yerr=cache[0:4, 2], capsize=5, ecolor='r', color='r', label='std=.2')


plt.title('Expected loss')
plt.ylabel('estimated loss')
plt.xlabel('number of iterations of SGD')

plt.axis([-1/2, 3.5, -1, 1])

plt.legend()

plt.subplot(1, 2, 2)

plt.plot(names, cache[0:4, 1], 'rs', label='std=.2')

plt.errorbar(names, cache[0:4, 1], yerr=cache[0:4, 3], capsize=5, ecolor='r', color='r', label='std=.2')


plt.title('Expected error')
plt.ylabel('estimated error')
plt.xlabel('number of iterations of SGD')

plt.axis([-1/2, 3.5, -1, 1])

plt.legend()

plt.tight_layout()

plt.show()