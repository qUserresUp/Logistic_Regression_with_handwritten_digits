import numpy as np
import sys
from helper import *


def third_order(X):
	"""Third order polynomial transform on features X.

	Args:
		X: An array with shape [n_samples, 2].

	Returns:
		poly: An (numpy) array with shape [n_samples, 10].
	"""
	### YOUR CODE HERE
	#n_samples, n_features = X.shape

	third_X = []
	n_samples, n_features = X.shape
	#print("samples: " , n_samples)
	for i in range(n_samples):
		x1 = X[i][0]
		x2 = X[i][1]
		third_X.append([1, x1, x2, math.pow(x1,2), x1*x2, math.pow(x2,2), math.pow(x1,3), math.pow(x1,2)*x2, x1*math.pow(x2,2), math.pow(x2,3)])


	return third_X

	### END YOUR CODE


class LogisticRegression(object):
	
	def __init__(self, max_iter, learning_rate, third_order=False):
		self.max_iter = max_iter
		self.lr = learning_rate
		self.third_order = third_order


	def _gradient(self, X, y):
		"""Compute the gradient with samples (X, y) and weights self.W.

		Args:
			X: An array with shape [n_samples, n_features].
			   (n_features depends on whether third_order is applied.)
			y: An array with shape [n_samples,]. Only contains 1 or -1.

		Returns:
			gradient: An array with shape [n_features,].
		"""
		### YOUR CODE HERE

		w = self.get_params()
		n_samples, n_features = X.shape
		tot_grad = np.zeros(n_features)
		for i in range(n_samples):
			g_numerator = -y[i]
			wx = w.dot(X[i])
			g_denominator = np.exp(np.multiply(y[i],wx))+1
			curr_grad = np.multiply(g_numerator/g_denominator, X[i])
			tot_grad = np.add(tot_grad, curr_grad)

		return np.divide(tot_grad, n_samples)

		### END YOUR CODE


	def fit(self, X, y):
		"""Train logistic regression model on data (X,y).
		(If third_order is true, do the 3rd order polynomial transform)

		Args:
			X: An array with shape [n_samples, 3].
			y: An array with shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE

		MAX_iter = self.max_iter
		Learning_rate = self.lr

		if(self.third_order == True):
			temp = X[:,1:]
			X = np.array(third_order(temp))	 #third_order() accept input size [sample, 2]

		n_samples, n_features = X.shape
		self.W = np.zeros(n_features)
		#print("n_features: ", n_features)

		for i in range(MAX_iter):
			gt = self._gradient(X, y)
			vt = -gt
			nvt = np.multiply(Learning_rate, vt)
			self.W = self.W + nvt

		#print("fit W: " , self.W)
		#print("fit Y: " , y)

		### END YOUR CODE
		return self


	def get_params(self):
		"""Get parameters for this perceptron model.

		Returns:
			W: An array of shape [n_features,].
			   (n_features depends on whether third_order is applied.)
		"""
		if self.W is None:
			print("Run fit first!")
			sys.exit(-1)
		return self.W


	def predict(self, X):
		"""Predict class labels for samples in X.
		(If third_order is true, do the 3rd order polynomial transform)

		Args:
			X: An array of shape [n_samples, 3].

		Returns:
			preds: An array of shape [n_samples,]. Only contains 1 or -1.
		"""
		### YOUR CODE HERE
		#print("predit W:", self.W)
		#print("predit X:", X)

		if (self.third_order == True):
			temp = X[:, 1:]
			X = np.array(third_order(temp))

		xT = np.matrix.transpose(X)
		wx = np.dot(self.W, xT)
		n_samples, n_features = X.shape
		predict_arr = []
		for i in range(n_samples):
			psb = 1/(1+np.exp(-wx[i]))
			if (psb > 0.5):
				predict_arr.append(1)
			else:
				predict_arr.append(-1)

		return predict_arr


		### END YOUR CODE


	def score(self, X, y):
		"""Returns the mean accuracy on the given test data and labels.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			score: A float. Mean accuracy of self.predict(X) wrt. y.
		"""
		return np.mean(self.predict(X)==y)



def accuracy_logreg(max_iter, learning_rate, third_order, 
					X_train, y_train, X_test, y_test):

	# train perceptron
	model = LogisticRegression(max_iter, learning_rate, third_order)
	model.fit(X_train, y_train)
	train_acc = model.score(X_train, y_train)

	# test perceptron model
	test_acc = model.score(X_test, y_test)

	return train_acc, test_acc