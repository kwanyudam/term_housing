import numpy as np
from sklearn.decomposition import PCA

class myPCA:
	def __init__(self):
		return

	def fit(self, X, Y):
		pca = new PCA()
		pca.fit(X)

		newY = pca.transform(Y)
		return newY

	def test(self, X):
		for x in X:
			ans = R * x * x.transpose()

		