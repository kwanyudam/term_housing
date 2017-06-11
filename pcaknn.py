import numpy as np
from sklearn.decomposition import PCA

class PCAKNN:
	def __init__(self, n_comp=5):
		self.pca = PCA(n_comp)
		return

	def fit(self, X, Y):
		self.pca.fit(X)
		#print "Variance : "
		#print self.pca.explained_variance_ratio_
		self.newX = self.pca.transform(X)
		self.result = Y

		print self.newX
		return

	def test(self, Y, k=4):
		newY = self.pca.transform(Y)
		resultY=[]
		for y in newY:
			for x in self.newX:
				#create array with index & dist value

			#rearrange & get k max vals of means of result

			#append to resultY
			resultY.append(meanY)

		return resultY

		

		