import numpy as np
from sklearn.decomposition import PCA
from operator import itemgetter

class PCAKNN:
	def __init__(self, n_comp=10):
		self.pca = PCA(n_comp)
		return

	def fit(self, X, Y):
		self.pca.fit(X)

		self.newX = self.pca.transform(X)
		self.result = Y

		print "PCA Data Loss : ",
		print 100 *(1.0 - np.sum(self.pca.explained_variance_ratio_)) , "%"

		return

	def test(self, Y, resultY, k=5):
		newY = self.pca.transform(Y)

		train_result=[]
		for y, y_ in zip(newY, resultY):

			distList = []
			for res, x in zip(self.result, self.newX):
				distList.append([res, np.linalg.norm(x-y)])

			#rearrange & get k max vals of means of result
			distList= sorted(distList, key=lambda x:x[1])

			meanY=0.0
			for i in range(0, k):
				meanY += distList[i][0]

			meanY= meanY / k

			train_result.append(meanY)

		train_result = np.array(train_result)
		#print train_result, resultY
		print "Error Rate : ", np.mean(np.absolute(train_result-resultY) / resultY) * 100.0, " %"

		#return resultY