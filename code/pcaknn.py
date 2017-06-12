import numpy as np
from sklearn.decomposition import PCA
from operator import itemgetter

class PCAKNN:
	def __init__(self, n_comp=10):
		self.n_comp = n_comp
		self.pca = PCA(self.n_comp)
		return

	def fit(self, X, Y):
		self.pca.fit(X)

		self.newX = self.pca.transform(X)
		self.result = Y

		print "PCA EigenValues Ratio :",
		print self.pca.explained_variance_ratio_


		#This part is for analyzing vector dependency
		'''attr_len = X.shape[1]
		attr_weight = np.full((attr_len, 1), 0, dtype=float)
		for i in range(self.n_comp):

			curr_comp = np.array(self.pca.components_[i]).reshape((attr_len, 1))

			alist = np.hstack((np.arange(attr_len).reshape((attr_len, 1)), curr_comp))

			alist = sorted(alist, key=lambda x:-abs(x[1]))


			for j in range(attr_len):
				attr_weight[j]+=self.pca.explained_variance_ratio_[i] * abs(alist[j][1])

		attr_weight = np.hstack((np.arange(attr_len).reshape((attr_len, 1)), attr_weight))

		attr_weight = sorted(alist, key=lambda x:-x[1])

		print attr_weight[0][0]
		print attr_weight[1][0]
		print attr_weight[2][0]
		print attr_weight[3][0]
		print attr_weight[4][0]
		print "==========="
		'''
			
		print "PCA Data Loss : ",
		print 100 *(1.0 - np.sum(self.pca.explained_variance_ratio_)) , "%"

		return

	def test(self, Y, k=5):
		newY = self.pca.transform(Y)

		train_result=[]
		for y in newY:

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

		return train_result