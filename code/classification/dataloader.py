import numpy as np
import pandas as pd
from pandas import ExcelWriter
from itertools import chain
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import skew

import metadata

import random

class AmesLoader:
	def __init__(self, filepath, testpath=None, isTest=False):
		self.dataX, self.dataY = self.loadRawData(filepath)

		if isTest:
			self.testX, self.testY = self.loadRawData(testpath)

		self.isTest = isTest

		return 

	def loadRawData(self, filepath):
		df = pd.read_csv(filepath)

		var_types = metadata.VAR_TYPES

		# Drop `Id` column.
		df.drop('Id', axis=1, inplace=True)

		for c in df.columns:
			if c == 'SalePrice':
				pass
			elif c in var_types['continuous']:
				df[c].fillna(df[c].median(), inplace=True)
			elif c in var_types['discrete']:
				df[c].fillna(df[c].median(), inplace=True)
			elif c in var_types['nominal']:
				df[c].fillna('None', inplace=True)
				value_set = np.unique(df[c])
				for v in value_set:
					new_c = '{}-{}'.format(c, v)
					df[new_c] = map(int, df[c] == v)
			elif c in var_types['ordinal']:
				df[c].fillna('None', inplace=True)
				if c in metadata.STR_ORDINAL_VALUES:
					df[c] = df[c].apply(lambda x: metadata.STR_ORDINAL_VALUES[c].index(x))

		for v in var_types['nominal']:
			df.drop(v, axis=1, inplace=True)

		dataX = np.array(df.drop('SalePrice', axis=1), dtype=float)
		dataY = np.array(df['SalePrice'], dtype=float)


		colnames= df.columns.values

		return dataX, dataY

	def getTestY(self):
		return self.testY
	def getMinMaxData(self, isminibatch=True, mbSize=100):
		trainX, testX, trainY, testY = train_test_split(self.dataX, self.dataY, test_size=0.2, random_state=random.randint(0, 50))
		if self.isTest:			
			testX = self.testX
			testY = self.testY

		tmpY = []
		for y in trainY:
			if y > 160000:
				tmpY.append([0, 1])
			else:
				tmpY.append([1, 0])
		trainY = np.array(tmpY)
		
		tmpY = []
		for y in testY:
			if y > 160000:
				tmpY.append([0, 1])
			else:
				tmpY.append([1, 0])
		testY = np.array(tmpY)

		self.mm_scaler_x = preprocessing.MinMaxScaler()
		self.mm_scaler_y = preprocessing.MinMaxScaler()

		self.mm_scaler_x.fit(trainX)
		self.mm_scaler_y.fit(trainY)


		trainX = self.mm_scaler_x.transform(trainX)
		testX = self.mm_scaler_x.transform(testX)
		trainY = self.mm_scaler_y.transform(trainY)

		testX = np.clip(testX, 0, 1)

				

		if isminibatch == True:
			batches_x = []
			batches_y = []
			for i in range(0, len(trainX[0]), mbSize):
				batches_x.append(trainX[i:i+mbSize])
				batches_y.append(trainY[i:i+mbSize])

			return batches_x, batches_y, testX, testY
		else:
			return trainX, trainY, testX, testY

	def getTestBatchSize(self):
		if self.isTest:
			return len(self.testY)
		else:
			return (int)(len(self.dataY)*0.2)
	def restoreMinMaxSalePrice(self, price):
		return self.mm_scaler_y.inverse_transform(price)

	def restoreNormalizedPrice(self, price):
		return (price * (self.stdY+self.eps))+self.meanY

	def getNormalizedData(self, cross_val=False, isminibatch=False, mbSize=100, normalizeY=False):
		trainX, testX, trainY, testY = train_test_split(self.dataX, self.dataY, test_size=0.2, random_state=random.randint(0, 50))
		if self.isTest:			
			testX = self.testX
			testY = self.testY

		trainX = np.array(trainX)
		trainY = np.array(trainY)
		testX = np.array(testX)
		testY = np.array(testY)

		self.meanX = np.mean(trainX, axis=0)
		self.stdX = np.std(trainX, axis=0)

		self.eps = 0.0001

		trainX = (trainX - self.meanX) / (self.stdX+self.eps)
		testX = (testX - self.meanX) / (self.stdX+self.eps)

		if normalizeY:
			self.meanY = np.mean(trainY)
			self.stdY = np.std(trainY)
			trainY = (trainY - self.meanY) / (self.stdY+self.eps)
			testY = (testY - self.meanY) / (self.stdY+self.eps)

		if isminibatch == True:
			batches_x = []
			batches_y = []
			for i in range(0, len(trainX[0]), mbSize):
				batches_x.append(trainX[i:i+mbSize])
				batches_y.append(trainY[i:i+mbSize])

			return batches_x, batches_y, testX, testY

		return trainX, trainY, testX, testY
