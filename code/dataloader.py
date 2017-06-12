import numpy as np
import pandas as pd
from pandas import ExcelWriter
from itertools import chain
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from bhloader import BHLoader

import metadata

import random

class AmesLoader:
	def __init__(self, filepath):
		self.dataX, self.dataY = self.loadRawData(filepath)

		#print self.dataY

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
				# df[c].fillna(0, inplace=True)
				df[c].fillna(df[c].median(), inplace=True)
			elif c in var_types['discrete']:
				# df[c].fillna(0, inplace=True)
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

		dataX = np.array(df.drop('SalePrice', axis=1))
		dataY = np.array(df['SalePrice'])

		return dataX, dataY

	def getMinMaxData(self, isminibatch=True, mbSize=100):
		trainX, testX, trainY, testY = train_test_split(self.dataX, self.dataY, test_size=0.2, random_state=random.randint(0, 50))

		mm_scaler_x = preprocessing.MinMaxScaler()
		mm_scaler_y = preprocessing.MinMaxScaler()

		mm_scaler_x.fit(trainX)
		mm_scaler_y.fit(trainY)

		trainX = mm_scaler_x.transform(trainX)
		testX = mm_scaler_x.transform(testX)
		trainY = mm_scaler_y.transform(trainY)
		testY = mm_scaler_y.transform(testY)

		if isminibatch == True:
			batches_x = []
			batches_y = []
			for i in range(0, len(trainX[0]), mbSize):
				batches_x.append(trainX[i:i+mbSize])
				batches_y.append(trainY[i:i+mbSize])

			return batches_x, batches_y, testX, testY
		else:
			return trainX, trainY, testX, testY

	def getNormalizedData(self, cross_val=False):
		trainX, testX, trainY, testY = train_test_split(self.dataX, self.dataY, test_size=0.2, random_state=random.randint(0, 50))

		trainX = np.array(trainX)
		trainY = np.array(trainY)
		testX = np.array(testX)
		testY = np.array(testY)

		meanX = np.mean(trainX, axis=0)
		stdX = np.std(trainX, axis=0)

		#meanY = np.mean(trainY)
		#stdY = np.std(trainY)

		eps = 0.0001

		trainX = (trainX - meanX) / (stdX+eps)
		#trainY = (trainY - meanY) / (stdY+eps)

		testX = (testX - meanX) / (stdX+eps)
		#testY = (testY - meanY) / (stdY+eps)

		return trainX, trainY, testX, testY
