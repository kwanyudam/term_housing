import random
import os.path
import numpy as np

from dataloader import AmesLoader
from nn import NeuralNetwork
from customnn import CustomNeuralNetwork
from pcaknn import PCAKNN

import warnings
warnings.filterwarnings("ignore")

IS_OVERWRITE = True
TEST_EXISTS = False

DATA_PATH = '../../data/ml_project_train.csv'
TEST_PATH = '../../data/ml_project_test.csv'

CKPT_PATH = '../../ckpt/nn.ckpt'
#EPOCHS>1000 Recommended...
TF_EPOCHS = 1000
NN_EPOCHS = 500

MB_SIZE = 100

K_FOLD_SET_SIZE = 11

def main():
	global TEST_EXISTS
	# Load Ames Housing Data
	#	- Load Data (from Raw or Refined)
	#	- MinMaxScaling for NeuralNetwork
	#	- Normal Distribution for PCA
	print "Loading Data..."

	if not os.path.isfile(DATA_PATH):
		print "Error ! Training Data Not Found!"
		return
	
	if os.path.isfile(TEST_PATH):
		print "Test Data Exists!!"
		loader = AmesLoader(DATA_PATH, TEST_PATH, True)
		TEST_EXISTS=True
	else:
		print "No Test Data : Cross Validation"
		loader = AmesLoader(DATA_PATH)


	network_arch=[235, 128, 64, 32, 2]
	learning_rate = 0.1
	rectifier = 'relu'

	myNN = NeuralNetwork(network_arch, 
		learning_rate=learning_rate, 
		rectifier=rectifier)

	myOwnNN = CustomNeuralNetwork(network_arch,
		learning_rate = learning_rate,
		rectifier=rectifier)

	tbSize = loader.getTestBatchSize()

	mlp_err_rate = np.full((tbSize, 1), 0)
	knn_err_rate = np.full((tbSize, 1), 0)
	nn_err_rate = np.full((tbSize, 1), 0)
	#Cross Validation
	for i in range(K_FOLD_SET_SIZE):
		print "==", i+1, "th Set of K-Fold Cross Validation======="
		# Training part -- NeuralNetwork with TensorFlow
		#	- Settings for NeuralNetwork
		#	- check NeuralNetwork CheckPoint
		#	- train Neural Network
		print "\nNN with Tensorflow..."

		batches_x, batches_y, test_x, test_y = loader.getMinMaxData(isminibatch=True,mbSize=MB_SIZE)

		if(not IS_OVERWRITE) and os.path.isfile(CKPT_PATH):
			print "\tCHECK POINT EXISTS!!",
			myNN.load(CKPT_PATH)
			print "----Loaded : ", CKPT_PATH
		else:
			print "\tTraining Network..."
			curr_cost=0
			myNN.init()
			for i in range(0, TF_EPOCHS):
				for tr_x, tr_y in zip(batches_x, batches_y):
					curr_cost = myNN.train(tr_x, tr_y.reshape((len(tr_y), 2)))

			print "\tTraining Complete!"
			myNN.save(CKPT_PATH)
			print "\tNetwork saved..."

		# Testing part -- NeuralNetwork with TensorFlow
		print "\tTesting NN with Tensorflow..."
		result_y = myNN.test(test_x)
		#result_y = loader.restoreMinMaxSalePrice(result_y)
		count =0

		for i in range(0, result_y.shape[0]):
			if result_y[i][0]>result_y[i][1]:
				if test_y[i][0]==1:
					count+=1
			elif test_y[i][1]==1:
				mlp_err_rate[i]+=1
				count+=1

		print "\tError Rate : ", 100.0 * (result_y.shape[0] - count) / result_y.shape[0], " %"
		#mlp_err_rate.append(np.mean(np.absolute(result_y-test_y)/test_y))
		#print "\tError Rate : ", mlp_err_rate[-1]  * 100.0, " %"


		# Training part -- PCA + KNN
		print "\nPCA + KNN"
		myKNN = PCAKNN()

		train_x, train_y, test_x, test_y = loader.getNormalizedData()

		myKNN.fit(train_x, train_y)

		# Testing part -- PCA + KNN
		print "\tTesting PCA + KNN"
		result_y = myKNN.test(test_x)

		count=0

		for i in range(0, result_y.shape[0]):
			if result_y[i][0]>result_y[i][1]:
				if test_y[i]<160000:
					count+=1
			elif test_y[i]>=160000:
				knn_err_rate[i]+=1
				count+=1
		#knn_err_rate.append(np.mean(np.absolute(result_y-test_y) / test_y))
		print "\tError Rate : ", 100.0 * (result_y.shape[0] - count) / result_y.shape[0], " %"

		# Training part -- Custom NN
		print "\nNeural Network - Custom..."

		batches_x, batches_y, test_x, test_y = loader.getMinMaxData(isminibatch=True,mbSize=MB_SIZE)

		print "\tTraining Network..."

		curr_cost=0
		for i in range(0, NN_EPOCHS):
			#print "ITER : ", i ,", COST : ", curr_cost
			for tr_x, tr_y in zip(batches_x, batches_y):
				curr_cost = myOwnNN.train(tr_x, tr_y.reshape((len(tr_y), 2)))

		print "\tTraining Complete!"

		# Testing part -- Custom NN
		print "\tTesting NN - Custom..."
		result_y = myNN.test(test_x)
		#result_y = loader.restoreMinMaxSalePrice(result_y)

		count=0
		for i in range(0, len(result_y)):
			if result_y[i][0]>result_y[i][1]:
				if test_y[i][0]==1:
					count+=1
			elif test_y[i][1]==1:
				nn_err_rate[i]+=1
				count+=1
		print "\tError Rate : ", 100.0 * (len(result_y) - count) / len(result_y), " %"

		print "\n==========================================\n"

	if TEST_EXISTS:
		test_y = loader.getTestY()
		tN = len(test_y)
		count=0
		for i, j in zip(mlp_err_rate, test_y):
			if i<K_FOLD_SET_SIZE/2.0 and j<160000:
				count+=1
			elif i>K_FOLD_SET_SIZE/2.0 and j>=160000:
				count+=1
		print "TensorFlow Neural Network Error Rate : ", 100.0* (tN - count) / tN, " %"

		count=0
		for i, j in zip(knn_err_rate, test_y):
			if i<K_FOLD_SET_SIZE/2.0 and j<160000:
				count+=1
			elif i>K_FOLD_SET_SIZE/2.0 and j>=160000:
				count+=1
		print "PCA + KNN Error Rate : ", 100.0* (tN - count) / tN, " %"

		count=0
		for i, j in zip(nn_err_rate, test_y):
			if i<K_FOLD_SET_SIZE/2.0 and j<160000:
				count+=1
			elif i>K_FOLD_SET_SIZE/2.0 and j>=160000:
				count+=1
		print "Custom Neural Network Error Rate : ", 100.0* (tN - count) / tN, " %"


if __name__=="__main__":
	main()