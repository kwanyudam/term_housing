import random
import os.path
import numpy as np

from dataloader import AmesLoader
from nn import NeuralNetwork
from pcaknn import PCAKNN

IS_OVERWRITE = False

#TRAINING_DATA_PATH_X = 'data/Preprocessed_X_train.csv'
#TRAINING_DATA_PATH_Y = 'data/Preprocessed_Y_train.csv'

DATA_PATH = '../data/ml_project_train.csv'

CKPT_PATH = '../ckpt/nn.ckpt'
EPOCHS = 1000
#EPOCHS 1000 Recommended...
MB_SIZE = 100

K_FOLD_SET_SIZE = 10

def main():
	# Load Ames Housing Data
	#	- Load Data (from Raw or Refined)
	#	- MinMaxScaling for NeuralNetwork
	#	- Normal Distribution for PCA
	print "Loading Data..."

	loader = AmesLoader(DATA_PATH)

	network_arch=[235, 128, 64, 32, 1]
	dropout_keep_prob = 0.9
	learning_rate = 0.1
	rectifier = 'relu'

	myNN = NeuralNetwork(network_arch, 
		drop_keep=dropout_keep_prob,
		learning_rate=learning_rate, 
		rectifier=rectifier)

	myOwnNN = CustomNeuralNetwork(network_arch,
		learning_rate = learning_rate,
		rectifier=rectifier)

	mlp_err_rate = []
	knn_err_rate = []
	for i in range(K_FOLD_SET_SIZE):
		print i, "th Set of K-Fold Cross Validation\n\n"
		# Training part -- NeuralNetwork with TensorFlow
		#	- Settings for NeuralNetwork
		#	- check NeuralNetwork CheckPoint
		#		!!!Need Fixing...(Date, Settings should be updated)
		#	- train Neural Network
		print "Training NN with Tensorflow..."

		batches_x, batches_y, test_x, test_y = loader.getMinMaxData(isminibatch=True,mbSize=MB_SIZE)

		if(not IS_OVERWRITE) and os.path.isfile(CKPT_PATH):
			print "CHECK POINT EXISTS!!",
			myNN.load(CKPT_PATH)
			print "----Loaded : ", CKPT_PATH
		else:
			print "Training Network..."
			curr_cost=0
			for i in range(0, EPOCHS):
				print "ITER : ", i ,", COST : ", curr_cost
				for tr_x, tr_y in zip(batches_x, batches_y):
					curr_cost = myNN.train(tr_x, tr_y.reshape((len(tr_y), 1)))

			print "Training Complete!"
			myNN.save(CKPT_PATH)
			print "Network saved..."

		# Testing part -- NeuralNetwork with TensorFlow
		print "Testing NN with Tensorflow..."
		result_y = myNN.test(test_x)
		result_y = loader.restoreMinMaxSalePrice(result_y)
		mlp_err_rate.append(np.mean(np.absolute(result_y-test_y)/test_y))
		print "Error Rate : ", mlp_err_rate[-1]  * 100.0, " %"


		# Training part -- PCA + KNN
		print "Training PCA + KNN"
		myKNN = PCAKNN()

		train_x, train_y, test_x, test_y = loader.getNormalizedData()

		myKNN.fit(train_x, train_y)

		# Testing part -- PCA + KNN
		print "Testing PCA + KNN"
		result_y = myKNN.test(test_x)
		knn_err_rate.append(np.mean(np.absolute(result_y-test_y) / test_y))
		print "Error Rate : ", knn_err_rate[-1]*100.0, " %"

		print "\n\n"

		#My Own NN


		#batches_x, batches_y, test_x, test_y = loader.getMinMaxData(isminibatch=True,mbSize=MB_SIZE)

	print "Neural Network Error Rate : ", np.mean(mlp_err_rate) * 100.0 , " %"
	print "PCA + KNN Error Rate : ", np.mean(knn_err_rate) * 100.0, " %"


if __name__=="__main__":
	main()