import random
import os.path
import numpy as np

from dataloader import AmesLoader
from nn import NeuralNetwork
from pca import PCA

IS_OVERWRITE = False

TRAINING_DATA_PATH_X = 'data/Preprocessed_X_train.csv'
TRAINING_DATA_PATH_Y = 'data/Preprocessed_Y_train.csv'
CKPT_PATH = 'ckpt/nn.ckpt'
EPOCHS = 500
MB_SIZE = 100


def main():
	#Load Ames Housing Data
	#	- Divide Data Set for Cross Validation
	print "Loading Data..."

	loader = AmesLoader()
	loader.loadRefinedData(TRAINING_DATA_PATH_X, TRAINING_DATA_PATH_Y)

	# Training part -- PCA || NeuralNetwork
	print "Training..."
	network_arch=[233, 128, 64, 32, 1]
	dropout_keep_prob = 0.9
	learning_rate = 0.1
	rectifier = 'relu'
	myNN = NeuralNetwork(network_arch, 
		drop_keep=dropout_keep_prob,
		learning_rate=learning_rate, 
		rectifier=rectifier)

	if(not IS_OVERWRITE) and os.path.isfile(CKPT_PATH):
		print "CHECK POINT EXISTS!!",
		myNN.load(CKPT_PATH)
		print "----Loaded : ", CKPT_PATH
	else:
		print "Training Network..."
		curr_cost=0
		for i in range(0, EPOCHS):
			print "ITER : ", i ,", COST : ", curr_cost
			for j in range(0, loader.getSize(), MB_SIZE):
				tr_x, tr_y = loader.minibatches(j, MB_SIZE)
				#tr_y = np.ndarray(shape=(MB_SIZE, 1), dtype=float, buffer = tr_y['0'])
				tr_y = np.array(tr_y[1])
				tr_y = tr_y.reshape((len(tr_y), 1))
				curr_cost = myNN.train(tr_x, tr_y)

		print "Training Complete!"
		myNN.save(CKPT_PATH)
		print "Network saved..."

	# Cross Validation...

	test_x, test_y = loader.testbatch()
	test_y = np.array(test_y[1])
	test_y = test_y.reshape((len(test_y), 1))
	myNN.test(test_x, test_y)


	# Testing...



if __name__=="__main__":
	main()