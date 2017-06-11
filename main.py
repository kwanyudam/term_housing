import random
import os.path
import numpy as np

from dataloader import AmesLoader
from nn import NeuralNetwork
from pcaknn import PCAKNN

IS_OVERWRITE = False

TRAINING_DATA_PATH_X = 'data/Preprocessed_X_train.csv'
TRAINING_DATA_PATH_Y = 'data/Preprocessed_Y_train.csv'

CKPT_PATH = 'ckpt/nn.ckpt'
EPOCHS = 100
#EPOCHS 1000 Recommended...
MB_SIZE = 100


def main():
	# Load Ames Housing Data
	#	- Load Data (from Raw or Refined)
	#	- MinMaxScaling for NeuralNetwork
	#	- Normal Distribution for PCA
	#	!!! Cross Validation
	print "Loading Data..."

	loader = AmesLoader(TRAINING_DATA_PATH_X, TRAINING_DATA_PATH_Y)

	# Training part -- NeuralNetwork with TensorFlow
	#	- Settings for NeuralNetwork
	#	- check NeuralNetwork CheckPoint
	#		!!!Need Fixing...(Date, Settings should be updated)
	#	- train Neural Network
	print "Training NN with Tensorflow..."

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
		batches_x, batches_y = loader.getMinMaxData(isminibatch=True,mbSize=MB_SIZE)
		for i in range(0, EPOCHS):
			print "ITER : ", i ,", COST : ", curr_cost
			for tr_x, tr_y in zip(batches_x, batches_y):
				curr_cost = myNN.train(tr_x, tr_y)

		print "Training Complete!"
		myNN.save(CKPT_PATH)
		print "Network saved..."

	# Training part -- PCA + KNN

	myKNN = PCAKNN()

	batches_x, batches_y = loader.getNormalizedData()
	myKNN.fit(batches_x)

	#My Own NN



	# Testing With Cross Validation...


if __name__=="__main__":
	main()