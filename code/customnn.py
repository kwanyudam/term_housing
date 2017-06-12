import numpy as np

def xavier_initialization(input_size, output_size):             #xavier initialization determines initial value from input & output size
    low = -np.sqrt(6.0/(input_size + output_size))
    high = np.sqrt(6.0/(input_size + output_size))
    return np.random.uniform(low=low, high=high, size=(input_size, output_size))


def c_relu(x, deriv=False):
	if deriv:
		return np.ceil(x)
	else:
		return np.clip(x, a_min=0)

class CustomNeuralNetwork:
	def __init__(self, network_arch, learning_rate=0.001, rectifier='relu'):
		self.x = network_arch[0]
		self.y = network_arch[-1]

		self.depth = len(network_arch)
		self.w = [xavier_initialization(network_arch[i], network_arch[i+1]) for i in range(0, self.depth-1)]

		self.lr = learning_rate

	def train(self, train_x, train_y):
		cost=0
		weight_update=[np.full((network_arch[i], network_arch[i+1]), 0) for i in range(0, self.depth-1)]
		for x in train_x:
			#Feed Forward
			layers=[self.x]
			for l in range(0, self.depth-2):
				# Go Through Fully_Connected Network
				layers.append(c_relu(np.matmul(layers[-1], w[l])))

			z = np.matmul(layers[-1], w[l])

			loss = [np.square(train_y - z)]
			cost = np.square(train_y - z)

			#Back Propagation
			l_delta = [loss[0] * c_relu(layers[-1], deriv=True)]

			for l in reversed(range(0,self.depth-1)):
				loss.insert(0, np.matmul(l_delta[0], self.w[l].transpose()))
				l_delta.insert(0, loss[0] * c_relu(layers[l], deriv=True))

			weight_update += layers * l_delta

		#Update weight with regards to learning rate
		self.w += weight_update / len(train_x) * self.lr 

		return cost

	def test(self, test_x):
		z = []
		for x in test_x:
			#Feed Forward
			layers=[self.x]
			for l in range(0, self.depth-2):
				# Go Through Fully_Connected Network
				layers.append(c_relu(np.matmul(layers[-1], w[l])))

			z.append(np.matmul(layers[-1], w[l]))
		return z



'''

import numpy as np
from scipy.special import expit


class dnn:

	def __init__(self):
		self.label = 0
		self.networkSize1 = 128
		self.networkSize2 = 32
		self.w0 = None
		self.w1 = None
		self.w2 = None
		self.lr = 1.0
				
	def setNetworkSize(self, layer1, layer2):
		self.networkSize1 = layer1
		self.networkSize2 = layer2

	def TrainDNN(self, X, y, maxVal, iteration = 10000):
		np.random.seed(1)

		InputDataSize = X.shape[0]
		InputDimension = X.shape[1]
		outputDimension = y.shape[1]
		batchSize = InputDataSize/10
		# randomly initialize our weights with mean 0
		self.w0 = 2*np.random.random((InputDimension,self.networkSize1)) - 1
		self.w1 = 2*np.random.random((self.networkSize1,self.networkSize2)) - 1
		self.w2 = 2*np.random.random((self.networkSize2,outputDimension)) - 1
		

		for j in xrange(iteration):

			#input
			#batch
			for k in range(0,InputDataSize,batchSize):
				miniBatch = X[k:k+batchSize]
				l0 = X
				l1 = nonlin(np.dot(l0,self.w0))
				l2 = nonlin(np.dot(l1,self.w1))
				#output
				l3 = nonlin(np.dot(l2,self.w2))


				# error
				l3_error = y - l3
	

		
				l3_delta = l3_error*nonlin(l3,deriv=True)
		
				l2_error = l3_delta.dot(self.w2.T)
				l2_delta = l2_error*nonlin(l2,deriv=True)

				l1_error = l2_delta.dot(self.w1.T)
				l1_delta = l1_error * nonlin(l1,deriv=True)

				self.w2 += self.lr*l2.T.dot(l3_delta)
				self.w1 += self.lr*l1.T.dot(l2_delta)
				self.w0 += self.lr*l0.T.dot(l1_delta)
				if (j% (iteration/10)) == 0:
					print "Error:" + str(np.mean(np.abs(l3_error/y*100)))
		
		print "train Finished"
		print "Error:" + str(np.mean(np.abs(l3_error/y*100)))
		print "expected Price"
		print (l3*maxVal).T
		print "Price"
		print (y*maxVal).T
		print "Original Error:" + str(np.mean(np.abs(l3_error*maxVal)))        
		#print "Output After Training:"
		#print l3

	def TestDNN(self, X, y, maxVal):
		#input
		l0 = X
		l1 = nonlin(np.dot(l0,self.w0))
		l2 = nonlin(np.dot(l1,self.w1))
		#output
		l3 = nonlin(np.dot(l2,self.w2))


		# error
		l3_error = y - l3
		print "Output After Training:"
		print "Error:" + str(np.mean(np.abs(l3_error/y*100)))
		print "expected Price"
		print l3*maxVal
		print "Price"
		print y*maxVal
		print "Original Error:" + str(np.mean(np.abs(l3_error*maxVal)))
		print "mean price" + str(np.mean(y*maxVal))

			







def nonlin(x,Relu=False, deriv=False):

	#print x


	if(deriv==True):
		#sigmoid deriv
		return x*(1-x)

	if(Relu==True):
		return np.tanh(x)
		#return np.maximum(x,0)

	#return 1/(1+np.exp(-x))
	return expit(x)

'''