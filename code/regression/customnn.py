import numpy as np

def xavier_initialization(input_size, output_size):             #xavier initialization determines initial value from input & output size
	low = -np.sqrt(6.0/(input_size + output_size))
	high = np.sqrt(6.0/(input_size + output_size))
	return np.random.uniform(low=low, high=high, size=(input_size, output_size))


def c_relu(x, deriv=False):
	if deriv:
		return np.ceil(x)
	else:
		return np.clip(x, a_min=0, a_max=None)

class CustomNeuralNetwork:
	def __init__(self, network_arch, learning_rate=0.001, rectifier='relu'):
		self.network_arch = network_arch

		self.depth = len(network_arch)
		self.w = [xavier_initialization(network_arch[i], network_arch[i+1]) for i in range(0, self.depth-1)]

		self.lr = learning_rate

	def init(self):
		self.w = [xavier_initialization(self.network_arch[i], self.network_arch[i+1]) for i in range(0, self.depth-1)]		

	def train(self, train_x, train_y):
		cost=0
		weight_update=[np.full((self.network_arch[i], self.network_arch[i+1]), 0, dtype=float) for i in range(0, self.depth-1)]
		for x, y in zip(train_x, train_y):
			#Feed Forward
			layers=[x]
			for l in range(0, self.depth-2):
				# Go Through Fully_Connected Network
				layers.append(c_relu(np.matmul(layers[-1], self.w[l])))


			z = np.matmul(layers[-1], self.w[-1])
			layers.append(z)

			loss = [y - z]
			cost = y - z

			#Back Propagation
			l_delta = [loss[0] * c_relu(layers[-1], deriv=True)]

			for l in reversed(range(0,self.depth-1)):
				loss.insert(0, np.matmul(l_delta[0], self.w[l].transpose()))
				l_delta.insert(0, loss[0] * c_relu(layers[l], deriv=True))


			for l in range(0, self.depth-1):
				weight_update[l] += l_delta[l+1]

		#Update weight with regards to learning rate

		for l in range(0, self.depth-1):
			self.w[l]+= (weight_update[l] * self.lr / len(train_x))

		return cost

	def test(self, test_x):
		z = []
		for x in test_x:
			layers=[x]
			for l in range(0, self.depth-2):
				# Go Through Fully_Connected Network
				layers.append(c_relu(np.matmul(layers[-1], self.w[l])))

			z.append(np.matmul(layers[-1], self.w[-1]))
		return z