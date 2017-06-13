import tensorflow as tf
import numpy as np

def xavier_initialization(input_size, output_size):             #xavier initialization determines initial value from input & output size
    low = -np.sqrt(6.0/(input_size + output_size))
    high = np.sqrt(6.0/(input_size + output_size))
    return tf.random_uniform((input_size, output_size),
        minval=low, maxval=high, dtype=tf.float32)

class NeuralNetwork:
	def __init__(self, network_arch, drop_keep=0.1, learning_rate=1.0e-8, decay_rate=0.99, rectifier='tanh', optimizer='gradient'):
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

		self.input_shape = network_arch[0]

		self.x = tf.placeholder(tf.float32, [None, network_arch[0]])
		self.y = tf.placeholder(tf.float32, [None, network_arch[-1]])

		#layers

		self.w = [tf.Variable(xavier_initialization(network_arch[i], network_arch[i+1]), dtype=tf.float32) for i in range(0, len(network_arch)-1)]
		self.b = [tf.Variable(tf.zeros([network_arch[i]], dtype=tf.float32)) for i in range(0,len(network_arch))]

		self.layers = [self.x]
		for i in range(1, len(network_arch)-1):
			if rectifier=='tanh':
				self.layers.append(tf.nn.tanh(tf.matmul(self.layers[i-1], self.w[i-1])+ self.b[i]))
			elif rectifier=='softmax':
				self.layers.append(tf.nn.softplus(tf.matmul(self.layers[i-1], self.w[i-1])+ self.b[i]))
			elif rectifier=='relu':
				self.layers.append(tf.nn.relu(tf.matmul(self.layers[i-1], self.w[i-1])+ self.b[i]))
				#Dropout
			else:
				raise ValueError(rectifier)

		self.z = tf.matmul(self.layers[-1], self.w[-1]) + self.b[-1]

		self.loss = tf.square(self.y-self.z)

		self.cost = tf.reduce_mean(self.loss)

		self.global_step = tf.Variable(0, trainable=False)
		self.lr = tf.train.exponential_decay(learning_rate, self.global_step, 1000, decay_rate, staircase=True)
		if optimizer=='gradient':
			self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cost, global_step=self.global_step)
		elif optimizer=='adam':
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost, global_step=self.global_step)

		self.saver = tf.train.Saver()
	def init(self):
		init = tf.initialize_all_variables()
		self.sess.run(init)

	def train(self, train_x, train_y):
		z, loss, opt, cost = self.sess.run((self.z, self.loss, self.optimizer, self.cost),
			feed_dict={
			self.x:train_x,
			self.y:train_y
			})

		#print "Y : ", train_y
		#print "Z : ", z

		#print "Loss : ", loss
		#print "Cost : ", cost
		return cost

	def test(self, test_x):
		train_result = self.sess.run((self.z),
			feed_dict={
			self.x:test_x
			})
		#print "NN Sales Price Estimation : ", train_result
		#print "Actual Price : ", test_y

		return train_result

	def load(self, ckptname):
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, ckptname)
		pass

	def save(self, ckptname):
		self.saver.save(self.sess, ckptname)
		pass