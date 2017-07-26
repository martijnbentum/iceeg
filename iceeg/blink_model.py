import tensorflow as tf
import numpy as np
import path
import random


# python version: Python 3.6.0
# tensorflow version: 1.2.1


# extract data that is manually classified -9 no classification -1 classified file
# but last epoch which was skipped due to coding error

#load data
class Blink_model:
	'''Train regression model with gradient descent using the tensorflow package
	on blink epochs. The epochs are pre-selected with peak detection using the
	peakutils package, see blink.py.
	'''

	def __init__(self,info = None,data = None,train_threshold = 29,gradient_descent_opt = 0.01,batch_size = 100,tf_version = '1.2.1'):
		'''Create blink model object
		
		info 					np.ndarray with information about participant 
								experiment and peak samplenumber (with respect 
								to start of block and start of eeg file). Can
								be filename or array
		data 					np.ndarray with blink epoch, blink epoch is 
								normalized between 0 1, can be filename or array
		train_threshold 		participant number cut off for training testing set, 
								default is 2/3 training 1/3 testing
		gradient_descent_opt 	step size for gradient descent algorithm
		batch_size 				number of epoch for one training step in the 
								stochastic training
		tf_version 				tensorflow version that this code was tested with 
								(seems to change a lot)
		'''
		self.info = info
		self.data = data
		self.train_threshold = train_threshold
		self.gradient_descent_opt = gradient_descent_opt
		self.batch_size = batch_size
		self.tf_version = tf_version
		self.load_data()
		self.initialize()

	def load_data(self):
		'''Load data and info, both np.ndarray made with blinks2array.py
		it is possible to supply the array or filename when init model object
		default is to load files from disk.
		'''
		if type(self.info) == np.ndarray and type(self.data) == np.ndarray: return 1
		if type(self.info) == str and type(self.data) == str:
			self.info_f = self.info
			self.data_f = self.info

		self.info_f = path.data + 'blinks_np_array1000_info.npy'
		self.data_f = path.data + 'blinks_np_array1000_data.npy'
		print('loading data')
		self.info = np.load(self.info_f)
		self.data= np.load(self.data_f)


	def initialize(self):
		'''Make all variables for training and testing and set all tensorflow
		variables.
		cinfo 			info for each epoch that is manually classified
		coutput 		epoch of each classified blink
		train_y 		one hot vector (0 index no blink 1 index blink)
		train_x 		epochs of training set
		test_y 			one hot vector (0 index no blink 1 index blink)
		test_x 			epochs of test set
		tensorflow 		settings taken from: 
						https://www.tensorflow.org/get_started/mnist/pros
						(made some adaption before this worked)
		'''
		# 1 codes for blink, 0 no blink
		self.cindex = self.info[:,-1] >= 0 
		self.cinfo = self.info[self.cindex,:]
		self.coutput = self.data[self.cindex,:]

		# correct blinks classifications that are p < 500, 
		# peak should be center of 1000 sample window
		self.smallp_index = self.cinfo[:,-3] < 500
		self.cinfo[self.smallp_index,-1] = 0

		# create 1 hot vector for classes
		self.one_hot_class = np.zeros([len(self.cinfo),2])
		for i,line in enumerate(self.cinfo):
			if line[-1] == 1:
				self.one_hot_class[i] = np.array([0,1])
			elif line[-1] == 0:
				self.one_hot_class[i] = np.array([1,0])
			else:
				raise ValueError(line[-1] , 'should be 1 or 0')

		# split into trainding and test data
		self.train_index = self.cinfo[:,0] < self.train_threshold
		self.test_index = self.cinfo[:,0] >= self.train_threshold

		self.train_y = self.one_hot_class[self.train_index,:]
		self.test_y = self.one_hot_class[self.test_index,:]

		self.train_x = self.coutput[self.train_index,:]
		self.test_x = self.coutput[self.test_index,:]

		# tensorflow variable setup
		self.x = tf.placeholder("float", shape=[None, 1000])
		self.W = tf.Variable(tf.zeros([1000,2]))
		self.b = tf.Variable(tf.zeros([2]))
		self.y = tf.nn.softmax(tf.matmul(self.x,self.W) + self.b)
		self.y_ = tf.placeholder("float", shape=[None, 2])
		self.cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
		self.train_step = tf.train.GradientDescentOptimizer(self.gradient_descent_opt).minimize(self.cross_entropy)

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)


	def train(self,n=1):
		'''Train model with stochastic training, selecting self.batch_size epochs 
		from training set train model and do this n times.
		'''
		print('Stochastic training... random draw of',self.batch_size,'epochs. Do this',n,'times')
		for i in range(n):
			print(i,)
			start = random.randint(0,len(self.train_x)-self.batch_size)
			end = start + self.batch_size 
			print(start,end)
			batch_x, batch_y = self.train_x[start:end,:],self.train_y[start:end,:]
			self.sess.run(self.train_step,feed_dict={self.x:batch_x,self.y_:batch_y})

	def eval(self):
		'''Print accuracy of the model on held out test set.'''
		self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'))
		print('accuracy on test set')
		print(self.accuracy.eval(feed_dict={self.x: self.test_x, self.y_: self.test_y},session=self.sess))


	def prediction_perc(self,data = None):
		'''Return percentage for each class for each epoch.
		default is testing on test set, alternatively data can be provided
		must be ndarray of correct dimensions (nepochs,1000)
		'''
		print('precentage for both classes')
		if type(data) == np.ndarray: self.predict_data = data
		else: self.predict_data = self.test_x
		return self.y.eval(feed_dict={self.x:self.predict_data},session=self.sess)

	def prediction_class(self,data = None):
		'''Return class for each epoch 1=blinks 0=no blinks.
		default is testing on test set, alternatively data can be provided
		must be ndarray of correct dimensions (nepochs,1000)
		'''
		print('return class label, 1=blink, 0=no blink')
		if type(data) == np.ndarray: self.predict_data = data
		else: self.predict_data = self.test_x
		prediction = tf.argmax(self.y,1)
		return prediction.eval(feed_dict={self.x:self.predict_data},session=self.sess)

