# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import path
import random
import sklearn.metrics
import sys
import tempfile


import tensorflow as tf


class model_deep_artifact:
	'''Train a convolution neural network on EEG data slices of 1 second (sf=100), labelled for artifacts.'''

	def __init__(self,info = None,data = None,batch_size = 200,nchannels=24,nsamples= 100,test_set_perc= 0.01,tf_version = '1.2.1'):
		'''Create object to train CNN on EEG data slices, for an artifact classifier.

		info 			np matrix with artifact info and pp info, indices correspond with data matrix
		data 			np matrix with eeg data, each rows contains 2400 data point, 24 channels with 100 dp each -> 1second
		batch_size 		number of EEG slices to use per training cycle
		nchannel 		number of channels (24)
		nsamples 		number of samples per channel
		test_set_perc 	number of EEG slices to leave out for testing
		tf_version 		tensorflow version used for training
		'''

		self.fn_info, self.fn_data = 'unk','unk'
		if type(info) == type(None):
			info = path.data + 'info_artifact_training.npy'
		if type(data) == type(None):
			data= path.data + 'data24_artifact_training.npy'
		print(info,data)
		if type(info) == str:
			self.info = np.load(info)
			self.fn_info = info
		if type(data) == str:
			self.data= np.load(data)
			self.fn_data = data
		if type(info) == np.ndarray:
			self.info = info
		if type(data) == np.ndarray:
			self.data= data 
		self.nrows = self.data.shape[0]
		self.batch_size = batch_size
		self.nchannels = nchannels
		self.nsamples = nsamples
		if test_set_perc != None and type(test_set_perc) == float and 0 < test_set_perc < 1:
			self.test_set_perc= test_set_perc
		else: self.test_set_perc = 0.1
		self.tf_version = tf_version
		self.train_accuracy_history, self.test_accuracy_history = [], []
		self.define_network()


	def set_test_train_data(self, test_set_perc = None):
		'''Load info and data matrix and devide into train and test set.'''
		if test_set_perc != None and type(test_set_perc) == float and 0 < test_set_perc < 1: self. test_set_perc = test_set_perc
		self.train_indices = random.sample(range(self.nrows),int(self.nrows * (1 - self.test_set_perc)))
		self.test_indices = list(np.setdiff1d(range(self.nrows),self.train_indices))

		self.train_data = self.data[self.train_indices,:]
		self.train_info = self.info[self.train_indices,:]

		self.test_data = self.data[self.test_indices,:]
		self.test_info= self.info[self.test_indices,:]
 

	def define_network(self):
		'''Declare the tensorflow placeholder variables.
		This defines the graph of the model and creates handles on the model object to interact with tensorflow.
		It is used when starting from scratch and when loading a new model.
		'''
		# Create the model
		self.x = tf.placeholder(tf.float32, [None, self.nchannels * self.nsamples])
		print(self.x)
		# Define loss and optimizer
		self.y_ = tf.placeholder(tf.float32, [None, 2])
		print(self.y_)
		# Build the graph for the deep net
		self.y_conv, self.keep_prob = deepnn(self.x,self.nchannels,self.nsamples)

		with tf.name_scope('loss'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)
																	
		self.cross_entropy = tf.reduce_mean(cross_entropy)

		with tf.name_scope('adam_optimizer'):
			self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

		with tf.name_scope('accuracy'):
			correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
			self.correct_prediction = tf.cast(correct_prediction, tf.float32)
		self.accuracy = tf.reduce_mean(self.correct_prediction)

		self.graph_location = tempfile.mkdtemp()
		print('Saving graph to: %s' % self.graph_location)
		self.train_writer = tf.summary.FileWriter(self.graph_location)
		self.train_writer.add_graph(tf.get_default_graph())


	def initialize(self):
		'''If a model is trained from scratch the variables need to be initialized in a new session.'''
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	def clean_up(self):
		'''Close tensorflow session and clear graph.
		this is necessary to clear the namespace of tensorflow and release resources
		'''
		self.sess.close()
		tf.reset_default_graph()


	def train(self,nepochs = 100):
		'''Train a batch of EEG slices and report accuracy.'''
		for i in range(nepochs):
			print(i,'training')
			batch = next_batch(self.train_data,self.train_info,100)
			if i % 100 == 0:
				nrows = self.test_data.shape[0]
				indices = random.sample(range(nrows),500)
				train_accuracy = self.accuracy.eval(feed_dict={
						self.x: self.test_data[indices,:], self.y_: self.test_info[indices,:], self.keep_prob: 1.0},session = self.sess)
				print('step %d, accuracy on 500 test items:%g' % (i, train_accuracy))
				self.train_accuracy_history.append(train_accuracy)
			self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5},session = self.sess)

		test_accuracy = self.accuracy.eval(feed_dict={ self.x: self.test_data, self.y_: self.test_info, self.keep_prob: 1.0},session = self.sess)
		self.test_accuracy_history.append(test_accuracy)
		print('test accuracy %g' %test_accuracy) 


	def eval(self):
		'''Print precission and recall.'''
		self.ground_truth = np.array([1 if line[0] == 0 else 0 for line in self.test_info])
		temp = np.unique(self.ground_truth, return_counts = True)
		self.n_nonartifacts, self.n_artifacts = temp[1][0], temp[1][1]
		self.predicted = self.compute_prediction_class()
		self.confusion_matrix = sklearn.metrics.confusion_matrix(self.ground_truth,self.predicted)
		self.report = sklearn.metrics.classification_report(self.ground_truth,self.predicted)
		print(self.report)
		

	def compute_prediction_perc(self,data = None):
		'''Return percentage for each class for each epoch.
		default is testing on test set, alternatively data can be provided
		must be ndarray of correct dimensions (nepochs,2400 = nchannelsXnsamples)
		'''
		print('precentage for both classes')
		if type(data) == np.ndarray: self.predict_data = data
		else: self.predict_data = self.test_data
		self.prediction_conv_raw = self.y_conv.eval(feed_dict={self.x:self.predict_data,self.keep_prob:1.0},session=self.sess)
		self.prediction_perc = self.sess.run(tf.nn.softmax(self.prediction_conv_raw))
		return self.prediction_perc


	def compute_prediction_class(self,data = None):
		'''Return class for each epoch 1=artifact 0=artifact.
		default is testing on test set, alternatively data can be provided
		must be ndarray of correct dimensions (nepochs,2400 = nchannelsXnsamples)
		'''
		print('class label, 1=artifact, 0=no artifact')
		if type(data) == np.ndarray: self.predict_data = data
		else: self.predict_data = self.test_data
		temp = self.compute_prediction_perc(self.predict_data)
		self.prediction_class = self.sess.run(tf.argmax(self.prediction_perc,1))
		return self.prediction_class


	def plot_classes(self,data = None):
		'''Plot all artifacts an non blinks in data seperately according to prediction class.
		(NOT TESTED)'''
		if type(data) == np.ndarray: self.predict_data = data
		else: self.predict_data = self.test_x
		prediction = self.prediction_class(self.predict_data)
		blinks = prediction == 1
		non_blinks = prediction == 0
		plt.figure()
		plt.plot(self.predict_data[blinks].transpose(),color= 'black',alpha = 0.01, linewidth=0.5)
		plt.title('artifacts')
		plt.figure()
		plt.plot(self.predict_data[non_blinks].transpose(),color= 'black',alpha = 0.01, linewidth=0.5)
		plt.title('Not artifacts')


def load_model(model_name):
	'''Load a previously trained model by name and return model object.
	'''
	ma = model_deep_artifact()
	ma.set_test_train_data()
	ma.sess = tf.Session()
	ma.saver = tf.train.Saver()
	ma.saver.restore(ma.sess,model_name)
	# ma.initialize()
	return ma

def predict2onehot(pred):
	d = np.zeros((pred.shape[0],2))
	print(d.shape)
	d[np.where(pred == 0)[0],:] = (1,0)
	d[np.where(pred == 1)[0],:] = (0,1)
	return d

def data2prediction(filename,model_object):
	d = np.load(filename)
	print('Loaded data with dimensions:',d.shape, 'and filename:',filename)
	pred = model_object.compute_prediction_class(data= d)
	output = predict2onehot(pred)
	clean, artifact = np.sum(output,axis=0)
	print('Clean segments:',clean,'Artifacts:',artifact)
	return output


def data2prediction_perc(filename,model_object):
	d = np.load(filename)
	print('Loaded data with dimensions:',d.shape, 'and filename:',filename)
	pred = model_object.compute_prediction_perc(data= d)
	return pred
	

	
				

# Helper Functions, taken from the mnist tutorial for CNN: https://www.tensorflow.org/get_started/mnist/pros


def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
												strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape,name = None):
	"""weight_variable generates a weight variable of a given shape."""
	print(shape)
	initial = tf.truncated_normal(shape, stddev=0.1)
	if type(name) != str: temp = tf.Variable(initial)
	else: temp = tf.Variable(initial,name = name)
	return temp


def bias_variable(shape,name = None):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	if type(name) != str: temp = tf.Variable(initial)
	else: temp = tf.Variable(initial,name = name)
	return temp

def next_batch(data,info, batch_size):
	nrows = data.shape[0]
	indices = random.sample(range(nrows),batch_size)
	return data[indices,:],info[indices,:]


def deepnn(x,nchannels,nsamples,fmap_size = 1248):
	"""deepnn builds the graph for a deep net for classifying digits.
	Args:
		x: an input tensor with the dimensions (N_examples, 784), where 784 is the
		number of pixels in a standard MNIST image.
	Returns:
		A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
		equal to the logits of classifying the digit into one of 10 classes (the
		digits 0-9). keep_prob is a scalar placeholder for the probability of
		dropout.
	"""
	# Reshape to use within a convolutional neural net.
	# Last dimension is for "features" - there is only one here, since images are
	# grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
	with tf.name_scope('reshape'):
		print(x)
		# x_image = tf.reshape(x, [-1, 24, 100, 1])
		x_image = tf.reshape(x, [-1, nchannels, nsamples, 1])
		print('reshaped')
		print(x_image)

	# First convolutional layer - maps one grayscale image to 32 feature maps.
	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		print(W_conv1,'W_conv1')
		print(b_conv1,'b_conv1')
		print(h_conv1,'h_conv1')

	# Pooling layer - downsamples by 2X.
	with tf.name_scope('pool1'):
		h_pool1 = max_pool_2x2(h_conv1)

	# Second convolutional layer -- maps 32 feature maps to 64.
	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		print(W_conv2,'W_conv2')
		print(b_conv2,'b_conv2')
		print(h_conv2,'h_conv2')

	# Second pooling layer.
	with tf.name_scope('pool2'):
		h_pool2 = max_pool_2x2(h_conv2)
		print(h_pool2,'h_pool2')

	with tf.name_scope('conv3'):
		W_conv3 = weight_variable([5, 5, 64, 128])
		b_conv3 = bias_variable([128])
		h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
		print(W_conv3,'W_conv3')
		print(b_conv3,'b_conv3')
		print(h_conv3,'h_conv3')
	
	with tf.name_scope('pool3'):
		h_pool3 = max_pool_2x2(h_conv3)
		print(h_pool3,'h_pool3')

	# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
	# is down to 7x7x64 feature maps -- maps this to 1024 features.
	with tf.name_scope('fc1'):
		# W_fc1 = weight_variable([int(nchannels / 4) * int(nsamples / 4)* 128, fmap_size])
		W_fc1 = weight_variable([3  * 13* 128, fmap_size])
		# W_fc1 = weight_variable([6  * 25* 128, fmap_size])
		b_fc1 = bias_variable([fmap_size])
		print(W_fc1,'W_fc1')
		print(b_fc1,'b_fc1')

		# h_pool2_flat = tf.reshape(h_pool2, [-1, int(nchannels/4)*int(nsamples/4)*64],name = 'h_pool2_flat-bla')
		# h_conv3_flat = tf.reshape(h_conv3, [-1, int(nchannels/4)*int(nsamples/4)*128]
		h_pool3_flat = tf.reshape(h_pool3, [-1, 3*13*128])
		# h_conv3_flat = tf.reshape(h_conv3, [-1, 6*25*128])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1,name='h_fc1-bla')
		print(h_pool3_flat,'h_pool2_flat')
		print(h_fc1,'h_fc1')

	# Dropout - controls the complexity of the model, prevents co-adaptation of
	# features.
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32,name = 'keep_prob-bla')
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name = 'h_fc1_drop-bla')

	# Map the final features to 2 classes, one for each digit
	with tf.name_scope('fc2'):
		W_fc2 = weight_variable([fmap_size, 2],name='W_fc2-bla')
		b_fc2 = bias_variable([2],name='b_fc2-bla')
		print(W_fc2,'W_fc2')
		print(b_fc2,'b_fc2')

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	print(y_conv,'y_conv')
	return y_conv, keep_prob

