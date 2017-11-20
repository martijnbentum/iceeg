# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import random
import sys
import tempfile


import tensorflow as tf


class model_deep_artifact:
	'''Train a convolution neural network on EEG data slices of 1 second (sf=100), labelled for artifacts.'''

	def __init__(self,info = None,data = None,batch_size = 200,nchannels=24,nsamples= 100,test_set_perc= 0.1,tf_version = '1.2.1'):
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
		if type(info) == None:
			info = path.data + 'info_artifact_training.npy'
		if type(data) == None:
			data= path.data + 'data24_artifact_training.npy'
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


	def set_test_train_data(self, test_set_perc = None)
		'''Load info and data matrix and devide into train and test set.'''
		if test_set_perc != None and type(test_set_perc) == float and 0 < test_set_perc < 1:
		self.train_indices = random.sample(range(self.nrows),int(self.nrows * (1 - self.test_set_perc)))
		self.test_indices = list(np.setdiff1d(range(data.shape[0]),train_indices))

		self.train_data = data[self.train_indices,:]
		self.train_info = info[self.train_indices,:]

		self.test_data = data[self.test_indices,:]
		self.test_info= info[self.test_indices,:]
 
	
	def initialize(self):
		'''Initialize the tensorflow placeholder variables and start a session.'''
		# Create the model
		self.x = tf.placeholder(tf.float32, [None, self.nsamples])
		print(x)

		# Define loss and optimizer
		self.y_ = tf.placeholder(tf.float32, [None, 2])
		print(y_)

		# Build the graph for the deep net
		self.y_conv, self.keep_prob = deepnn(x,self.nchannels,self.nsamples)

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
		self.train_writer = tf.summary.FileWriter(graph_location)
		self.train_writer.add_graph(tf.get_default_graph())

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	def train(self,nepochs = 100):
		'''Train a batch of EEG slices and report accuracy.'''

		for i in range(nepochs):
			print(i,'training')
			batch = next_batch(self.train_data,self.train_info,100)
			if i % 100 == 0:
				train_accuracy = self.accuracy.eval(feed_dict={
						self.x: test_data, self.y_: self.test_info, self.keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
				self.train_accuracy_history.append(train_accuracy)
			self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

		test_accuracy = self.accuracy.eval(feed_dict={ x: test_data, y_: test_info, keep_prob: 1.0}))
		self.test_accuracy_history.append(test_accuracy)
		print('test accuracy %g' %test_accuracy) 
				

# Helper Functions, taken from the mnist tutorial for CNN: https://www.tensorflow.org/get_started/mnist/pros


def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
												strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def next_batch(data,info, batch_size):
	nrows = data.shape[0]
	indices = random.sample(range(nrows),batch_size)
	return data[indices,:],info[indices,:]


def deepnn(x,nchannels,nsamples,fmap_size 2048):
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

	# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
	# is down to 7x7x64 feature maps -- maps this to 1024 features.
	with tf.name_scope('fc1'):
		W_fc1 = weight_variable([(nchannels / 4) * (nsamples / 4)* 64, fmap_size])
		b_fc1 = bias_variable([fmap_size])
		print(W_fc1,'W_fc1')
		print(b_fc1,'b_fc1')

		h_pool2_flat = tf.reshape(h_pool2, [-1, (nchannels/4)*(nsamples/4)*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
		print(h_pool2_flat,'h_pool2_flat')
		print(h_fc1,'h_fc1')

	# Dropout - controls the complexity of the model, prevents co-adaptation of
	# features.
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# Map the 1024 features to 10 classes, one for each digit
	with tf.name_scope('fc2'):
		W_fc2 = weight_variable([fmap_size, 2])
		b_fc2 = bias_variable([2])
		print(W_fc2,'W_fc2')
		print(b_fc2,'b_fc2')

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	print(y_conv,'y_conv')
	return y_conv, keep_prob
