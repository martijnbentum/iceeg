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
import scipy.signal as signal
import windower

import model4
import model5
import model6


import tensorflow as tf


class model_deep_artifact:
	'''Train a convolution neural network on EEG data slices of 1 second (sf=100), labelled for artifacts.'''

	def __init__(self,data,batch_size = 200,nchannels=25,nsamples= 100,perc_artifact = .5,tf_version = '1.2.1',model_number =4):
		'''Create object to train CNN on EEG data slices, for an artifact classifier.

		info 			np matrix with artifact info and pp info, indices correspond with data matrix
		data 			np matrix with eeg data, each rows contains 2400 data point, 24 channels with 100 dp each -> 1second
		batch_size 		number of EEG slices to use per training cycle
		nchannel 		number of channels (24)
		nsamples 		number of samples per channel
		test_set_perc 	number of EEG slices to leave out for testing
		tf_version 		tensorflow version used for training
		model_number 	the number of the cnn model used, see model[number].py, define a model file with number before changing
						default and add to import statements, (model5 is a more complicate version of 4, see file)
		'''
		self.data = data
		self.nchannels = nchannels
		self.nsamples = nsamples
		self.perc_artifact = perc_artifact
		self.model_number = model_number
		self.tf_version = tf_version
		self.train_accuracy_history, self.test_accuracy_history = [], []
		self.define_network()


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
		if self.model_number == 4:
			self.y_conv, self.keep_prob = model4.deepnn(self.x,self.nchannels,self.nsamples)
		if self.model_number == 5:
			self.y_conv, self.keep_prob = model5.deepnn(self.x,self.nchannels,self.nsamples)
		if self.model_number == 6:
			self.y_conv, self.keep_prob = model6.deepnn(self.x,self.nchannels,self.nsamples)

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


	def train(self,ncycles= 100):
		'''Train a batch of EEG slices and report accuracy on a small test set.
		data is loaded in the cnn_data object, clean and artifact data is loaded seperately
		the ratio of clean and artifact is set in perc_artifact and executed in next_batch_ratio
		the batch size is set in self.nsamples

		ncycles 	number of times to load a batch and train'''

		if not hasattr(self.data,'artifact_train'): self.data.load_next_training_part()
		if not hasattr(self.data,'small_artifact_test'): self.data.load_small_test_set()
		for i in range(ncycles):
			print(i,'training')
			batch = next_batch_ratio(self.data.artifact_train,self.data.clean_train,self.perc_artifact,self.nsamples)
			if i % 100 == 0:
				small_test_batch = next_batch_ratio(self.data.small_artifact_test,self.data.small_clean_test,.5,500)
				train_accuracy = self.accuracy.eval(feed_dict={
						self.x: small_test_batch[0], self.y_: small_test_batch[1], self.keep_prob: 1.0},session = self.sess)
				print('step %d, accuracy on 500 test items:%g' % (i, train_accuracy))
				self.train_accuracy_history.append(train_accuracy)
			self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5},session = self.sess)

		test_batch = next_batch_ratio(self.data.small_artifact_test,self.data.small_clean_test,.5,1000)
		test_accuracy = self.accuracy.eval(feed_dict={ self.x: test_batch[0], self.y_: test_batch[1], self.keep_prob: 1.0},session = self.sess)
		self.test_accuracy_history.append(test_accuracy)
		print('test accuracy %g' %test_accuracy) 


	def eval_test_set(self,load_next_test_part = True, batch_size=800,save = False,identifier = ''):
		'''Evaluate a test set, artifact and clean data is loaded and test seperately.
		the test sets are loaded in the cnn_data object according to 10 fold cross validation, each fold
		is divided in multiple parts due to data size, one part is loaded per evaluation.

		load_next... 	whether to load the file at the next index
		batch_size 		number of epochs to test in one go, depends on RAM size for optimum value
		save 			whether to save the evaluation output
		identifier 		string to be prepended to the evaluation filename
		'''
		if load_next_test_part: 
			loaded = self.data.load_next_test_part()
			if not loaded: return False
		print('testing artifacts')
		self.predict_artifacts_class = self.compute_prediction_class(self.data.artifact_test,batch_size)
		self.predict_artifacts_perc = self.prediction_perc
		print('testing clean')
		self.predict_clean_class = self.compute_prediction_class(self.data.clean_test,batch_size)
		self.predict_clean_perc = self.prediction_perc

		self.predicted = np.concatenate((self.predict_artifacts_class,self.predict_clean_class))
		temp1 = np.ones((self.data.artifact_test.shape[0]))
		temp2 = np.zeros((self.data.clean_test.shape[0]))
		self.ground_truth = np.concatenate((temp1,temp2))

		self.confusion_matrix = sklearn.metrics.confusion_matrix(self.ground_truth,self.predicted)
		self.report = sklearn.metrics.classification_report(self.ground_truth,self.predicted)
		print(self.report)
		self.save_evaluation(identifier)
		return True


	def predict_block(self,b, batch_size=500, identifier = ''):
		'''Use currently loaded model to classify epochs of block.
		The corresponding eeg data is loaded and windowed in the cnn_data object
		a percentage and class prediction is saved.

		batch_size 		number of epochs to classify in one go
		identifier 		string to be prepended to the filename
		'''
		self.b = b
		self.data.block2eegdata(b)
		name = windower.make_name(self.b)
		print('testing block:',name)
		self.prediction_block_class = self.compute_prediction_class(self.data.d,batch_size)
		self.prediction_block_perc = self.prediction_perc
		output_name = path.model_predictions + identifier + '_' + self.filename_model.split('/')[-1] + '_' + name 
		print('saving predictions to filename:',output_name)
		np.save(output_name + '_class',self.prediction_block_class)
		np.save(output_name + '_perc',self.prediction_block_perc)


	def save_evaluation(self, identifier = ''):
		'''Save the ground truth, predicted and confusion matrix.

		identifier 		string to be prepended to the filename
		'''
		part = str(self.data.current_part_index + 1)
		test_part = str(self.data.current_test_part_index + 1)
		fold = str(self.data.fold)
		perc_artifact = str(int(self.perc_artifact *100))
		eval_name = path.model + identifier + 'evaluation_perc-'+perc_artifact+'_fold-'+ fold + '_part-' + part + '_tp-' + test_part
		cm_name= path.data+ identifier + 'evaluation_perc-'+perc_artifact+'_fold-'+ fold + '_part-' + part + '_tp-' + test_part
		print('saving evaluation:',eval_name)
		np.save(eval_name + '_gt',self.ground_truth)
		np.save(eval_name + '_predicted',self.predicted)
		np.save(eval_name + '_cm',self.confusion_matrix)
		np.save(cm_name+ '_cm',self.confusion_matrix)
		

	def compute_prediction_perc(self,data, batch_size = 800):
		'''Return percentage for each class for each epoch.
		data 	must be ndarray of correct dimensions (nepochs,(2500 = nchannelsXnsamples))
		'''
		print('precentage for both classes, first column: clean, second column artifact')
		self.predict_data = hamming_data(data)
		batch_indices = make_consecutive_batch_indices(data.shape[0],batch_size)
		self.prediction_conv_raw = np.zeros((data.shape[0],2))
		print('stepping throug data with:',batch_size,'samples in:',int(data.shape[0]/batch_size),'steps')
		i = 0
		for start_index, end_index in batch_indices:
			print(i)
			self.prediction_conv_raw[start_index:end_index,:] = self.y_conv.eval(feed_dict={self.x:self.predict_data[start_index:end_index],self.keep_prob:1.0},session=self.sess)
			i += 1

		self.prediction_perc = self.sess.run(tf.nn.softmax(self.prediction_conv_raw))
		
		return self.prediction_perc


	def compute_prediction_class(self,data,batch_size = 800):
		'''Return class for each epoch 1=artifact 0=artifact.
		data 	must be ndarray of correct dimensions (nepochs,(2500 = nchannelsXnsamples))
		'''
		print('class label, 1=artifact, 0=no artifact')
		self.predict_data = data
		temp = self.compute_prediction_perc(self.predict_data,batch_size)
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


	def make_model_name(self, identifier = ''):
		part = str(self.data.current_part_index + 1)
		fold = str(self.data.fold)
		perc_artifact = str(int(self.perc_artifact *100))
		self.filename_model = path.model + identifier +'perc-'+perc_artifact+'_fold-'+ fold + '_part-' + part


	def save_model(self,identifier = ''):
		'''Load a previously trained model by name and return model object.
		'''
		self.make_model_name()
		saver = tf.train.Saver()
		saver.save(self.sess,self.filename_model)
		# ma.initialize()


	def handle_folds(self,start_part = 1,save_every_nsteps = 10,evaluate = True,identifier = '',nparts ='all', ntrain = 1000):
		'''Train model on a specific fold (ordering of training and test files) and train them on some of all of these files.
		start_part 		the part to start training on
		save_every... 	the number of training files after which to evaluate the model
		identifier 		the string that is prepended to the model and evaluation files
		nparts 			the part number after which to stop training, all is training until the end
		ntrain 			number of training cycles
		'''
		if nparts != 'all' and nparts < save_every_nsteps: save_every_nsteps = nparts
		print('start from part',start_part,'save model at every',save_every_nsteps,'steps.')
		print('training on',nparts,'parts')
		print('artifact percentage while training:',self.perc_artifact)
		if start_part > 1:
			self.data.current_part_index = start_part -2
			self.data.part = start_part -1
			
		while True:
			print('start fold:',self.data.fold)
			loaded = self.data.load_next_training_part()
			print('working on part:',self.data.part)
			if not loaded:
				print('Done')
				break
			self.train(ntrain)
			if self.data.part % save_every_nsteps == 0:
				print('saving model and evaluating...')
				self.save_model(identifier)
				self.eval_test_set(save = True, identifier = identifier)
				delattr(self.data,'current_test_part_index')
				if nparts != 'all' and nparts <= self.data.part: 
					print('trained on',self.data.part,'parts, stop training.')
					break

	def handle_eval_all_test_sets(self,start_part =2, identifier= '',batch_size = 800):
		'''Test the current model on all test files.
		start_part 		there are multiple test files (10), typically the first is already tested during training.
		identifier 		string prepended to the filename to distinguish different test runs
		batch_size 		number of epochs to test in 1 go, 800 works well with 16gb of RAM
		'''

		self.data.current_test_part_index = start_part - 2 # the eval_test_set method will load the test set at the next index
		print('starting at part',start_part,'setting index to',self.data.current_test_part_index,'the evaluation method will load the file at the next index.')
		i = 0 
		while True:
			loaded = self.eval_test_set(save=True,identifier = identifier,batch_size = batch_size)
			if not loaded:
				print('done with testing,tested:',i,'test_files.')
			i +=1

	def block2prediction(self,b):
		self.data.block2eegdata(b)
		
			
def handle_artifact_percs(m,artifact_percs = [0.9375,0.0625,0.875,0.125,0.75,0.25],nparts = 10,start_part = 1,identifier = 'perc_comparison',ntrain = 1000):
	if not hasattr(m,'sess'): m.initialize()
	for i,perc in enumerate(artifact_percs):
		print(i,len(artifact_percs),'artifact perc:',perc,'train cycles:',ntrain,'nparts',nparts)
		if start_part > 1:
			print('loading previous model',path.model + identifier + 'perc-' + str(int(perc*100)) + '_fold-'+str(m.data.fold) + '_part-' + str(start_part - 1))
			m.clean_up()
			m = load_model(path.model + identifier + 'perc-' + str(int(perc*100)) + '_fold-'+str(m.data.fold) + '_part-' + str(start_part - 1),m.data)
		m.perc_artifact = perc
		m.handle_folds(start_part = start_part,nparts = nparts, ntrain = ntrain, identifier = identifier)
		m.clean_up()
		m.data.clean_up()
		d = m.data
		m = model_deep_artifact(d)
		m.initialize()


def load_model(model_name,d):
	'''Load a previously trained model by name and return model object.
	'''
	m = model_deep_artifact(d)
	m.sess = tf.Session()
	m.saver = tf.train.Saver()
	m.saver.restore(m.sess,model_name)
	m.filename_model = model_name
	# ma.initialize()
	return m


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
	

def samplen_artifact_clean(nsamples,perc_artifacts):
	'''Create nartifact nclean int based on nsample (total n samples) and perc_artifacts.'''
	nartifact = int(round(nsamples * perc_artifacts))
	nclean = int(round(nsamples * (1-perc_artifacts)))
	assert nartifact + nclean == nsamples
	return nartifact,nclean

def hamming_data_obsolete(d):
	#Multiply all epochs with a hamming window. obsolete
	window_length = d.shape[1]
	hamming = signal.hamming(window_length)
	return d * hamming

def hamming_data(d,nchannels = 25):
	'''Multiply all epochs with a hamming window per channel.
	Each epoch (row of data in d) contains 100 datapoints from 25 channels
	The data from each channel should be multiplied with the hamming window
	'''
	window_length = int(d.shape[1]/ nchannels)
	hamming = signal.hamming(window_length)
	hamming = np.concatenate([hamming] * nchannels)
	return d * hamming

def next_batch_ratio(artifact_data ,clean_data ,perc_artifacts =.5, nsamples = 200):
	'''Create a batch of nsmamples randomly drawn rows with a ratio clean/artifact 
	defined by perc_artifact.'''
	nartifact, nclean = samplen_artifact_clean(nsamples,perc_artifacts)
	nrows_artifact = artifact_data.shape[0]
	nrows_clean = clean_data.shape[0]
	indices_artifact = random.sample(range(nrows_artifact),nartifact)
	indices_clean = random.sample(range(nrows_clean),nclean)
	# print(indices_artifact,indices_clean)
	data = np.concatenate((artifact_data[indices_artifact,:],clean_data[indices_clean,:]))
	info = np.zeros((nartifact + nclean, 2))
	
	data = hamming_data(data)

	info[:nartifact,1] = 1
	info[nartifact:,0] = 1
	indices = list(range(nartifact+nclean))
	random.shuffle(indices)
	return data[indices,:],info[indices,:]


def  make_consecutive_batch_indices(nrows,nsamples):
	'''Return list of [start_index,end_index] which slices up the nrows into nsamples batches.'''
	start_indices = np.arange(0,nrows,nsamples)
	end_indices = start_indices + nsamples
	end_indices[-1] = nrows
	return zip(start_indices,end_indices)

