# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import path
import progressbar as pb
import random
import sklearn.metrics
import sys
import tempfile
import scipy.signal as signal
import windower

# import model1


import tensorflow as tf


class model_cnn_output:
	'''Train a convolution neural network on EEG data slices of 1 second (sf=100), labelled for artifacts.'''

	def __init__(self,data,batch_size = 200,nsamples= 301,perc_artifact= .9,gradient_descent_opt= 0.01,hamming=False,tf_version = '1.2.1'):
		'''Create object to train CNN on EEG data slices, for an artifact classifier.

		data 			np matrix with eeg data, each rows contains 2400 data point, 24 channels with 100 dp each -> 1second
		batch_size 		number of EEG slices to use per training cycle
		nsamples 		number of context data points surrounding target dp
		perc_arti... 	percentage of true artifact in a training batch, the higher this number the better
						the recall of artifacts (lower recall for true clean instance)
		tf_version 		tensorflow version used for training
						define a model file with number before changing
		'''
		self.data = data
		self.batch_size = batch_size
		self.nsamples = nsamples
		self.perc_artifact = perc_artifact
		self.gradient_descent_opt = gradient_descent_opt
		self.hamming = hamming
		self.tf_version = tf_version
		self.train_accuracy_history, self.test_accuracy_history = [], []
		self.define_network()
		self.bar = pb.ProgressBar()


	def define_network(self):
		'''Declare the tensorflow placeholder variables.
		This defines the graph of the model and creates handles on the model object to interact with tensorflow.
		It is used when starting from scratch and when loading a new model.
		'''
		#set place holders for the input of the data and info
		self.x = tf.placeholder(tf.float32, [None, self.nsamples])
		print(self.x)
		self.y_ = tf.placeholder(tf.float32, [None, 2])
		print(self.y_)

		# Create the model
		self.W = tf.Variable(tf.zeros([self.nsamples,2]))
		self.b = tf.Variable(tf.zeros([2]))
		self.y = tf.nn.softmax(tf.matmul(self.x,self.W) + self.b)

		# Define loss and optimizer
		self.cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
		self.train_step = tf.train.GradientDescentOptimizer(self.gradient_descent_opt).minimize(self.cross_entropy)

		with tf.name_scope('accuracy'):
			correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
			self.correct_prediction = tf.cast(correct_prediction, tf.float32)
		self.accuracy = tf.reduce_mean(self.correct_prediction)



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


	def train(self,ncycles= 100000):
		'''Train a batch of EEG slices and report accuracy on a small test set.
		data is loaded in the cnn_output_data object, clean and artifact data is loaded seperately
		the ratio of clean and artifact is set in perc_artifact and executed in next_batch_ratio
		the batch size is set in self.batch_size

		ncycles 	number of times to load a batch and train
		'''
		self.bar(range(ncycles))

		for i in range(ncycles):
			self.bar.update(i)
			# def next_batch_ratio(d, name = 'train',perc_artifacts =.5, nsamples = 200, hamming = True):
			batch = next_batch_ratio(self.data,perc_artifacts =self.perc_artifact,nsamples=self.batch_size,hamming = self.hamming,window_length = self.nsamples)
			if i % 100000 == 0:
				small_test_batch = next_batch_ratio(self.data,name='smalltest',perc_artifacts=.5,nsamples=500,hamming=self.hamming,window_length = self.nsamples)
				train_accuracy = self.accuracy.eval(feed_dict={
						self.x: small_test_batch[0], self.y_: small_test_batch[1]},session = self.sess)
				print('step %d, accuracy on 500 test items:%g' % (i, train_accuracy))
				self.train_accuracy_history.append(train_accuracy)
			self.sess.run(self.train_step,feed_dict={self.x:batch[0],self.y_:batch[1]})


	def eval_test_set(self, batch_size=800,save = False,identifier = ''):
		'''Evaluate a test set, artifact and clean data is loaded and test seperately.

		batch_size 		number of epochs to test in one go, depends on RAM size for optimum value
		save 			whether to save the evaluation output
		identifier 		string to be prepended to the evaluation filename
		'''
		print('testing artifacts')
		test_artifacts = self.data.test_data[self.data.test_artifact_indices,:]
		test_clean = self.data.test_data[self.data.test_clean_indices,:]

		self.predict_artifacts_class = self.compute_prediction_class(test_artifacts, batch_size)
		self.predict_artifacts_perc = self.prediction_perc
		print('testing clean')
		self.predict_clean_class = self.compute_prediction_class(test_clean, batch_size)
		self.predict_clean_perc = self.prediction_perc

		self.predicted = np.concatenate((self.predict_artifacts_class,self.predict_clean_class))
		temp1 = np.ones((test_artifacts.shape[0]))
		temp2 = np.zeros((test_clean.shape[0]))
		self.ground_truth = np.concatenate((temp1,temp2))

		self.confusion_matrix = sklearn.metrics.confusion_matrix(self.ground_truth,self.predicted)
		self.report = sklearn.metrics.classification_report(self.ground_truth,self.predicted)
		print(self.report)
		if save:
			self.save_evaluation(identifier)
		return True


	def save_evaluation(self, identifier = ''):
		'''Save the ground truth, predicted and confusion matrix.

		identifier 		string to be prepended to the filename
		'''
		perc_artifact = str(int(self.perc_artifact *100))
		eval_name = path.channel_cnn_output_data+'rep-'+str(self.rep)+'_cnn_outputdata_' + identifier + 'evaluation_perc-'+perc_artifact 
		cm_name= path.data+'channel_cnn_outputdata_'+ identifier + 'evaluation_perc-'+perc_artifact
		print('saving evaluation:',eval_name)
		np.save(eval_name + '_gt',self.ground_truth)
		np.save(eval_name + '_predicted',self.predicted)
		np.save(eval_name + '_cm',self.confusion_matrix)
		np.save(cm_name+ '_cm',self.confusion_matrix)
		fout = open(eval_name + '_report.txt','w')
		fout.write(self.report)
		fout.close()
		

	def compute_prediction_perc(self,data, batch_size = 800):
		'''Return percentage for each class for each epoch.

		data 	must be ndarray of correct dimensions (nepochs,nsamples)
		'''
		print('precentage for both classes, first column: clean, second column artifact')
		batch_indices = make_consecutive_batch_indices(data.shape[0],batch_size)
		self.prediction_raw = np.zeros((data.shape[0],2))
		nsteps = int(data.shape[0]/batch_size)
		self.bar(range(nsteps))
		print('stepping throug data with:',batch_size,'samples in:',nsteps,'steps')
		i = 0
		for start_index, end_index in batch_indices:
			# print(i)
			self.bar.update(i)
			self.prediction_raw[start_index:end_index,:] = self.y.eval(feed_dict={self.x:self.predict_data[start_index:end_index]},session=self.sess)
			i += 1

		self.prediction_perc = self.sess.run(tf.nn.softmax(self.prediction_raw))
		
		return self.prediction_perc


	def compute_prediction_class(self,data,batch_size = 800):
		'''Return class for each epoch 1=artifact 0=artifact.
		data 	must be ndarray of correct dimensions (nepochs,nsamples)
		'''
		print('class label, 1=artifact, 0=no artifact')
		self.predict_data = data
		temp = self.compute_prediction_perc(self.predict_data,batch_size)
		self.prediction_class = self.sess.run(tf.argmax(self.prediction_perc,1))
		return self.prediction_class


	def make_model_name(self, identifier = ''):
		perc_artifact = str(int(self.perc_artifact *100))
		self.filename_model = path.model_channel + 'rep-'+str(self.rep)+'_cnn_output_model_'+ identifier +'perc-'+perc_artifact


	def save_model(self,identifier = ''):
		'''Load a previously trained model by name and return model object.
		'''
		self.make_model_name(identifier = identifier)
		saver = tf.train.Saver()
		saver.save(self.sess,self.filename_model)
		# ma.initialize()

	def handle_parts(self,randomize_order = True, nparts = 50, ntrain = 500000,eval_every = 10,identifier = ''):
		self.rep = 1
		for part in range(nparts):
			print('will train:',nparts,'currently at:',part+1)
			if part > 0:
				if randomize_order:
					self.data.load_next_training_part(index = random.randint(0,8))
				else: self.data.load_next_training_part()
			self.train(ntrain)
			if part != 0 and part % eval_every == 0:
				self.rep += 1
				self.save_model(identifier)
				self.eval_test_set(save = True, identifier = identifier)
		
	
	

			
def handle_artifact_percs(m,artifact_percs = [0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.9,0.75,0.5],identifier = 'perc_comparison',ntrain = 500000):
	if not hasattr(m,'sess'): m.initialize()
	for i,perc in enumerate(artifact_percs):
		print(i,len(artifact_percs),'artifact perc:',perc,'train cycles:',ntrain)
		m.perc_artifact = perc
		m.train(ntrain)
		print('saving model and evaluating...')
		m.save_model(identifier)
		m.eval_test_set(save = True, identifier = identifier)
		m.clean_up()
		do = m.data
		m = model_cnn_output(do,nsamples= m.nsamples)
		m.initialize()


def load_model(model_name,do):
	'''Load a previously trained model by name and return model object.
	'''
	m = model_cnn_output(do)
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
	

def samplen_artifact_clean(batch_size,perc_artifacts):
	'''Create nartifact nclean int based on nsample (total n samples) and perc_artifacts.'''
	nartifact = int(round(batch_size* perc_artifacts))
	nclean = int(round(batch_size* (1-perc_artifacts)))
	assert nartifact + nclean == batch_size 
	return nartifact,nclean


def hamming_data(d,window_length):
	'''Multiply all epochs with a hamming window.
	'''
	hamming = signal.hamming(window_length)
	return d * hamming


def next_batch_ratio(d, name = 'train',perc_artifacts =.5, nsamples = 200, hamming = False,window_length = 301):
	'''Create a batch of nsmamples randomly drawn rows with a ratio clean/artifact 
	defined by perc_artifact.'''
	nartifact, nclean = samplen_artifact_clean(nsamples,perc_artifacts)

	ai =getattr(d,name+ '_artifact_indices')
	ci =getattr(d,name+ '_clean_indices')

	data =getattr(d,name+ '_data')

	indices_artifact = random.sample(range(ai.shape[0]),nartifact)
	indices_clean = random.sample(range(ci.shape[0]),nclean)
	# print(indices_artifact,indices_clean)
	data = np.concatenate((data[ai[indices_artifact],:],data[ci[indices_clean],:]))
	info = np.zeros((nartifact + nclean, 2))

	if hamming:
		data = hamming_data(data,window_length)

	info[:nartifact,1] = 1
	info[nartifact:,0] = 1
	# indices = list(range(nartifact+nclean))
	# random.shuffle(indices)
	# return data[indices,:],info[indices,:]
	return data,info


def  make_consecutive_batch_indices(nrows,nsamples):
	'''Return list of [start_index,end_index] which slices up the nrows into nsamples batches.'''
	start_indices = np.arange(0,nrows,nsamples)
	end_indices = start_indices + nsamples
	end_indices[-1] = nrows
	return zip(start_indices,end_indices)

