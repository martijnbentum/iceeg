# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

#based on model_cnn
#model file 7 (model7) based on model 4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import cnn_output_data 
import copy
import glob
# import model_cnn_output
import numpy as np
import path
import progressbar as pb
import random
import sklearn.metrics
import sys
import time
import tempfile
import scipy.signal as signal
import windower

import importlib
'''
temp = glob.glob('model*')
models = []
for m in temp:
	try:
		int(m[5])
		models.append(m.split('.')[0])
	except: continue
[importlib.import_module(m) for m in models]
'''

import tensorflow as tf


class model_channel_artifact:
	'''Train a convolution neural network on EEG data slices of 1 second (sf=100), labelled for channel artifacts.'''

	def __init__(self,data,batch_size = 200,nchannels=26,nsamples= 100,perc_artifact = .5,kernel_size = 6,tf_version = '1.2.1',model_number =7):
		'''Create object to train CNN on EEG data slices, for an artifact classifier.

		info 			np matrix with artifact info and pp info, indices correspond with data matrix
		data 			np matrix with eeg data, each rows contains 2600 data point, 26 channels with 100 dp each -> 1second
		batch_size 		number of EEG slices to use per training cycle
		nchannel 		number of channels (26)
		nsamples 		number of samples per channel
		tf_version 		tensorflow version used for training
		model_number 	the number of the cnn model used, see model[number].py, define a model file with number before changing default and add to import statements
						
		'''
		self.chindex = make_ch_index_dict(nchannels= nchannels,kernel_size = kernel_size)
		self.nrows_per_sample = nchannels + len(list(range(0,26,kernel_size-1)))
		self.data = data
		self.nchannels = nchannels
		self.nsamples = nsamples
		self.perc_artifact = perc_artifact
		self.model_number = model_number
		self.kernel_size = kernel_size
		self.tf_version = tf_version
		self.model_name = 'model' + str(self.model_number)
		self.model_module = importlib.import_module(self.model_name)
		self.train_accuracy_history, self.test_accuracy_history = [], []
		self.define_network()


	def define_network(self):
		'''Declare the tensorflow placeholder variables.
		This defines the graph of the model and creates handles on the model object to interact with tensorflow.
		It is used when starting from scratch and when loading a new model.
		'''
		# Create the model
		self.x = tf.placeholder(tf.float32, [None, self.nrows_per_sample* self.nsamples])
		print(self.x,'x')
		# Define loss and optimizer
		self.y_ = tf.placeholder(tf.float32, [None, 2])
		print(self.y_,'y_')
		# Build the graph for the deep net
		deepnn = getattr(self.model_module,'deepnn')
		self.y_conv, self.keep_prob = deepnn(self.x,self.nrows_per_sample,self.nsamples,self.kernel_size)

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
		bar = pb.ProgressBar()
		bar(range(ncycles))

		if not hasattr(self.data,'train_data'): self.data.load_next_training_part()
		if not hasattr(self.data,'smalltest_data'): self.data.load_small_test_set()
		for i in range(ncycles):
			# print(i,'training')
			bar.update(i)
			batch = next_batch_ratio(self.data,perc_artifacts= self.perc_artifact,nsamples=self.nsamples)
			if i % int(ncycles/10) == 0:
				small_test_batch = next_batch_ratio(self.data,name='smalltest',perc_artifacts=.5,nsamples=500)
				train_accuracy = self.accuracy.eval(feed_dict={
						self.x: small_test_batch[0], self.y_: small_test_batch[1], self.keep_prob: 1.0},session = self.sess)
				print('step %d, accuracy on 500 test items:%g' % (i, train_accuracy))
				# print('step %d, loss:%g' % (i, loss))
	
				self.train_accuracy_history.append(train_accuracy)
			self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5},session = self.sess)

		test_batch = next_batch_ratio(self.data,'smalltest',perc_artifacts=.5,nsamples=1000)
		test_accuracy = self.accuracy.eval(feed_dict={ self.x: test_batch[0], self.y_: test_batch[1], self.keep_prob: 1.0},session = self.sess)
		self.test_accuracy_history.append(test_accuracy)
		print('test accuracy %g' %test_accuracy) 
		clock = time.strftime("%H:%M:%S\t%b-%d-%Y", time.localtime(time.time()))
		fout = open(path.data + 'test_output.txt','a')
		fout.write(str(test_accuracy) +'\t' +str(self.perc_artifact) + '\t' + str(self.data.part)+'\t' + clock+ '\tkernel' + str(self.kernel_size) + '\t' + self.model_name + '\n')
		fout.close()


	def eval_test_set(self,load_next_test_part = True, batch_size=1200,save = False,identifier = '',subset = True):
		'''Evaluate a test set, artifact and clean data is loaded and test seperately.
		the test sets are loaded in the cnn_data object according to 10 fold cross validation, each fold
		is divided in multiple parts due to data size, one part is loaded per evaluation.

		load_next... 	whether to load the file at the next index
		batch_size 		number of epochs to test in one go, depends on RAM size for optimum value
		save 			whether to save the evaluation output
		identifier 		string to be prepended to the evaluation filename
		subset 			whether to select 25% of the test file for evaluation
						cuts down 4 M to 1 M of samples
		'''
		if load_next_test_part: 
			loaded = self.data.load_next_test_part()
			if not loaded: return False

		data =getattr(self.data,'test_data')
		info =getattr(self.data,'test_info')
		if subset:
			indices = np.load(path.channel_artifact_training_data + 'PART_INDICES/eval_subset_indices_perc-25.npy')
			data = data[indices,:]
			info = info[indices,:]
		self.ground_truth = np.ravel(info,'F')
		self.predicted = np.zeros(self.ground_truth.shape[0])
		# self.predicted_perc = np.zeros((self.ground_truth.shape[0]))
		nrows = data.shape[0]
		data = np.reshape(data,(nrows,-1))

		for i in range(self.nchannels):
			print('original ch index:',i,'transelated index:',self.chindex[i])
			set_target_channel(data,[self.chindex[i]] * data.shape[0])
			self.predicted[i*nrows:i*nrows+nrows]= self.compute_prediction_class(data,batch_size)
			# self.predicted_perc[i*nrows:i*nrows+nrows]= self.prediction_perc

		self.confusion_matrix = sklearn.metrics.confusion_matrix(self.ground_truth,self.predicted)
		self.report = sklearn.metrics.classification_report(self.ground_truth,self.predicted)
		print(self.report)
		if save:
			self.save_evaluation(identifier)
		return True

	def predict_block(self,b, batch_size=800, identifier = '', save = True):
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

		data = self.data.d
		nrows = data.shape[0]
		data = np.reshape(data,(nrows,-1))

		self.prediction_block_class= np.zeros((nrows,self.nchannels))
		self.prediction_block_perc= np.zeros((nrows,self.nchannels))

		for i in range(self.nchannels):
			print('original ch index:',i,'transelated index:',self.chindex[i])
			set_target_channel(data,[self.chindex[i]] * data.shape[0])
			self.prediction_block_class[:,i]= self.compute_prediction_class(data,batch_size)
			self.prediction_block_perc[:,i]= self.prediction_perc[:,1]
			# self.predicted[i*nrows:i*nrows+nrows]= self.compute_prediction_class(data,batch_size)
			# self.predicted_perc[i*nrows:i*nrows+nrows,i]= self.prediction_perc[:,1]

		output_name = path.channel_snippet_annotation+ identifier + self.filename_model.split('/')[-1] + '_' + name 
		if save:
			print('saving predictions to filename:',output_name)
			np.save(output_name + '_class',self.prediction_block_class)
			np.save(output_name + '_perc',self.prediction_block_perc)
		return self.prediction_block_class, self.prediction_block_perc


	def save_evaluation(self, identifier = ''):
		'''Save the ground truth, predicted and confusion matrix.

		identifier 		string to be prepended to the filename
		'''
		part = str(self.data.current_part_index + 1)
		test_part = str(self.data.current_test_part_index + 1)
		fold = str(self.data.fold)
		perc_artifact = str(int(self.perc_artifact *100))
		kernel = str(self.kernel_size)
		model = self.model_name
		eval_name = path.model_channel + identifier + 'evaluation_perc-'+perc_artifact+'_fold-'+ fold + '_part-' + part + '_tp-' + test_part + '_kernel-'+kernel + model
		cm_name= path.data+ identifier + 'evaluation_perc-'+perc_artifact+'_fold-'+ fold + '_part-' + part + '_tp-' + test_part + '_kernel-'+kernel + model
		print('saving evaluation:',eval_name)
		np.save(eval_name + '_gt',self.ground_truth)
		np.save(eval_name + '_predicted',self.predicted)
		np.save(eval_name + '_cm',self.confusion_matrix)
		np.save(cm_name+ '_cm',self.confusion_matrix)

		fout = open(eval_name + '_report.txt','w')
		fout.write(self.report)
		fout.close()


		if hasattr(self,'predicted_adj'):
			np.save(eval_name + '_predicted_adj',self.predicted_adj)
			np.save(eval_name + '_cm_adj',self.confusion_matrix_adj)
			np.save(cm_name+ '_cm_adj',self.confusion_matrix_adj)
		

	def compute_prediction_perc(self,data, batch_size = 800):
		'''Return percentage for each class for each epoch.
		data 	must be ndarray of correct dimensions (nepochs,(2500 = nchannelsXnsamples))
		'''
		print('precentage for both classes, first column: clean, second column artifact')
		self.predict_data = hamming_data(data)
		batch_indices = make_consecutive_batch_indices(data.shape[0],batch_size)
		self.prediction_conv_raw = np.zeros((data.shape[0],2))
		nsteps = int(data.shape[0]/batch_size)
		bar = pb.ProgressBar()
		bar(range(nsteps))
		print('stepping throug data with:',batch_size,'samples in:',nsteps,'steps')
		i = 0
		for start_index, end_index in batch_indices:
			# print(i)
			bar.update(i)
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
		kernel = str(self.kernel_size)
		model = self.model_name
		perc_artifact = str(int(self.perc_artifact *100))
		self.filename_model = path.model_channel + identifier +'perc-'+perc_artifact+'_fold-'+ fold + '_part-' + part + '_kernel-' + kernel + '_' + model


	def save_model(self,identifier = ''):
		'''Load a previously trained model by name and return model object.
		'''
		self.make_model_name(identifier = identifier)
		saver = tf.train.Saver()
		saver.save(self.sess,self.filename_model,write_meta_graph=False)
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

	def handle_eval_all_test_sets(self,start_part =2, identifier= '',batch_size = 800,cpi = 89,add_adjustment = False):
		'''Test the current model on all test files.
		start_part 		there are multiple test files (10), typically the first is already tested during training.
		identifier 		string prepended to the filename to distinguish different test runs
		batch_size 		number of epochs to test in 1 go, 800 works well with 16gb of RAM
		cpi 			current part index, assumed to be 90 (trained on all files')
		'''

		self.data.current_test_part_index = start_part - 2 # the eval_test_set method will load the test set at the next index
		self.data.current_part_index = cpi
		print('starting at part',start_part,'setting index to',self.data.current_test_part_index,'the evaluation method will load the file at the next index.')
		i = 0 
		while True:
			loaded = self.eval_test_set(save=True,identifier = identifier,batch_size = batch_size,add_adjustment = add_adjustment)
			if not loaded:
				print('done with testing,tested:',i,'test_files.')
				break
			i +=1
			if add_adjustment: 
				filename_model = self.filename_model
				self.clean_up()
				self = load_model(filename_model,self.data)

	def block2prediction(self,b):
		self.data.block2eegdata(b)
		
			
def handle_artifact_percs(m,artifact_percs = [0.1,0.6,0.4,0.875,0.125,0.75,0.25],nparts = 10,start_part = 1,identifier = 'perc-comparison_',ntrain = 1000,save_every_nsteps=10):
	if not hasattr(m,'sess'): m.initialize()
	for i,perc in enumerate(artifact_percs):
		print(i,len(artifact_percs),'artifact perc:',perc,'train cycles:',ntrain,'nparts',nparts)
		if start_part > 1:
			print('loading previous model',path.model_channel + identifier + 'perc-' + str(int(perc*100)) + '_fold-'+str(m.data.fold) + '_part-' + str(start_part - 1))
			m.clean_up()
			m = load_model(path.model_channel + identifier + 'perc-' + str(int(perc*100)) + '_fold-'+str(m.data.fold) + '_part-' + str(start_part - 1),m.data)
		m.perc_artifact = perc
		m.handle_folds(start_part = start_part,nparts = nparts, ntrain = ntrain, identifier = identifier,save_every_nsteps=save_every_nsteps)
		m.clean_up()
		m.data.clean_up()
		d = m.data
		m = model_channel_artifact(d)
		m.initialize()


def load_model(model_name,d,nthreads = 'all'):
	'''Load a previously trained model by name and return model object.
	'''
	m = model_channel_artifact(d)
	if nthreads != 'all' and type(nthreads) == int:
		session_config = tf.ConfigProto(intra_op_parallelism_threads=nthreads,inter_op_parallelism_threads=nthreads)
		m.sess=tf.Session(config = session_config)
	else: m.sess = tf.Session()
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


def hamming_data(d,nchannels = None,nsamples = 100):
	'''Multiply all epochs with a hamming window per channel.
	Each epoch (row of data in d) contains 100 datapoints from 26 channels
	The data from each channel should be multiplied with the hamming window
	'''
	if nchannels == None: nchannels = int(d.shape[1] / nsamples)
	# self.hamming_data_nchannels = nchannels
	# print('hamming nchannels:',nchannels)
	window_length = int(d.shape[1]/ nchannels)
	hamming = signal.hamming(window_length)
	hamming = np.concatenate([hamming] * nchannels)
	return d * hamming

def next_batch_ratio(d, name = 'train',perc_artifacts =.5, nsamples = 200):
	'''Create a batch of nsmamples randomly drawn rows with a ratio clean/artifact 
	defined by perc_artifact.'''
	nartifact, nclean = samplen_artifact_clean(nsamples,perc_artifacts)

	asi =getattr(d,name+ '_artifact_sample_indices')
	csi =getattr(d,name+ '_clean_sample_indices')

	aci =getattr(d,name+ '_artifact_ch_indices')
	cci =getattr(d,name+ '_clean_ch_indices')
	data =getattr(d,name+ '_data')

	indices_artifact = random.sample(range(asi.shape[0]),nartifact)
	indices_clean = random.sample(range(csi.shape[0]),nclean)
	# print(indices_artifact,indices_clean)
	data = np.concatenate((data[asi[indices_artifact],:],data[csi[indices_clean],:]))
	target_channel_indices = np.concatenate((aci[indices_artifact],cci[indices_clean]))
	info = np.zeros((nartifact + nclean, 2))
	
	data = np.reshape(data,(nsamples,-1))
	data = hamming_data(data)
	set_target_channel(data,target_channel_indices)

	info[:nartifact,1] = 1
	info[nartifact:,0] = 1
	# indices = list(range(nartifact+nclean))
	# random.shuffle(indices)
	# return data[indices,:],info[indices,:]
	return data,info,target_channel_indices


def  make_consecutive_batch_indices(nrows,nsamples):
	'''Return list of [start_index,end_index] which slices up the nrows into nsamples batches.'''
	start_indices = np.arange(0,nrows,nsamples)
	end_indices = start_indices + nsamples
	end_indices[-1] = nrows
	return zip(start_indices,end_indices)


def set_target_channel(data,target_channels,kernel_size = 6, nchannels = None, nsamples = 100):
	'''sets target channel to dataset which already has target channel inserted works in place.

	data 		data which is already inserted with a target channel, provided target channel
				will overwrite the previously inserted target channel is faster: 570 vs 8.5 ms
	target_ch.. the channel to be set on previously insert target channel rows
				channel index should already be converted to inserted matrix index
	kernel_size height of the kernel
	nchannels 	number of channels in a sample
	'''
	if nchannels == None and len(data.shape) == 3: nchannels = data.shape[1]
	elif nchannels == None and len(data.shape) == 2: nchannels = int(data.shape[1] / nsamples)
	# self.set_target_nschannels = nchannels
	indices = list(range(0,nchannels,kernel_size))
	for i,m in enumerate(data):
		m = np.reshape(m,[nchannels,-1])
		d = m[indices,:] = m[target_channels[i],:]
		d = np.reshape(d,[1,-1])


		
def make_ch_index_dict(nchannels = 26, kernel_size = 6):
	'''Return a dictthat transelates between org ch index and index after insert target ch.
	Target channel is inserted at multiple locations which shifts the index of the channels
	this dict transelate between the index of the original location and the location after
	the insertion of the target channel.

	nchannels 		the number eeg channels used in training
	kernel_size 	the height of the kernel used in training, this determines the number
					and location of target channel isertions and is therefore needed to compute
					the new indices of the channels
	'''
	chindex_dict = {}
	insert_indices = list(range(0,nchannels,kernel_size-1))
	org_indices = np.arange(26)
	new_indices = list(np.insert(org_indices,insert_indices,-1,axis=0))
	for i in range(26):
		chindex_dict[i] = new_indices.index(i)
	return chindex_dict
	
	
