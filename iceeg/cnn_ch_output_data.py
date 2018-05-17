import copy
import glob
import min_max_artifact as mma
import numpy as np
import os 
import path
import plot_roc
import progressbar as pb
import random
import sklearn.metrics
import windower

#adapted from cnn_output_data

class cnn_ch_output_data:
	'''Object to handle i/o for cnn ch output training.
	It presupposes that train and test files are generated based on model_ch_cnn output for each block
	These files can be created with cnn_channel_output2data class
	'''
	def __init__(self):
		'''Initialize data object that handles loading of training and test files for cnn ch output data training.
		the dataset is created with functions in cnn_channel_output2data class
		'''
		self.train_data_filename = path.channel_cnn_output_data +'train_data.npy'
		self.train_info_filename = path.channel_cnn_output_data +'train_info.npy'
		self.test_data_filename = path.channel_cnn_output_data + 'test_data.npy'
		self.test_info_filename = path.channel_cnn_output_data +'test_info.npy'

		# data is flattened: rowsXwindow_length, rows corresponds with a model_ch_cnn 
		# channel specific sample (a sample is 1sec window).
		# channel specific whether a 1 sec window is an artiact depends on which channels is in focus.
		self.train_data = np.load(self.train_data_filename)
		self.train_info = np.load(self.train_info_filename)
		self.test_data = np.load(self.test_data_filename)
		self.test_info = np.load(self.test_info_filename)

		self.smalltest_data= self.test_data[:1500,:]
		self.smalltest_info= self.test_info[:1500]

		self.ntrain = self.train_info.shape[0]
		self.ntest = self.test_info.shape[0]
		self.nsmalltest = self.smalltest_info.shape[0]

		# create set of indices for clean and artifact samples for train test and smalltest datasets
		self.train_artifact_indices = np.where(self.train_info >= .5)[0]
		self.train_clean_indices = np.where(self.train_info < .5)[0]
		self.test_artifact_indices = np.where(self.train_info >= .5)[0]
		self.test_clean_indices = np.where(self.train_info < .5)[0]
		self.smalltest_artifact_indices = np.where(self.smalltest_info >= .5)[0]
		self.smalltest_clean_indices = np.where(self.smalltest_info < .5)[0]

		self.ntrain_artifact = sum(self.train_info)
		self.ntest_artifact = sum(self.test_info)
		self.ntrain_clean = self.ntrain - self.ntrain_artifact
		self.ntest_clean = self.ntest - self.ntest_artifact

		self.nsmalltest_artifact = sum(self.smalltest_info)
		self.nsmalltest_clean = self.nsmalltest - self.nsmalltest_artifact

	def __str__(self):
		m = '\ncnn_output_data\n--------\n'
		m += 'nartifact train:\t' + str(self.ntrain_artifact) +'\n'
		m += 'nclean train:\t\t' + str(self.ntrain_clean) +'\n'
		m += 'nartifact test:\t\t' + str(self.ntest_artifact) +'\n'
		m += 'nclean test:\t\t' + str(self.ntest_clean) +'\n'
		m += 'nartifact smalltest:\t' + str(self.nsmalltest_artifact) +'\n'
		m += 'nclean smalltest:\t' + str(self.nsmalltest_clean) +'\n'
		return m
		

class cnn_ch_output2data:
	'''Create dataset to train a regression model on the predicted artifacts and remove the false positives.
	Create dataset for a specific eeg block and test predicted artifact and remove false positives.
	'''
	def __init__(self,name= '', window_length = 201, model_name = 'unk',predicted_perc = None):
		'''Initialize data object that handles loading of training and test files for cnn training.
		Can also be used to test predicted block data 

		name 		the name corresponding to predicted and ground truth, name 
					format windower.make_name(b) (b = block object) 		
		window_... 	the number of time points for each datapoints. The cnn ouput 
					gives a perc artifact for each each epoch
					but does not take into account the data before or after, 
					this dataset is for removing false positive
					predicted artifacts (the cnn_model has lower precision to achieve 
					high recall of artifacts
		model_n... 	name of the cnn model that preformed the perc artifact predictions
		predi... 	a indicesXnchannels np array of perc artifacts from cnn model 
					to be adjusted by the cnn_ch_output_model
					if none the data will be loaded based on name info
		'''
		if name == '' and list(predicted_perc) != np.ndarray:
			print('provide block_name or predicted_perc artifact')
		self.handle_annotated_block = False # sets to true when there is a ground truth file present
		self.name = name
		self.window_length = window_length
		self.model_name = model_name
		self.output_filename = path.channel_cnn_output_data + self.model_name + '_' + self.name

		if predicted_perc != np.ndarray: self.load_files()
		else: self.predicted = predicted_perc

		self.nrows,self.nchannels = self.predicted.shape
		self.make_indices()
		self.predicted2data()
		self.select_predicted_artifact_info()


	def __str__(self):
		m = 'name:\t\t\t' + str(self.name) + '\n'
		m += 'model_name:\t\t' + str(self.model_name) + '\n'
		m += 'window_length:\t\t' + str(self.window_length) + '\n'
		m += 'output_filename:\t' + str(self.output_filename) + '\n'
		m += 'nrows:\t\t\t' + str(self.nrows) + '\n'
		m += 'ground_tuth:\t\t' + str(self.ground_truth_present) + '\n'
		m += 'n_artifact:\t\t' + str(self.n_artifact) + '\n'
		m += 'mcc:\t\t\t' + str(self.mcc) + '\n'
		return m

	def __repr__(self):
		return self.name + '\tnrows: ' + str(self.nrows) + '\tnartifact: ' + str(self.n_artifact) + '\tmcc: ' + str(self.mcc)
		


	def load_files(self):
		'''Load ground truth predicted and class files of a block corresponding 
		to the provided name.
		Fails if predicted perc / predicted class are not available
		'''
		self.ground_truth_filename = path.channel_artifact_training_data+ self.name + '_info.npy'
		self.predicted_filename = path.channel_snippet_annotation + self.model_name + '_' + self.name + '_perc.npy'
		self.pc_filename = path.channel_snippet_annotation + self.model_name + '_' + self.name + '_class.npy'

		try:
			self.predicted = np.load(self.predicted_filename)
			self.pc= np.load(self.pc_filename)
		except: raise ValueError('file not found: ',self.predicted_filename,self.pc_filename)

		if os.path.isfile(self.ground_truth_filename):
			self.ground_truth_present = 'available'
			self.handle_annotated_block = True 
			self.ground_truth = np.load(self.ground_truth_filename)
			self.ground_truth = (self.ground_truth >= .5).astype(int)
			flatten_ground_truth = np.ravel(self.ground_truth,'F')
			flatten_pc = np.ravel(self.pc,'F')
			self.confusion_matrix = sklearn.metrics.confusion_matrix(flatten_ground_truth,flatten_pc)#, labels = [0,1])
			self.n_artifact = np.sum(self.pc)
			cm = plot_roc.confusion_matrix(data = self.confusion_matrix)
			self.f1,self.mcc,self.recall,self.precision,self.fpr,self.cm = cm.f1,cm.mcc,cm.recall,cm.precision,cm.fpr, cm
		else: 
			self.ground_truth_present = 'not available'
			self.ground_truth = None
			self.confusion_matrix = None
			self.f1,self.mcc,self.recall,self.precision,self.fpr,self.cm = [None]*6

		self.nrows,self.nchannels = self.predicted.shape
		

	def make_indices(self):
		'''Return list of [start_index,end_index] which slices up the nrows into window_length nsamples batches.'''
		self.start_indices = np.arange(0,self.nrows,1)
		self.end_indices = self.start_indices + self.window_length 
		# self.predict_goal_indices = self.start_indices + int((self.window_length -1) /2)
		# self.gt_goal_indices = copy.copy(self.start_indices) maybe wrong? is not used
		self.indices = list(zip(range(self.nrows),self.start_indices, self.end_indices ))

	def pad_predicted_column(self,column):
		'''Padd zeros to for datapoints with not enough preceding or superceding datapoints (< (windowlength-1) /2).
		'''
		self.padding = int((self.window_length -1) / 2)
		self.padded_predicted_column = np.pad(column,self.padding,'constant')

	def predicted2data(self):
		'''Make dataset from perc artifact (nindicesXnchannels) to list (nchannels) of np arrays with predicted artifact data'''
		self.data = np.zeros((self.nrows , self.nchannels,self.window_length))
		self.info= np.zeros((self.nrows , self.nchannels,self.window_length))
		for column in range(self.nchannels):
			self.pad_predicted_column(column = self.predicted[:,column])
			self.make_indices()
			for row,start_index, end_index in self.indices:
				self.data[row,column,:] = self.padded_predicted_column[start_index:end_index]

	def select_predicted_artifact_info(self):
		'''Select the subset that is predicted to be artifact, with a score > 0.5
		make list (nchannels) of np arrays with predicted artifact data (and info if present)
		'''

		self.predicted_artifact_indices = []
		self.predicted_artifact_data = []
		self.predicted_artifact_info = []
		for column in range(self.nchannels):
			self.predicted_artifact_indices.append( np.where(self.predicted[:,column] >= .5)[0] )
			self.predicted_artifact_data.append( self.data[self.predicted_artifact_indices[-1],column,:] )
			if self.handle_annotated_block:
				self.predicted_artifact_info.append( self.ground_truth[self.predicted_artifact_indices[-1],column] )

	def save(self):
		'''Save the predicted perc dataset np array to a data file. Obsolete (only use predicted artifacts).'''
		np.save(self.output_filename + '_data',self.data)

	def save_predicted_artifacts(self,overwrite = False,verbose = True):
		'''Save predicted artifacts dataset and the ground truth to np data files.
		The datasets are lists of np.arrays, an array for each channels, thay are flattened
		into 1d np array before saving.
		'''
		if not overwrite and os.path.isfile(self.output_filename +'_artifact-data.npy'): 
			print('Files already saved, doing nothing, use overwrite TRUE to overwrite the files')
			return False
		if verbose:
			print('saving predicted artifact data and info to:',self.output_filename)
		np.save(self.output_filename + '_artifact-data',np.concatenate(self.predicted_artifact_data))
		if self.handle_annotated_block:
			np.save(self.output_filename + '_artifact-info',np.concatenate(self.predicted_artifact_info))
		else:
			print('\nWARNING:\nCannot save info file to corresponding data file because no ground truth file was found. Check whether there is an info file for the current block in the channel_artifact_training_data folder.\n',self.name)
		return True

def get_names_output_files(model_name = 'unk'):
	'''Get the filenames of the perc files, which are output from a cnn model.

	model_name  	name of the cnn model to generate the output files.
	'''
	fn = glob.glob(path.channel_snippet_annotation + model_name + '_pp*class.npy')
	names = [f.split('part-90_')[-1].split('_class.npy')[0] for f in fn]
	return names

def save_all_predicted_artifacts(model_name = 'unk', window_length = 201,overwrite = False):
	'''Save all predicted artifacts of all annotated eeg block files.

	model_name 	 	model used to generate output files
	window_length 	size of context to of the perc_artifact output
	overwrite 		whether to overwrite existing predicted artifact files
	'''
	names = get_names_output_files(model_name)
	bar = pb.ProgressBar()
	bar(range(len(names)))
	for i,name in enumerate(names):
		bar.update(i)
		output_filename = name2output_name(name,model_name)
		if not overwrite and os.path.isfile(output_filename +'_artifact-data.npy'): 
			print('Skipping name:',name,'artifact-data file already exists, use overwrite TRUE to overwrite')
			continue
		d = cnn_ch_output2data(name = name, window_length = window_length)
		d.save_predicted_artifacts(overwrite = overwrite,verbose =False)

def name2output_name(name, model_name = 'unk'):
	'''Generate an output name based on the name of an eeg block and the cnn model name used to create predictions.'''
	output_filename = path.channel_cnn_output_data + model_name + '_' + name
	return output_filename
		
def make_all_data(window_length = 201, overwrite = False):
	'''Generates all predicted artifact files based on model cnn output predictions and combines these into one dataset
	and info np array.

	window_le... 	size of context to of the perc_artifact output
	overwrite 		whether to overwrite existing predicted artifact files
	'''

	# create cnn_output format per block file
	names = get_names_output_files()
	save_all_predicted_artifacts(window_length = window_length, overwrite = overwrite)
	bar = pb.ProgressBar()
	bar(range(len(names)))

	# concatenate cnn_output format files into one data structure
	for i,name in enumerate(names):
		bar.update(i)
		output_filename = name2output_name(name)
		if i == 0:
			data = np.load(output_filename + '_artifact-data.npy')
			info = np.load(output_filename + '_artifact-info.npy')
		else:
			data = np.concatenate((data,np.load(output_filename + '_artifact-data.npy')))
			info = np.concatenate((info,np.load(output_filename + '_artifact-info.npy')))

	not_other_indices = np.where(info != 2)[0]

	if remove_other:
		data = data[not_other_indices,:]
		info = info[not_other_indices]
	return data,info
		
	
def save_training_test(data,info):
	'''Save training and test partitions of the combined predicted artifact data and info from all annotated files.'''
	artifacts_indices = list(np.where(info == 1)[0])
	clean_indices = list(np.where(info == 0)[0])

	nartifacts = len(artifacts_indices)
	nclean = len(clean_indices)
	ntrain_artifacts = nartifacts - int(nartifacts/10)
	ntrain_clean= nclean- int(nclean/10)
	ntest_artifacts = nartifacts - ntrain_artifacts 
	ntest_clean= nclean - ntrain_clean
	print('nartifacts:',nartifacts,'nclean:',nclean,'\nntrain_artifacts:',ntrain_artifacts,'ntrain_clean:',ntrain_clean)
	
	print(artifacts_indices[:20],'\n\n',clean_indices[:20])
	random.shuffle(artifacts_indices)
	random.shuffle(clean_indices)
	print(artifacts_indices[:20],'\n\n',clean_indices[:20])

	train_artifacts_data = data[artifacts_indices[:ntrain_artifacts],:]
	test_artifacts_data = data[artifacts_indices[ntrain_artifacts:],:]

	train_clean_data = data[clean_indices[:ntrain_clean],:]
	test_clean_data = data[clean_indices[ntrain_clean:],:]

	np.save(path.cnn_output_data +'train_artifacts_data',train_artifacts_data)
	np.save(path.cnn_output_data +'test_artifacts_data',test_artifacts_data)
	np.save(path.cnn_output_data +'train_clean_data',train_clean_data)
	np.save(path.cnn_output_data + 'test_clean_data',test_clean_data)

	
		
