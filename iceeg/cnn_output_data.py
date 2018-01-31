import copy
import glob
import min_max_artifact as mma
import numpy as np
import os 
import path
import progressbar as pb
import random
import sklearn.metrics
import windower

class cnn_output_data:
	'''Object to handle i/o for cnn output training.'''
	def __init__(self):
		'''Initialize data object that handles loading of training and test files for cnn output data training.
		the dataset is created with functions in cnn_output2data class
		'''
		self.train_artifact_filename = path.cnn_output_data +'train_artifacts_data.npy'
		self.test_artifact_filename = path.cnn_output_data +'test_artifacts_data.npy'
		self.train_clean_filename = path.cnn_output_data +'train_clean_data.npy'
		self.test_clean_filename = path.cnn_output_data + 'test_clean_data.npy'

		self.artifact_train = np.load(self.train_artifact_filename)
		self.artifact_test = np.load(self.test_artifact_filename)
		self.clean_train = np.load(self.train_clean_filename)
		self.clean_test = np.load(self.test_clean_filename)

		self.small_artifact_test = self.artifact_test[:250]
		self.small_clean_test = self.clean_test[:250]

	def __str__(self):
		m = '\ncnn_output_data\n--------\n'
		m += 'nartifact train:\t\t' + str(self.artifact_train.shape[0]) +'\n'
		m += 'nartifact test:\t\t' + str(self.artifact_test.shape[0]) +'\n'
		m += 'nclean train:\t\t' + str(self.clean_train.shape[0]) +'\n'
		m += 'nclean test:\t\t' + str(self.clean_test.shape[0]) +'\n'
		return m
		

class cnn_output2data:
	'''Create dataset to train a regression model on the predicted artifacts and remove the false positives.
	'''
	def __init__(self,name= '', window_length = 201, model_name = 'rep-3_perc-50_fold-2_part-90',predicted_perc = None):
		'''Initialize data object that handles loading of training and test files for cnn training.
		name 		the name corresponding to predicted and ground truth, format windower.make_name(b) (b = block object) 		
		window_... 	the number of time points for each datapoints. The cnn ouput gives a perc artifact for each each epoch
					but does not take into account the data before or after, this dataset is for removing false positive
					predicted artifacts (the cnn_model has lower precision to achieve high recall of artifacts
		model_n... 	name of the cnn model that preformed the perc artifact predictions
		predi... 	a 1d np array of perc artifacts from cnn model to be adjusted by the cnn_output_model
		'''
		if name == None and list(predicted_perc) != np.ndarray:
			print('provide block_name or predicted_perc artifact')
		self.name = name
		self.window_length = window_length
		self.model_name = model_name
		self.output_filename = path.cnn_output_data + self.model_name + '_' + self.name
		if not predicted_perc != np.ndarray:
			self.load_files()
			self.handle_block = True
		else: 
			self.predicted = predicted_perc
			self.handle_block = False
		self.nrows = self.predicted.shape[0]
		self.make_indices()
		self.pad_predicted()
		self.predicted2data()
		self.select_predicted_artifact_info()

	def load_files(self):
		'''Load ground truth predicted and class files of a block corresponding to the provided name
		should be extended for use with not annotated files (no ground truth).
		'''
		self.ground_truth_filename = path.snippet_annotation + self.name + '.gt_indices.npy'
		self.predicted_filename = path.snippet_annotation + self.model_name + '_' + self.name + '_perc.npy'
		self.pc_filename = path.snippet_annotation + self.model_name + '_' + self.name + '_class.npy'

		self.ground_truth = np.load(self.ground_truth_filename)
		self.predicted = np.load(self.predicted_filename)[:,1]
		self.pc= np.load(self.pc_filename)

		self.confusion_matrix = sklearn.metrics.confusion_matrix(self.ground_truth,self.pc)
		if self.confusion_matrix.shape == (2,2):
			r0,p0,f10,r1,p1,f11,mcc,fpr0,fpr1 =mma.precision_recall(self.confusion_matrix)
			self.f1,self.mcc,self.recall,self.precision,self.fpr = f11,mcc,r1,p1,fpr1
		else: self.f1,self.mcc,self.recall,self.precision,self.fpr = ['NA']*5
		self.nrows = self.predicted.shape[0]
		assert self.nrows == self.predicted.shape[0]
		

	def make_indices(self):
		'''Return list of [start_index,end_index] which slices up the nrows into nsamples batches.'''
		self.start_indices = np.arange(0,self.nrows,1)
		self.end_indices = self.start_indices + self.window_length 
		self.predict_goal_indices = self.start_indices + int((self.window_length -1) /2)
		self.gt_goal_indices = copy.copy(self.start_indices)
		self.indices = zip(range(self.nrows),self.start_indices, self.end_indices )

	def pad_predicted(self):
		'''Padd zeros to for datapoints with not enough preceding or superceding datapoints (< (windowlength-1) /2).
		'''
		self.padding = int((self.window_length -1) / 2)
		self.padded_predicted = np.array([0] * self.padding + list(self.predicted) + [0] * self.padding)

	def predicted2data(self):
		'''Make dataset from perc artifact 1d np array using the indices.'''
		self.data = np.zeros((self.nrows,self.window_length))
		for i,start_index, end_index in self.indices:
			self.data[i,:] = self.padded_predicted[start_index:end_index]

	def select_predicted_artifact_info(self):
		'''Select the subset that is predicted to be artifact, with a score > 0.5'''
		self.predicted_artifact_indices = np.where(self.predicted >= .5)[0]
		self.predicted_artifact_data = self.data[self.predicted_artifact_indices,:]
		if self.handle_block:
			self.predicted_artifact_info = self.ground_truth[self.predicted_artifact_indices]

	def save(self):
		'''Save the predicted perc dataset np array to a data file. Obsolete (only use predicted artifacts).'''
		np.save(self.output_filename + '_data',self.data)

	def save_predicted_artifacts(self,overwrite = False,verbose = True,skip_other = True):
		'''Save predicted artifacts dataset and the ground truth to np data files.'''
		if not overwrite and os.path.isfile(self.output_filename +'_artifact-data.npy'): 
			print('Files already saved, doing nothing, use overwrite TRUE to overwrite the files')
			return False
		if verbose:
			print('saving predicted artifact data and info to:',self.output_filename)
		np.save(self.output_filename + '_artifact-data',self.predicted_artifact_data)
		np.save(self.output_filename + '_artifact-info',self.predicted_artifact_info)
		return True

def get_names_output_files(model_name = 'rep-3_perc-50_fold-2_part-90'):
	'''Get the filenames of the perc files, which are output from a cnn model.

	model_name  	name of the cnn model to generate the output files.
	'''
	fn = glob.glob(path.snippet_annotation + model_name + '_pp*class.npy')
	names = [f.split('part-90_')[-1].split('_class.npy')[0] for f in fn]
	return names

def save_all_predicted_artifacts(model_name = 'rep-3_perc-50_fold-2_part-90', window_length = 201,overwrite = False):
	'''Save all predicted artifacts of all annotated eeg block files.

	model_name 		model used to generate output files
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
		d = cnn_output2data(name = name, window_length = window_length)
		d.save_predicted_artifacts(overwrite = overwrite,verbose =False)

def name2output_name(name, model_name = 'rep-3_perc-50_fold-2_part-90'):
	'''Generate an output name based on the name of an eeg block and the cnn model name used to create predictions.'''
	output_filename = path.cnn_output_data + model_name + '_' + name
	return output_filename
		
def make_all_data(remove_other = True,window_length = 201, overwrite = False):
	'''Generates all predicted artifact files based on model cnn output predictions and combines these into one dataset
	and info np array.

	remove_other 	whether to remove the instances where the coding is other (instead of clean or artifact)
					the other coding was used in eeg block with noisy data, it does not provide info about artifact or clean.
					if an instance has other codings it will not be used, this means no data is used from blocks with
					other coding. (these blocks do not have any clean stretches, only artifact and other)
	window_le... 	size of context to of the perc_artifact output
	overwrite 		whether to overwrite existing predicted artifact files
	'''

	names = get_names_output_files()
	save_all_predicted_artifacts(window_length = window_length, overwrite = overwrite)
	bar = pb.ProgressBar()
	bar(range(len(names)))

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

	
		
