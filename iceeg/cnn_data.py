import numpy as np
import path
import random
import windower

class cnn_data:
	'''Object to handle i/o for cnn training.'''
	def __init__(self,fold = 1,load_data_now = False, nfolds = 10, nparts =100, artifact_dir = None, clean_dir = None, small_test_set_size = 500):
		'''Initialize data object that handles loading of training and test files for cnn training.

		fold 			fold of the data object, if multiple data objects are use it is possible to start in a higher fold
		load_dat... 	whether to immediately load training data and small test set
		nfolds 			number of folds to divide the training and test data in
		nparts 			number of data parts to use. Because of file sizes the data was split into 100 clean and artifact files
						it is possible to use a subset, if it is set to n, files upto and including n will be used
		artifact... 	the directory to load artifact datafiles from, default is artifact_training_data
		clean... 		the directory to load clean datafiles from, default is artifact_training_data
		'''

		if artifact_dir == None: self.artifact_dir = path.artifact_training_data
		else: self.artifact_dir = artifact_dir
		if clean_dir == None: self.clean_dir = path.artifact_training_data + 'PART_INDICES/'
		else: self.clean_dir = clean_dir

		self.fold = fold
		self.nfolds = nfolds
		self.nparts = nparts
		self.small_test_set_size = small_test_set_size

		self.all_train_test_indices = train_test_folds_x(nfolds,nparts)
		self.all_folds_filenames = train_test_indices2filenames(self,self.all_train_test_indices)

		self.load_next_fold()

		self.ntrain_artifact_filenames = len(self.train_artifact_filenames)
		self.ntrain_clean_filenames = len(self.train_clean_filenames)
		
		self.ntest_artifact_filenames = len(self.test_artifact_filenames)
		self.ntest_clean_filenames = len(self.test_clean_filenames)

		# self.current_part_index = 0
		if load_data_now:
			self.load_next_training_part()
			self.load_small_test_set(self.small_test_set_size)


	def __str__(self):
		m = '\ncnn_data\n--------\n'
		m += 'nfolds\t\t\t'+str(self.nfolds) + '\n'
		m += 'nparts\t\t\t'+str(self.nparts) + '\n'
		if hasattr(self,'current_part_index'):
			m += 'current part\t\t\t'+str(self.current_part_index +1) + '\n'
		else:
			m += 'current part\t\t1\n'
		m += 'current fold\t\t'+str(self.fold) + '\n'
		if hasattr(self,'artifact_train'):
			m += 'artifact train nepochs\t'+str(self.artifact_train.shape[0]) + '\n'
		else: m += 'artifact train nepochs\t not loaded\n'
		if hasattr(self,'clean_train'):
			m += 'artifact clean nepochs\t'+str(self.clean_train.shape[0]) + '\n'
		else: m += 'clean train nepochs\t not loaded\n'
		return m
		

	def clean_up(self):
		'''Remove data and part indices from the object, resets object to initial state and frees up space.'''
		if hasattr(self,'artifact_test'):
			delattr(self,'artifact_test')
			delattr(self,'clean_test')
			print('removed test files')
		if hasattr(self,'artifact_train'):
			delattr(self,'artifact_train')			
			delattr(self,'clean_train')			
			print('removed training data')
		if hasattr(self,'current_part_index'): 
			delattr(self,'current_part_index')
			print('removed training part index')
		if hasattr(self,'current_test_part_index'): 
			delattr(self,'current_test_part_index')
			print('removed testing part index')
		if hasattr(self,'d'): 
			delattr(self,'d')
		print('data object has been reset, fold number is:',self.fold)

	def load_next_training_part(self,remove_testing_data= True):
		'''Load the next artifact and clean data files for training.
		files where split into parts to decrease size per file.

		remove_testing_data 	removes the test data if present to free up RAM
		'''
		if not hasattr(self,'current_part_index'): self.current_part_index = 0
		elif self.current_part_index >= len(self.train_artifact_filenames)-1:
			print('Already at last index, no more training files.')
			return False
		else: 
			self.current_part_index +=1
		self.part = self.current_part_index + 1

		if remove_testing_data and hasattr(self,'artifact_test'):
			delattr(self,'artifact_test')
			delattr(self,'clean_test')
			print('removed test files')
		self.artifact_filename = self.train_artifact_filenames[self.current_part_index]
		self.clean_filename = self.train_clean_filenames[self.current_part_index]

		self.artifact_train= np.load(self.artifact_filename)
		self.clean_train= np.load(self.clean_filename)
		print('loaded next training files',self.artifact_filename,self.clean_filename)
		return True


	def load_small_test_set(self):
		'''Load a small set of clean and artifact epochs for testing purposes during training.'''
		f = self.test_artifact_filenames[0]
		f = '/'.join(f.split('/')[:-1])+ '/small-testset_' + f.split('/')[-1]
		self.small_testset_artifact_filename = f

		f = self.test_clean_filenames[0]
		f = '/'.join(f.split('/')[:-1])+ '/small-testset_' + f.split('/')[-1]
		self.small_testset_clean_filename = f

		self.small_artifact_test = np.load(self.small_testset_artifact_filename)
		self.small_clean_test = np.load(self.small_testset_clean_filename)


	def load_next_test_part(self,remove_training_data = True):
		'''Load the next artifact and clean data files for testing.
		files where split into parts to decrease size per file.
		'''
		if not hasattr(self,'current_test_part_index'): 
			print('first test, setting current_test_part_index to 0')
			self.current_test_part_index = 0
		elif self.current_test_part_index >= len(self.test_artifact_filenames)-1:
			print('Already at test last index, no more test files.')
			return False
		else: 
			self.current_test_part_index +=1
			print('Testing part:',self.current_test_part_index + 1,'setting current_test_part_index to:',self.current_test_part_index)

		if remove_training_data and hasattr(self,'artifact_train'):
			delattr(self,'artifact_train')			
			delattr(self,'clean_train')			
			print('removed training data')
		self.test_artifact_filename = self.test_artifact_filenames[self.current_test_part_index]
		self.test_clean_filename = self.test_clean_filenames[self.current_test_part_index]

		self.artifact_test = np.load(self.test_artifact_filename)
		self.clean_test = np.load(self.test_clean_filename)
		print('loaded next test files',self.test_artifact_filename,self.test_clean_filename)
		return True

	def load_next_fold(self):
		'''Loads next fold of nfold training regime. It could be better to create a data object for each fold.'''
		if not hasattr(self,'fold_index'): self.fold_index = self.fold -1
		elif self.fold == self.nfolds:
			print('Already at test last fold, no more folds.')
			return False
		else:
			self.fold += 1
			self.fold_index = self.fold -1
		self.current_fold_filenames = self.all_folds_filenames[self.fold_index]
		self.train_artifact_filenames = self.current_fold_filenames[0]
		self.train_clean_filenames = self.current_fold_filenames[1]
		self.test_artifact_filenames = self.current_fold_filenames[2]
		self.test_clean_filenames = self.current_fold_filenames[3]
		return True


	def block2eegdata(self,b):
		'''Load the 100hz eeg data that corresponds to the block object and returns a windowed version of it, identical to
		the method used in make_artifact_matrix_v2.py.'''
		self.clean_up()
		self.d = load_100hz_numpy_block(windower.make_name(b))
		if b.start_marker_missing or b.end_marker_missing:
			w = windower.Windower(b,nsamples= self.d.shape[1], sf = 100,window_overlap_percentage = .99)
		else:
			w = windower.Windower(b,sf = 100,window_overlap_percentage = .99)
		self.d = remove_channels(self.d)
		self.d = windower.window_data(self.d,w.windows['sf100'],flatten=True,normalize= True)


def train_test_folds_x(nfolds=10,parts=100):
	'''Set part number for all data files and split the files into training and test sets for each fold.

	nfolds 		number of folds to create
	parts 		number of datafiles to use. The data was split in 100 parts. 
				If you specify n parts, part files upto and including n will be used
	'''
	if parts > 100:
		print('only 100 parts were created, setting nparts to 100')
		parts = 100
	train_test_indices = []
	for i in range(nfolds):
		test_ul = (i+1) * int(parts / nfolds)
		test_ll = i * int(parts/ nfolds)
		train,test = [],[]
		for n in range(1,parts+1):
			if test_ll < n <= test_ul:
				test.append(n)
			else: train.append(n)
		train_test_indices.append([train,test])
	return train_test_indices


def train_test_indices2filenames(self,train_test_indices):
	'''Create filenames for all artifact and clean data parts.'''
	folds_filenames = []
	for train, test in train_test_indices:
		train_artifact_filenames,train_clean_filenames,test_artifact_filenames,test_clean_filenames = [],[],[],[]
		for i in train:
			train_artifact_filenames.append(self.artifact_dir + 'data_artifacts_part-'+str(i) + '.npy')
			train_clean_filenames.append(self.clean_dir + 'data_clean_part-'+str(i) + '.npy')
		for i in test:
			test_artifact_filenames.append(self.artifact_dir + 'data_artifacts_part-'+str(i) + '.npy')
			test_clean_filenames.append(self.clean_dir + 'data_clean_part-'+str(i) + '.npy')
		folds_filenames.append([train_artifact_filenames,train_clean_filenames,test_artifact_filenames,test_clean_filenames])
	return folds_filenames


def load_100hz_numpy_block(name):
	return np.load(path.eeg100hz + name + '.npy')

def remove_channels(d):
	ch_names = open(path.data + 'channel_names.txt').read().split('\n')
	remove_ch = ['Fp2','VEOG','HEOG','TP10_RM','STI 014','LM']
	'''remove eeg channels, by default the reference and eog channels.'''
	print('removing channels:',remove_ch)
	ch_mask = [n not in remove_ch for n in ch_names]
	ch_names= [n for n in ch_names if not n in remove_ch]
	d= d[ch_mask,:]
	nchannels = len(ch_names)
	print('remaining channels:',ch_names,'nchannels:',nchannels,'data shape:',d.shape)
	return d
