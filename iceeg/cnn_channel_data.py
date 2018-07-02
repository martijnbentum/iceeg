import numpy as np
import path
import random
import windower

class cnn_data:
	'''Object to handle i/o for cnn training.'''
	def __init__(self,fold = 1,load_data_now = False, nfolds = 10, nparts =100, artifact_dir = None, clean_dir = None, small_test_set_size = 500,nchannels = 26, kernel_size = 6):
		'''Initialize data object that handles loading of training and test files for cnn training.

		fold 			fold of the data object, if multiple data objects are use it is possible to start in a higher fold
		load_dat... 	whether to immediately load training data and small test set
		nfolds 			number of folds to divide the training and test data in
		nparts 			number of data parts to use. Because of file sizes the data was split into 100 clean and artifact files
						it is possible to use a subset, if it is set to n, files upto and including n will be used
		artifact... 	the directory to load artifact datafiles from, default is artifact_training_data
		clean... 		the directory to load clean datafiles from, default is artifact_training_data
		'''

		if artifact_dir == None: self.artifact_dir = path.channel_artifact_training_data + 'PART_INDICES/'
		else: self.artifact_dir = artifact_dir

		self.nchannels = nchannels
		self.kernel_size = kernel_size

		self.fold = fold
		self.nfolds = nfolds
		self.nparts = nparts
		self.small_test_set_size = small_test_set_size

		self.all_train_test_indices = train_test_folds_x(nfolds,nparts)
		self.all_folds_filenames = train_test_indices2filenames(self,self.all_train_test_indices)

		self.load_next_fold()


		self.ntrain_data_filenames = len(self.train_data_filenames)
		self.ntest_data_filenames = len(self.test_data_filenames)

		# self.current_part_index = 0
		if load_data_now:
			self.load_next_training_part()
			self.load_small_test_set(self.small_test_set_size)


	def __str__(self):
		m = '\ncnn_data\n--------\n'
		m += 'nfolds\t\t\t'+str(self.nfolds) + '\n'
		m += 'nparts\t\t\t'+str(self.nparts) + '\n'
		if hasattr(self,'current_part_index'):
			m += 'current part\t\t'+str(self.current_part_index +1) + '\n'
		else:
			m += 'current part\t\t1\n'
		m += 'current fold\t\t'+str(self.fold) + '\n\n'
		if hasattr(self,'train_data'):
			m += 'artifact train nepochs\t'+str(self.train_nartifact) + '\n'
			m += 'clean train nepochs\t'+str(self.train_nclean) + '\n'
			m += 'train perc_artifact\t'+str(self.train_artifact_perc) + '\n'
			m += 'train filename\t\t'+str(self.train_data_filename) + '\n'
		else: m += 'train data\t\tnot loaded\n'
		if hasattr(self,'test_data'):
			m += 'artifact test nepochs\t'+str(self.test_nartifact) + '\n'
			m += 'clean test nepochs\t'+str(self.test_nclean) + '\n'
			m += 'test perc_artifact\t'+str(self.test_artifact_perc) + '\n'
			m += 'test filename\t\t'+str(self.test_data_filename) + '\n'
		else: m += 'test data\t\tnot loaded\n'
		return m
		

	def clean_up(self):
		'''Remove data and part indices from the object, resets object to initial state and frees up space.'''
		if hasattr(self,'data_test'):
			delattr(self,'data_test')
			delattr(self,'info_test')
			print('removed test files')
		if hasattr(self,'data_train'):
			delattr(self,'data_train')			
			delattr(self,'info_train')			
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


	def set_clean_artifact_indices(self,name = 'train'):
		'''Set artifact and clean indices for the data based on info for the type specified by name.

		name the type of data: train test smalltest
		'''
		self.chindex_dict = make_ch_index_dict(nchannels = self.nchannels,kernel_size = self.kernel_size)
		info = getattr(self,name + '_info')
		setattr(self,name+ '_info',  (info>=0.5).astype(int) )
		si,chi= np.where(getattr(self,name + '_info')== 0) 
		setattr(self,name+ '_clean_sample_indices', si )
		setattr(self,name+'_clean_ch_indices' , np.array([self.chindex_dict[i] for i in chi]) )
		si,chi = np.where(getattr(self,name + '_info')== 1)
		setattr(self,name+ '_artifact_sample_indices', si )
		setattr(self,name+'_artifact_ch_indices' , np.array([self.chindex_dict[i] for i in chi]) )
		setattr(self,name+ '_artifact_perc' , len(getattr(self,name + '_artifact_sample_indices')) / len(getattr(self,name + '_clean_sample_indices')) )
		setattr(self,name+ '_nartifact' , len(getattr(self,name + '_artifact_sample_indices'))) 
		setattr(self,name+ '_nclean' , len(getattr(self,name + '_clean_sample_indices'))) 


	def load_next_training_part(self,remove_testing_data= True, random_pick = False):
		'''Load the next artifact and clean data files for training.
		files where split into parts to decrease size per file.

		remove_testing_data 	removes the test data if present to free up RAM
		'''
		if random_pick: self.current_part_index = random.sample(range(self.ntrain_data_filenames),1)[0]
		elif not hasattr(self,'current_part_index'): self.current_part_index = 0
		elif self.current_part_index >= len(self.train_data_filenames)-1:
			print('Already at last index, no more training files.')
			return False
		else: 
			self.current_part_index +=1

		self.part = self.current_part_index + 1

		if remove_testing_data and hasattr(self,'test_data'):
			delattr(self,'test_data')
			delattr(self,'test_info')
			print('removed test files')
		self.train_data_filename = self.train_data_filenames[self.current_part_index]
		self.train_info_filename = self.train_data_filename.replace('data','info')

		self.train_data= insert_target_channel_rows(np.load(self.train_data_filename), nchannels = self.nchannels,kernel_size = self.kernel_size)
		self.train_info= np.load(self.train_info_filename)

		self.set_clean_artifact_indices(name = 'train')
		print('loaded next training files',self.train_data_filename,self.train_info_filename)
		if random: print('loaded a random training file.')
		return True


	def load_small_test_set(self):
		'''Load a small set of clean and artifact epochs for testing purposes during training.'''
		f = self.test_data_filenames[0]
		f = '/'.join(f.split('/')[:-1])+ '/smalltest_' + f.split('/')[-1]
		self.small_testset_data_filename = f
		self.small_testset_info_filename = f.replace('data','info')

		self.smalltest_data= insert_target_channel_rows(np.load(self.small_testset_data_filename),nchannels = self.nchannels,kernel_size = self.kernel_size)
		self.smalltest_info= np.load(self.small_testset_info_filename)
		self.set_clean_artifact_indices(name = 'smalltest')


	def load_next_test_part(self,remove_training_data = True,load_next = True,loop_test_files= False):
		'''Load the next artifact and clean data files for testing.
		files where split into parts to decrease size per file.
		'''
		if not hasattr(self,'current_test_part_index'): 
			print('first test, setting current_test_part_index to 0')
			self.current_test_part_index = 0
		elif self.current_test_part_index >= len(self.test_data_filenames)-1:
			print('Already at test last index, no more test files.')
			return False
		elif not load_next: print('keeping test part at:',self.current_test_part_index+1,'the index is at:',self.current_test_part_index)
		else: 
			self.current_test_part_index +=1
			print('Testing part:',self.current_test_part_index + 1,'setting current_test_part_index to:',self.current_test_part_index)

		if remove_training_data and hasattr(self,'train_data'):
			delattr(self,'train_data')			
			delattr(self,'train_info')			
			print('removed training data')
		self.test_data_filename = self.test_data_filenames[self.current_test_part_index]
		self.test_info_filename = self.test_data_filename.replace('data','info')

		self.test_data = insert_target_channel_rows(np.load(self.test_data_filename), nchannels=self.nchannels,kernel_size = self.kernel_size)
		self.test_info= np.load(self.test_info_filename)
		self.set_clean_artifact_indices(name = 'test')
		print('loaded next test files',self.test_data_filename,self.test_info_filename)
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
		self.train_data_filenames = self.current_fold_filenames[0]
		self.test_data_filenames = self.current_fold_filenames[1]
		return True


	def block2eegdata(self,b):
		'''Load the 100hz eeg data that corresponds to the block object and returns a windowed version of it, identical to
		the method used in make_artifact_matrix_v2.py.'''
		self.clean_up()
		self.d = load_100hz_numpy_block(windower.make_name(b))
		if b.start_marker_missing or b.end_marker_missing:
			# the eeg data in d has a sf 100, windower expects an sf 1000, the sf parameter adjust the start and end times of snippets, therefore the nsamples needs to be multiplied by 10.
			w = windower.Windower(b,nsamples= self.d.shape[1] * 10, sf = 100,window_overlap_percentage = .99)
		else:
			w = windower.Windower(b,sf = 100,window_overlap_percentage = .99)
		self.d = remove_channels(self.d)
		self.d = windower.window_data(self.d,w.windows['sf100'],flatten=True,normalize= True)
		self.d = insert_target_channel_rows(self.d)


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
		train_filenames,test_filenames = [],[] 
		for i in train:
			train_filenames.append(self.artifact_dir + 'data_part-'+str(i) + '.npy')
		for i in test:
			test_filenames.append(self.artifact_dir + 'data_part-'+str(i) + '.npy')
		folds_filenames.append([train_filenames,test_filenames])
	return folds_filenames


def load_100hz_numpy_block(name):
	return np.load(path.eeg100hz + name + '.npy')

def remove_channels(d):
	ch_names = open(path.data + 'channel_names.txt').read().split('\n')
	# remove_ch = ['Fp2','VEOG','HEOG','TP10_RM','STI 014','LM']
	remove_ch = ['VEOG','HEOG','TP10_RM','STI 014','LM']
	'''remove eeg channels, by default the reference and eog channels.'''
	print('removing channels:',remove_ch)
	ch_mask = [n not in remove_ch for n in ch_names]
	ch_names= [n for n in ch_names if not n in remove_ch]
	d= d[ch_mask,:]
	nchannels = len(ch_names)
	print('remaining channels:',ch_names,'nchannels:',nchannels,'data shape:',d.shape)
	return d


def insert_target_channel_rows(data,kernel_size = 6, nchannels = 26):
	'''Insert the target channel at regular intervals in the training/test sample.
	The target channel is inserted in such a way that the kernel always sees the target channel once.
	Only at the position where the target channel is originally present does the kernel see
	the target channel twice.
	
	data 		data read in from channel_artifact_training_data, each line contains one sample
				with (default) 26 channels and 100 timepoints, shape is ntraining_samples * 
				(nchannel*nsamples = 2600)
	kernel_size height of the kernel
	nchannels 	number of channels in a sample
	'''
	rows = data.shape[0]	
	indices = list(range(0,nchannels,kernel_size-1))
	output = np.reshape(data,[rows,nchannels,-1])
	nsamples = output.shape[-1]
	output = np.insert(output,indices,np.zeros((nsamples)),axis =1)
	return output


	 
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
	org_indices = np.arange(nchannels)
	new_indices = list(np.insert(org_indices,insert_indices,-1,axis=0))
	for i in range(nchannels):
		chindex_dict[i] = new_indices.index(i)
	return chindex_dict



