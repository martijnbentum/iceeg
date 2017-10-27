import a2cep
import AUTOF
import block
import copy
import load_all_ort
from matplotlib import pyplot as plt
import numpy as np
import os
import path
# from scipy import stats
import session
from sklearn.decomposition import PCA
import pickle
import py_lpc
from scipy import signal
import xml_handler

#based on Garbage stats
class Data_stats:
	'''Object to hold statistics about windowed eeg data.'''

	def __init__(self,block = None,length = 0.5,remove_ch= None, window_overlap = True, window_overlap_percentage = 0.5, lpc_order = 10, coder = 'martijn'):
		'''Create object to hold statistics about windowed eeg data.
		data is windowed with hamming window and for now the LPC cepstrum coefficients are computed for each window and channel.
		
		block 						block object of an participants of a specific experimental session
		length 						the length of a window in seconds
		remove_ch 					whether to remove channels, default = eog and reference channels
		window_overlap 				whether to use overlapping windows
		window_overlap_percentage 	the percentage of overlap between consecutive windows
		lpc_order 					number of lpc coefficients requested (returns + 1)
		coder 						name of the coder of the label files (will only return artifact files from that coder
		'''
		self.annot2int = {'clean':0,'garbage':1,'unk':2,'drift':3,'other':4}
		if block == None or type(block) == str:  
			self.block = block
			return None
		self.block = block
		self.length = int(float(length) * 1000)
		self.window_overlap = window_overlap
		self.window_overlap_percentage = window_overlap_percentage
		self.fn_data_stats = make_filename(block)
		self.lpc_order = lpc_order
		self.coder = coder
		if remove_ch != None and type(remove_ch) == list: self.remove_ch = remove_ch
		else: self.remove_ch = ['VEOG','HEOG','TP10_RM','STI 014','LM']
		self.load_data()
		self.remove_channels()
		self.set_start_end_snippets()
		self.hamming = signal.hamming(self.length)
		self.window_data()
		self.lpc(self.lpc_order)
		self.set_info()
		self.fn_annotation = find_annotation_file(self.pp_id,self.exp_type,self.bid, self.coder)


	def set_info(self, other = None):
		'''Sets experiment and participant info on object.'''
		if other == None:
			self.pp_id = self.block.pp_id
			self.exp_type = self.block.exp_type
			self.bid = self.block.bid
			self.st_sample = self.block.st_sample
		else:
			other.pp_id = self.pp_id
			other.exp_type = self.exp_type
			other.bid = self.bid
			other.st_sample = self.st_sample
	

	def save_data_stats(self):
		'''Pickle the object removes original eeg data and only keep the lpc-cepstrum coefficients to reduce data size.''' 
		output = Data_stats()
		self.set_info(output)
		output.a_coef = copy.deepcopy(self.lpc_coefficients)
		changedif self.use_autof:
			output.cepstrum_coef = copy.deepcopy(self.cepstrum_coef)
		output.length = self.length
		output.duration_sample = self.duration_sample
		output.start_snippets = self.start_snippets
		output.end_snippets = self.end_snippets
		output.nsnippets = self.nsnippets
		output.nchannels = self.nchannels
		output.ch_names = self.ch_names
		output.window_overlap = self.window_overlap
		output.overlap_percentage = self.window_overlap_percentage
		output.data = None
		output.windowed_data = None
		fout = open(self.fn ,'wb')
		pickle.dump(output,fout,-1)


	def load_block(self, fo = None):
		'''Load block object based on participant and experimental session info.'''	
		if fo != None: self.fo = fo
		if not hasattr(self,'pp_id') or not hasattr(self,'exp_type') or not hasattr(self,'bid'):
			print('Cannot load block without pp info and block info')
			return 0
		if self.fo == None: 
			print('Supply fidort to speed up block loading. Be carefull that using the same fidort for multiple blocks is not a good idea if you want to use word info.')
			self.fo = load_all_ort.load_fid2ort()
		s = session.Session(self.pp_id, self.exp_type, fid2ort = self.fo)
		self.block = getattr(s,'b' + str(self.bid))


	def load_data(self):
		'''Load eeg data corresponding to block.'''
		if block == None or not hasattr(self,'block'): self.load_block()
		if not hasattr(self.block,'raw'):
			self.block.load_eeg_data()
		self.ch_names = self.block.raw.ch_names
		self.nchannels = len(self.ch_names)
		self.data = self.block.raw[:][0] * 10 ** 6
		self.st_sample = self.block.raw.first_samp
		self.duration_sample = len(self.block.raw)
		self.raw_ch_names = self.block.raw.ch_names
		del self.block.raw


	def remove_channels(self,channels = []):
		'''remove eeg channels, by default the reference and eog channels.'''
		self.remove_ch += channels
		self.ch_mask = [n not in self.remove_ch for n in self.ch_names]
		self.ch_names= [n for n in self.raw_ch_names if not n in self.remove_ch]
		self.data = self.data[self.ch_mask,:]
		self.nchannels = len(self.ch_names)
		

	def set_start_end_snippets(self):
		'''Create indices that mark the start and end of the sliding window.
		Overlap sets whether to overlap between adjacent windows and overlap_percentage
		set the amount overlap.
		All snippets < length will be deleted
		'''
		if self.window_overlap:
			self.start_snippets = np.arange(0,self.duration_sample,int(self.length*self.window_overlap_percentage))
		else:
			self.start_snippets = np.arange(0,self.duration_sample,self.length)
		self.end_snippets = self.start_snippets + self.length
		# minimum snippet length = should be of length snippet 
		bad_indices = []
		for i in list(range(len(self.start_snippets)))[::-1]:
			if self.end_snippets[i] - self.start_snippets[i] < self.length:
				bad_indices.append(i)
			elif self.end_snippets[i] > self.duration_sample: 
				bad_indices.append(i)
			else: break
		for i in bad_indices:
			self.start_snippets = np.delete(self.start_snippets,i)
			self.end_snippets = np.delete(self.end_snippets,i)
		self.nsnippets = len(self.start_snippets)


	def window_data(self):
		'''Multiply a hamming window with the windowed data.
		Length determines length of the windows, overlap whether windows overlap
		and overlap_percentage the amount of overlap
		Data is stored in np matrix of size number of window X number of channels 
		X length window.
		'''
		self.windowed_data = np.zeros((self.nsnippets,self.nchannels,self.length))
		for i in range(self.nsnippets):
			start,end = self.start_snippets[i],self.end_snippets[i]
			# print(i,start,end)
			self.windowed_data[i] = self.hamming * self.data[:,start:end]

	def lpc(self,order = 15, use_autof = True):
		'''Uses linear predictive coding to extract coefficients describing spectrum
		of window, can be used as features for training classifier

		if use_autof == False
		Based on py_lpc: 
		https://github.com/cournape/talkbox/tree/master/scikits/talkbox/linpred
		This is a slow implementation, needs to be improved, the scikits module
		contains a c implementation but this works in python2
		
		if use_autof == True
		Based on fortran code provide by Lou Boves
		The code is compiled in to a wrapper python file
		a-parameters, e-parameters (error), k-parameters = AUTOF.autof(signal,order)
		a 			, alpha 			  , k 			 = AUTOF.autof(signal,order)
		cepstra = a2cep.a2cep(a,alpha, a.shape[0])
		convert to cepstra to be able to computer a distance measure between windows
		'''
		
		self.a_coef = np.zeros((self.nsnippets,self.nchannels,order + 1))
		self.cepstrum_coef = np.zeros((self.nsnippets,self.nchannels,order + 2))
		for wi,window in enumerate(self.windowed_data):
			for ci,channel in enumerate(window):
				if not use_autof:
					self.a_coef[wi,ci] = py_lpc.lpc_ref(channel,order)
				else:
					self.a_coef[wi,ci], alpha, k = AUTOF.autof(channel,order) 
					self.cepstrum_coef[wi,ci] = a2cep.a2cep(self.a_coef[wi,ci],alpha,order+1)
				

	def load_annotations(self):
		'''load the labels corresponding to this block in the bad_epoch filed  and order the bad_epochs based on start time.'''
		self.annotations = xml_handler.xml_handler(filename = self.fn_annotation)
		self.bad_epochs = self.annotations.xml2bad_epochs() 
		self.bad_epochs.sort()
		if len(self.bad_epochs) > 0:
			be_block_st = self.bad_epochs[0].block_st_sample
			if self.st_sample != be_block_st:
				raise ValueError('Start sample block datastats', be_block_st, 'does not equal start sample bad_epoch',self.st_sample)
		else: print('No bad epochs in this block.')

				
	def make_info_matrix(self, default = 'clean'):
		'''create a np matrix with the type of artefact and the amount of overlap between bad_epoch and a window.
		row indices correspond of the info matrix correspond with the row indices of the windowed data
		Column indices correspond with the annot2int dict, the overlap columns are a second set of columns besides this.
		'''
		if default not in self.annot2int.keys(): 
			default = 'other'
			print(default,'not in annot2int keys',self.annot2int,'will set the default annot to other.')
		if not hasattr(self,'annotations'): self.load_annotations()
		start_overlap_column = len(self.annot2int)
		clean = np.zeros(len(self.annot2int)*2)
		clean[self.annot2int[default]] = 1.0
		clean[start_overlap_column + self.annot2int['clean']] = 1.0
		self.info_matrix = np.zeros((len(self.start_snippets),len(self.annot2int)*2 ))
		for be in self.bad_epochs:
			annotation = be.annotation if be.annotation in self.annot2int.keys() else 'other'
			indices,overlap = find_snippet_index_and_overlap_bad_epoch(self.start_snippets,self.end_snippets,be)			
			for i, index in enumerate(indices):
				self.info_matrix[index,self.annot2int[annotation]] = 1
				self.info_matrix[index,start_overlap_column + self.annot2int[annotation]] = overlap[i]
		for i,line in enumerate(self.info_matrix):
			if max(line) == 0:
				self.info_matrix[i,:] = clean

def load_data_stats(block = None, fn = None, fo = None):
	'''Load object to file with name == eeg file and extension .data_stats in path.data_stats folder.'''
	if type(block)== str:
		fn = copy.copy(block)
		block = None
	if block: fn = path.data_stats + block.vmrk.vmrk_fn.split('/')[-1].strip('.vmrk') + '_' + str(block.marker) + '.data_stats' 
	elif fn == None: print('Need filename or block object to load data stats')
	if not os.path.isfile(fn):
		print('File does not excist')
		return 0
	fin = open(fn,'rb')
	output = pickle.load(fin)
	if block: output.block = block
	else: output.block = None
	output.fo = fo
	return output
	

def make_filename(b):
	'''create datastat filename based on participant and experiment info.'''
	return path.data_stats + 'pp'+str(b.pp_id) + '_' + b.exp_type + '_b' + str(b.bid) + '_m' + str(b.marker) + '.data_stats'


def find_annotation_file(pp_id,exp_type,bid, coder = 'martijn'):
	'''Check wheter there is a annotation file and return filename if there is.
	'''
	f = path.artifacts + coder +'_pp' + str(pp_id) + '_exp-' + exp_type + '_bid-' +str(bid) + '.xml'
	if os.path.isfile(f): return f
	else: 
		print('file:',f,'not found')
		return 0
	

def compute_overlap(start_a,end_a,start_b, end_b):
	'''compute the percentage b overlaps with a.
	if overlap = 1, b is equal in length or larger than a and start before or at the same time as a and
	b end later or ate the same time as a.
	'''
	if end_a < start_a:
		raise ValueError('first interval is invalid, function assumes increasing intervals',start_a,end_a)
	if end_b < start_b:
		raise ValueError('second interval is invalid, function assumes increasing intervals',start_b,end_b)
	if end_b < start_a or start_b > end_a: return 0 # b is completely before or after a
	elif start_a == start_b and end_a == end_b: return end_a - start_a # a and b are identical
	elif start_b < start_a: # first statement already removed b cases completely before a
		if end_b < end_a: return end_b - start_a # b starts before a and ends before end of a	
		else: return end_a - start_a # b starts before a and ends == or after end of a
	elif start_b < end_a: # first statement already romve b cases completely after a
		if end_b > end_a: return end_a - start_b # starts after start of a and ends == or after end of a
		else: return end_b - start_b  # b starts after start of a and ends before end of a #
	else:  print('error this case should be impossible')

def find_snippet_index_and_overlap_bad_epoch(starts,ends,be):
	'''Find the snippet interval with which the bad epoch overlaps.
	return the index for each interval
	return the percof overlap between the snippet interval and the bad_epoch (n_overlap_samples / n_samples_snippet_interval
	'''
	found = False
	indices, overlaps = [], []
	for i,start in enumerate(starts):
		end = ends[i]
		overlap = compute_overlap(start,end,be.start.x,be.end.x)
		if overlap > 0:
			indices.append(i)
			overlaps.append(overlap / (end - start) )
	return indices, overlaps


def reshape_cepstrum_matrix(ceptrum_matrix):
	'''Set all ceptrum coefficients of a window in a single vector (collapses the channel dimension).'''
	nrows = ceptrum_matrix.shape[0]
	ncolumns = ceptrum_matrix.shape[1] * ceptrum_matrix.shape[2]
	return np.reshape(ceptrum_matrix,(nrows,ncolumns)) 


def reduce_column_dimension(ceptrum_matrix,n_components = 10):
	'''Use sklearn pca to reduce dimensionality of ceptrum matrix.'''
	print('using pca to reduce dimension in column direction, a lot of repitition due to ceptra of correlated channels, from the same time window. Consider calling pca when all data from all blocks are aggregated')
	if len(ceptrum_matrix) > 2:
		print('Ceptrum matrix should be timewindow X (nchannels*lpc_order+2), a 2d matrix, calling reshape')
		ceptrum_matrix = reshape_cepstrum_matrix(ceptrum_matrix)
	print('shape matrix:',ceptrum_matrix.shape)
	pca = PCA(n_components = n_components)
	pca.fit(ceptrum_matrix)
	print('Perc explained variance per component:',pca.explained_variance_ratio_)
	return pca.fit_transform(ceptrum_matrix)
	

