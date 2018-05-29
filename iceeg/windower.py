#based on data_stats.py

# import block
import copy
import glob
import load_all_ort
import numpy as np
import os
import path
import session
import utils
import pickle
import xml_handler

#based on Garbage stats
class Windower:
	'''Create slices of eeg data based on sample frequency and length of slice in seconds.'''

	def __init__(self,block,nsamples = None,length_seconds = 1.0, window_overlap = True, window_overlap_percentage = 0.99,sf = 1000, file_extension= 'windows', fn_annotation = None,load_annotation =True):
		'''Create object to hold window indices of eeg data.
		
		block 						block object of an participants of a specific experimental session
		length 						the length of a window in seconds
		remove_ch 					whether to remove channels, default = eog and reference channels
		window_overlap 				whether to use overlapping windows
		window_overlap_percentage 	the percentage of overlap between consecutive windows
		sf 							sample_frequency
		'''
		self.b = block
		self.pp_id = block.pp_id
		self.exp_type = block.exp_type
		self.bid = block.bid
		self.st_sample = block.st_sample
		self.et_sample = block.et_sample
		if load_annotation: self.fn_annotation = block2fn_annotation(block)
		else: self.fn_annotation = 'unk'
		self.fn_windows = make_filename(block,file_extension)
		self.name = make_name(block)
		if nsamples == None: self.nsamples = block.duration_sample
		else: self.nsamples = nsamples
		self.annot2int = utils.annot2int
		self.length_seconds = length_seconds
		self.window_overlap = window_overlap
		self.window_overlap_percentage = window_overlap_percentage
		self.sf = sf
		self.file_extension = file_extension 
		self.windows = {}
		self.windows['sf1000'] = Snippets(self.nsamples,length_seconds,1000,window_overlap,window_overlap_percentage)
		if sf != 1000: self.windows['sf'+str(sf)] = Snippets(self.nsamples,length_seconds,sf,window_overlap,window_overlap_percentage)
		self.nsnippets = self.windows['sf1000'].nsnippets
		self.ok = True
		for k in self.windows.keys():
			if self.windows[k].nsnippets != self.nsnippets:
				self.ok = False
				print(k,self.windows[k].nsnippets, 'Does not have the same number of snippets as original',self.nsnippets)
	
	def __str__(self):
		m = 'name\t\t'+self.name + '\n'
		m += 'exp_type\t'+utils.exptype2explanation_dict[self.exp_type] +'\n'
		m += 'windows\t\t'+' '.join(list(self.windows.keys())) +'\n'
		m += 'overlap\t\t'+ str(self.window_overlap) +'\n'
		m += 'overlap perc\t'+str(self.window_overlap_percentage)+'\n'
		m += 'nsnippets\t'+str(self.nsnippets) +'\n'
		return m

	def __repr__(self):
		return 'window-object: ' + self.name + '\tnsnippets: ' + str(self.nsnippets) + '\tduration: ' +str(self.b.duration_sample)
		

	def save_windows(self):
		'''Save windows object to a pickle file.'''
		if hasattr(self,'block'): delattr(self,'block')
		fout = open(self.fn ,'wb')
		pickle.dump(self,fout,-1)


	def load_annotations(self, fn_annotation = None):
		'''Load the labels corresponding to this block in the bad_epoch filed  and order the bad_epochs based on start time.'''
		if fn_annotation != None:
			self.fn_annotation = fn_annotation
		if self.fn_annotation == None:
			self.annotations, self.bad_epoch = None, None
			print('Please provide annotation filename to load annotation.')
			return 0
		self.annotations = xml_handler.xml_handler(filename = self.fn_annotation)
		self.bad_epochs = self.annotations.xml2bad_epochs() 
		self.bad_epochs.sort()
		if len(self.bad_epochs) > 0:
			be_block_st = self.bad_epochs[0].block_st_sample
			if self.st_sample != be_block_st:
				raise ValueError('Start sample block datastats', be_block_st, 'does not equal start sample bad_epoch',self.st_sample)
		else: print('No bad epochs in this block.')

	def load_channel_annotations(self, fn_annotation = None):
		'''Load the labels corresponding to this block in the bad_channels filed  and order the bad_epochs based on start time.'''
		if fn_annotation != None:
			self.fn_annotation = fn_annotation
		if self.fn_annotation == None:
			self.annotations, self.bad_channel= None, None
			print('Please provide annotation filename to load annotation.')
			return 0
		self.annotations = xml_handler.xml_handler(filename = self.fn_annotation)
		self.bad_channels= self.annotations.xml2bad_channels() 
		self.bad_channels.sort()
		if len(self.bad_channels) > 0:
			bc_block_st = self.bad_channels[0].block_st_sample
			if self.st_sample != bc_block_st:
				raise ValueError('Start sample block datastats', bc_block_st, 'does not equal start sample bad_epoch',self.st_sample)
		else: print('No bad channels in this block.')
				
	def make_info_matrix(self, default_class = 'clean', add_pp_info = False):
		'''Create a np matrix with the type of artefact and the amount of overlap between bad_epoch and a window.
		Row indices correspond of the info matrix correspond with the row indices of the windowed data
		Column indices correspond with the annot2int dict, the overlap columns are a second set of columns besides this.
		pp_info is optionally set in the last three columns of the info matrix

		default_class 		sets the default class of the slices
		add_pp_info 		sets whether to add id info about the slices
		'''
		self.exptype2int = utils.exptype2int
		snips = self.windows['sf1000']
		if default_class not in self.annot2int.keys(): 
			default_class = 'other'
			print(default_class,'not in annot2int keys',self.annot2int,'will set the default annot to other.')
		if not hasattr(self,'annotations'): self.load_annotations()
		start_overlap_column = len(self.annot2int)
		if add_pp_info: 
			default= np.zeros(len(self.annot2int)*2 + 3) 
			default[-3:] = self.pp_id, self.exptype2int[self.exp_type], self.bid
		else: default= np.zeros(len(self.annot2int)*2)
		default[self.annot2int[default_class]] = 1.0
		default[start_overlap_column + self.annot2int[default_class]] = 1.0
		if add_pp_info: self.info_matrix = np.zeros((len(snips.start_snippets),len(self.annot2int)*2 + 3 ))
		else: self.info_matrix = np.zeros((len(snips.start_snippets),len(self.annot2int)*2 ))
		for be in self.bad_epochs:
			annotation = be.annotation if be.annotation in self.annot2int.keys() else 'other'
			indices,overlap = find_snippet_index_and_overlap_bad_epoch(snips.start_snippets,snips.end_snippets,be)			
			for i, index in enumerate(indices):
				self.info_matrix[index,self.annot2int[annotation]] = 1
				self.info_matrix[index,start_overlap_column + self.annot2int[annotation]] = overlap[i]
				self.info_matrix[index,-3:] = self.pp_id, self.exptype2int[self.exp_type], self.bid
		for i,line in enumerate(self.info_matrix):
			if max(line) == 0:
				self.info_matrix[i,:] = default 


	def make_ca_info_matrix(self, add_pp_info = False):
		'''Create a np matrix with clean-artifact annotation and the amount of overlap between bad_epoch and a window.
		Row indices correspond of the info matrix correspond with the row indices of the windowed data
		Column indices correspond with the clean - artifact - other, the overlap columns are a second set of columns besides this.
		pp_info is optionally set in the last three columns of the info matrix

		add_pp_info 		sets whether to add id info about the slices
		'''
		self.exptype2int = utils.exptype2int
		snips = self.windows['sf1000']
		self.annot2int = {'clean':0,'artifact':1,'other':2}
		self.fn_annotation = block2fn_annotation(self.b, directory = path.artifacts_clean)
		self.load_annotations(fn_annotation = self.fn_annotation)
		start_overlap_column = len(self.annot2int)
		if add_pp_info: self.info_matrix = np.zeros((len(snips.start_snippets),len(self.annot2int)*2 + 3 ))
		else: self.info_matrix = np.zeros((len(snips.start_snippets),len(self.annot2int)*2 ))
		for be in self.bad_epochs:
			annotation = be.annotation if be.annotation in self.annot2int.keys() else 'other'
			indices,overlap = find_snippet_index_and_overlap_bad_epoch(snips.start_snippets,snips.end_snippets,be)			
			for i, index in enumerate(indices):
				self.info_matrix[index,self.annot2int[annotation]] = 1.0
				self.info_matrix[index,start_overlap_column + self.annot2int[annotation]] = overlap[i]
				self.info_matrix[index,-3:] = self.pp_id, self.exptype2int[self.exp_type], self.bid


	def make_channel_ca_info_matrix(self, add_pp_info = False):
		'''Create a np matrix with clean-artifact annotation and the amount of overlap between bad_epoch and a window.
		Row indices correspond of the info matrix correspond with the row indices of the windowed data
		Column indices correspond with the channel.
		pp_info is optionally set in the last three columns of the info matrix

		add_pp_info 		sets whether to add id info about the slices
		'''
		self.exptype2int = utils.exptype2int
		self.channels = utils.load_selection_ch_names()
		self.channel2index = dict([[ch,self.channels.index(ch)] for ch in self.channels])
		snips = self.windows['sf1000']
		self.fn_annotation = block2channel_fn_annotation(self.b, directory = path.channel_artifacts_clean)
		self.load_channel_annotations(fn_annotation = self.fn_annotation)
		if add_pp_info: self.info_matrix = np.zeros((len(snips.start_snippets),len(self.channels) + 3 ))
		else: self.info_matrix = np.zeros((len(snips.start_snippets),len(self.channels)))
		for bc in self.bad_channels:
			if bc.annotation == 'clean':continue
			indices,overlap = find_snippet_index_and_overlap_bad_epoch(snips.start_snippets,snips.end_snippets,bc)			
			for i, index in enumerate(indices):
				self.info_matrix[index,self.channel2index[bc.channel]] = overlap[i]
				if add_pp_info:
					self.info_matrix[index,-3:] = self.pp_id, self.exptype2int[self.exp_type], self.bid

class Snippets:
	def __init__(self,nsamples,length_seconds,sf,window_overlap,window_overlap_percentage):
		self.nsamples = nsamples
		self.length_seconds = length_seconds
		self.length_samples = int(round(length_seconds * sf))
		self.sf = sf
		self.nsamples = int(self.nsamples / (1000/ self.sf))
		self.window_overlap = window_overlap
		self.window_overlap_percentage = window_overlap_percentage 
		self.set_start_end_snippets()

	def set_start_end_snippets(self):
		'''Create indices that mark the start and end of the sliding window.
		Overlap sets whether to overlap between adjacent windows and overlap_percentage
		set the amount overlap.
		All snippets < length will be deleted
		'''
		if self.window_overlap:
			self.start_snippets = np.arange(0,self.nsamples,int(round(self.length_samples*(1-self.window_overlap_percentage))))
		else:
			self.start_snippets = np.arange(0,self.nsamples,self.length_samples)
		self.end_snippets = self.start_snippets + self.length_samples
		# minimum snippet length = should be of length snippet 
		bad_indices = []
		for i in list(range(len(self.start_snippets)))[::-1]:
			if self.end_snippets[i] - self.start_snippets[i] < self.length_samples:
				bad_indices.append(i)
			elif self.end_snippets[i] > self.nsamples: 
				bad_indices.append(i)
			else: break
		for i in bad_indices:
			self.start_snippets = np.delete(self.start_snippets,i)
			self.end_snippets = np.delete(self.end_snippets,i)
		self.nsnippets = len(self.start_snippets)

def load_windows(block = None, fn = None, fo = None, file_extension = 'windows'):
	'''load object to file with name == eeg file and extension .data_stats in path.data_stats folder.'''
	if type(block)== str:
		fn = copy.copy(block)
		block = none
	if block: fn = make_filename(block,file_extension) 
	elif fn == none: print('need filename or block object to load windows')
	if not os.path.isfile(fn):
		print('file does not excist',fn)
		return 0
	fin = open(fn,'rb')
	output = pickle.load(fin)
	if block: output.block = block
	else: output.block = none
	output.fo = fo
	return output
	

def make_filename(b,file_extension = ''):
	'''create datastat filename based on participant and experiment info.'''
	return path.data_stats + make_name(b) + file_extension

def make_name(b):
	return 'pp'+str(b.pp_id) + '_exp-' + b.exp_type + '_bid-' + str(b.bid) 

def find_annotation_file(pp_id,exp_type,bid, coder = 'martijn',directory = None):
	'''Check wheter there is a annotation file and return filename if there is.
	'''
	if directory == None: directory = path.artifacts
	f = directory + coder +'_pp' + str(pp_id) + '_exp-' + exp_type + '_bid-' +str(bid) + '.xml'
	if os.path.isfile(f): return f
	else: 
		# print('file:',f,'not found')
		return 0

def block2fn_annotation(b, coder = 'martijn', directory = None):
	if directory == None: directory = path.artifacts
	return find_annotation_file(b.pp_id, b.exp_type,b.bid,coder = coder, directory = directory)
	
def block2channel_fn_annotation(b, directory = None):
	if directory == None: directory = path.channel_artifacts_clean
	f = directory + '*_pp' + str(b.pp_id) + '_exp-' + b.exp_type + '_bid-' +str(b.bid) + '_channels.xml'
	print(f)
	fn = glob.glob(f)
	# if len(fn) != 1: raise ValueError(len(fn),'should be 1',fn)
	if len(fn) > 1: raise ValueError(len(fn),'should be 1',fn)
	elif len(fn) == 0: return 0
	return fn[0]

def compute_overlap(start_a,end_a,start_b, end_b):
	'''compute the percentage b overlaps with a.
	if overlap = 1, b is equal in length or larger than a and start before or at the same time as a and
	b end later or ate the same time as a.
	'''
	# print(start_a,end_a,start_b,end_b)
	if end_a < start_a:
		raise ValueError('first interval is invalid, function assumes increasing intervals',start_a,end_a)
	if end_b < start_b:
		raise ValueError('second interval is invalid, function assumes increasing intervals',start_b,end_b)
	if end_b <= start_a or start_b >= end_a: return 0 # b is completely before or after a
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



def load_block(self, fo = None):
	'''Load block object based on participant and experimental session info.'''	
	if not hasattr(self,'pp_id') or not hasattr(self,'exp_type') or not hasattr(self,'bid'):
		print('Cannot load block without pp info and block info')
		return 0
	if fo == None: 
		print('Supply fidort to speed up block loading. Be carefull that using the same fidort for multiple blocks is not a good idea if you want to use word info.')
		self.fo = load_all_ort.load_fid2ort()
	s = session.Session(self.pp_id, self.exp_type, fid2ort = fo)
	self.block = getattr(s,'b' + str(self.bid))


def window_data(data,snips,flatten = False, normalize = False, cut_off=100):
	if flatten: windowed_data = np.zeros((snips.nsnippets,data.shape[0]*snips.length_samples))
	else: windowed_data = np.zeros((snips.nsnippets,data.shape[0],snips.length_samples))
	data *= 10 ** 6
	for i in range(snips.nsnippets):
		start,end = snips.start_snippets[i],snips.end_snippets[i]
		# print(i,start,end)
		if flatten:
			if normalize: d = normalize_numpy_matrix(data[:,start:end], cut_off=cut_off)
			else: d = data[:,start:end]
			windowed_data[i] = np.reshape( d, data.shape[0]*snips.length_samples)
		else:
			windowed_data[i] = data[:,start:end]
	return windowed_data


def normalize_numpy_matrix(d,cut_off = None):
	# normalize such that min = 0 and max =1
	if cut_off:
		# print('cutting of at',cut_off,'all values above will be sit to:',cut_off,'all values below:',-1*cut_off,'will be set to that value')
		d[d > cut_off] = cut_off
		d[d < -1 * cut_off] = -1 * cut_off
		mi = -1 * cut_off
		ma = cut_off
	else:
		mi = np.min(d,axis = 1)
		ma = np.max(d,axis = 1)
	dt = (d.transpose() - mi) / (ma - mi)
	return dt.transpose()
