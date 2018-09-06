import glob 
import matplotlib.pyplot as plt
import numpy as np
import path
import random
import scipy.signal as signal
import time


def make_xfold(total_nrows=None, x = 10):
	'''Create indices files for part files. Each index file contains the indices for that file to be loaded from h5 data.
	Since h5 loading was slow (4-6 hours for one clean part file to be made, I created code to convert
	the aggregated index to block file specific indices, which increased loading significantly ~ 1hour.
	'''
	if total_nrows == None: total_nrows = sum(load_dict_datafn2nrows().values())
	ai = list(range(total_nrows))
	random.shuffle(ai)
	start_ai,end_ai = 0,0
	ai_step = int(len(ai)/x)
	i_folds = []
	for i in range(x):
		end_ai += ai_step
		print(start_ai,end_ai)
		if i == x-1: 
			i_folds.append(ai[start_ai:])
		else: 
			i_folds.append(ai[start_ai:end_ai])
		start_ai += ai_step
	return i_folds


def save_fold_indices(i_folds):
	'''Save part indices refering to a specific index in a specific block in the block data files.'''
	for i,f in enumerate(i_folds):
		print('indices_part-'+str(i+1))
		np.save(path.channel_cnn_output_data+ 'PART_INDICES/indices_part-'+str(i+1),f)

# code below duplicate in make_artifact_matrix_v2.py

def load_dict_datafn2nrows():
	'''converts a datafilename to the number of rows the corresponding np array has.'''
	df2n = dict([[line.split('\t')[0],int(line.split('\t')[1])] for line in open(path.data + 'make_channel_output_data_nrows_per_block','r').read().split('\n') if line and 'data.npy' in line])
	return df2n

def make_channel_output_data_nrows_per_block():
	output = []
	fn = glob.glob(path.channel_cnn_output_data + '*.npy')
	for f in fn:
		d = np.load(f)
		output.append(f + '\t' + str(d.shape[0]))
	fout = open(path.data + 'make_channel_output_data_nrows_per_block','w')
	fout.write('\n'.join(output))
	



def make_list_datafn_all_rows_before(datafn2nrows):
	'''Computes the number of data rows come before the current file.
	= nrows of each file occuring in the glob.glob list of datafiles.''
	'''
	datafn_all_rows_before =[] 
	all_rows_before = 0
	for k in datafn2nrows:
		datafn_all_rows_before.append([k,  all_rows_before])
		all_rows_before += datafn2nrows[k]
	return datafn_all_rows_before

def find_correct_np_file(datafn_all_rows_before,all_index, total_nrows = None):
	'''return filename of np array that contains the all_index, and number of data rows before this file.
	all_index is the index of the epoch in the h5 file of all data file combined.
	the all index is used to find the block data file that contains this epoch.
	'''
	if total_nrows == None: total_nrows = sum(load_dict_datafn2nrows().values())
	for i,line in enumerate(datafn_all_rows_before):
		if i < len(datafn_all_rows_before) -1:
			if datafn_all_rows_before[i+1][1] > all_index >= line[1]:
				# print('found file',all_index,line[1],datafn_all_rows_before[i+1],i)
				return line
		elif total_nrows > all_index >= line[1]:
			# print('found file',all_index,line[1],i)
			return line
		else:
			raise ValueError('File not found',all_index,i)
			

def convert_index_all_2_specific(nrows_before,all_index):
	'''Converts the all_index to the index in the original block data file.'''
	return all_index - nrows_before

def make_part_file(filenames_and_indices):
	'''creates a part data file based on the filenames_and_indices dictionary.
	the filenames_and_indices links filenames to the indices of the epochs that are needed for the current part file.
	The function loads each filename and extract the corresponding indices and puts them in the output np.array which is returned.
	'''
	nrows = sum([len(filenames_and_indices[f]) for f in filenames_and_indices])
	data_output = np.zeros((nrows,301))
	info_output = np.zeros((nrows))
	start_index = 0
	for i,f in enumerate(filenames_and_indices):
		print(f,i,len(filenames_and_indices))
		d= np.load(f)
		info = np.load(f.replace('data','info'))
		print(d.shape,info.shape)
		temp = d[filenames_and_indices[f],:]
		itemp = info[filenames_and_indices[f]]
		data_output[start_index:start_index+temp.shape[0],:] = temp
		info_output[start_index:start_index+temp.shape[0]] = itemp
		start_index += temp.shape[0]
	return data_output,info_output

def make_filenames_and_indices(indices,datafn_all_rows_before):
	'''Create a dictionary that links a filename to the indices of the epochs in that np array that are needed for a part file.'''
	filenames_and_indices = {}
	for i in indices:
		f,nrows_before = find_correct_np_file(datafn_all_rows_before,i)
		ci = convert_index_all_2_specific(nrows_before,i)
		if f in filenames_and_indices:
			filenames_and_indices[f].append(ci)
		else: filenames_and_indices[f] = [ci]
		# print(i,len(indices_clean1))
	return filenames_and_indices

def write_update(f):
	'''Write the current part file in a text file to follow the progress.'''
	fout = open(path.data + '/part_progress.txt','a')
	fout.write(f+','+ time.asctime( time.localtime(time.time()) ) +'\n')
	fout.close()

def make_all_parts(make_parts = []):
	'''Makes 10 part data files. Artifact parts were loaded from h5 file (clean part are 14X bigger) and were
	prohibitevily slow. Load data from original block data files speeds it up to 1 hour per file instead of 4-6.'''
	if make_parts == []: make_parts = 'all'
	datafn2nrows = load_dict_datafn2nrows()
	datafn_all_rows_before = make_list_datafn_all_rows_before(datafn2nrows)
	fn = glob.glob(path.channel_cnn_output_data+ 'PART_INDICES/indices*.npy')
	for i,f in enumerate(fn):
		print(f,i,len(fn))
		if make_parts != 'all' and int(f.split('-')[-1].split('.')[0]) not in make_parts:
			print('skipping:',f)
			continue
		else:print('creating part file:', f)
		write_update(f)
		indices = np.load(f)
		filenames_and_indices = make_filenames_and_indices(indices,datafn_all_rows_before)
		data_output,info_output = make_part_file(filenames_and_indices)
		np.save(f.replace('indices','data'),data_output)
		np.save(f.replace('indices','info'),info_output)
		write_update(f.replace('indices_',''))


def make_smalltest(nrow_selection = 1000):
	number_part = list(map(str,list(range(1,11))))
	for n,i in enumerate(range(1,100,10)):
		if i == 1: 
			print('skipping',i)
			continue
		print('loading: data and info part-',str(i) , '.npy')
		d = np.load(path.channel_artifact_training_data +'PART_INDICES/data_part-'+str(i) + '.npy')
		i = np.load(path.channel_artifact_training_data +'PART_INDICES/info_part-'+str(i) + '.npy')
		nrows = d.shape[0]
		indices = random.sample(range(nrows),nrow_selection)
		sd = d[indices,:]
		si = i[indices,:]
		print('saving: data and info part-',number_part[n] , '.npy')
		np.save(path.channel_artifact_training_data +'PART_INDICES/smalltest_data_part-'+number_part[n]+'.npy',sd)
		np.save(path.channel_artifact_training_data +'PART_INDICES/smalltest_info_part-'+number_part[n]+'.npy',si)
		



