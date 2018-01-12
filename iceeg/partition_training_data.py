import glob
import matplotlib.pyplot as plt
import numpy as np
import path
import random
import scipy.signal as signal


def hamming_data(d):
	'''Multiply all epochs with a hamming window.'''
	window_length = d.shape[1]
	hamming = signal.hamming(window_length)
	return d * hamming


def load_indices():
	'''Load np array of all indices containing clean / artifact epochs.'''
	clean_indices = np.load(path.artifact_training_data + 'clean_indices.npy')
	artifact_indices = np.load(path.artifact_training_data + 'artifact_indices.npy')
	return clean_indices, artifact_indices


def make_xfold(ai ,ci , x = 100):
	'''Create indices files for part files. Each index file contains the indices for that file to be loaded from h5 data.
	Since h5 loading was slow (4-6 hours for one clean part file to be made, I created code to convert
	the aggregated index to block file specific indices, which increased loading significantly ~ 1hour.
	'''
	random.shuffle(ci)
	random.shuffle(ai)
	start_ai,end_ai,start_ci,end_ci = 0,0,0,0
	ai_step = int(len(ai)/x)
	ci_step = int(len(ci)/x)
	ai_folds, ci_folds = [], []
	for i in range(x):
		end_ai += ai_step
		end_ci += ci_step
		print(start_ai,end_ai, start_ci,end_ci)
		if i == x-1: 
			ai_folds.append(ai[start_ai:])
			ci_folds.append(ci[start_ci:])
		else: 
			ai_folds.append(ai[start_ai:end_ai])
			ci_folds.append(ci[start_ci:end_ci])
		start_ai += ai_step
		start_ci += ci_step
	return ai_folds,ci_folds


def samplen_artifact_clean(nsamples,perc_artifacts):
	'''Create nartifact nclean int based on nsample (total n samples) and perc_artifacts.'''
	nartifact = int(round(nsamples * perc_artifacts))
	nclean = int(round(nsamples * (1-perc_aartifacts)))
	assert nartifact + nclean == nsamples
	return nartifact,nclean


def save_folds(data,folds,name):
	'''Save part files as np arrays extracted from h5 file, was very slow OBSOLETE.
	data 	np arrays
	folds 	indices of a fold
	name 	the name clean / artifact of data file
	'''
	for i,f in enumerate(folds):
		print('saving to filename:',name+'_part-' + str(i+1))
		d = data.root.data[f,:]
		# d = load_slowly(datat,f)
		np.save(path.artifact_training_data + name + '_part-' + str(i+1),d)
		

def save_all_folds(data,ai_folds,ci_folds):
	'''Save botch clean and artifact data, loads from h5, very slow OBSOLETE.'''
	save_folds(data,ai_folds,'data_artifacts')
	save_folds(data,ci_folds,'data_clean')


def save_fold_indices(ai_folds,ci_folds):
	'''Save part indices (refering to a specific epoch in the combined h5 data file.'''
	for i,f in enumerate(ai_folds):
		print('indices_artifact_part-'+str(i+1))
		print('indices_clean_part-'+str(i+1))
		np.save(path.artifact_training_data + 'PART_INDICES/indices_artifact_part-'+str(i+1),f)
		np.save(path.artifact_training_data + 'PART_INDICES/indices_clean_part-'+str(i+1),ci_folds[i])



def load_slowly(d,indices):
	'''Load one index at a time, seems to make a difference (sometimes), OBSOLETE.'''
	output = np.array(())
	for i,line in enumerate(indices):
		print(i)
		temp = d.root.data[line,:]
		if i == 0:
			output = temp
		else: output = np.concatenate((output,temp))
		if i%1000 == 0: print(i,len(indices))
	return output

# OBSOLETE FUNCTIONS are replace by code below, can be removed in later versions of this file
# duplicate in make_artifact_matrix_v2.py

def load_dict_datafn2nrows():
	'''converts a datafilename to the number of rows the corresponding np array has.'''
	df2n = dict([[line.split(',')[0],int(line.split(',')[1])] for line in open(path.data + 'datafilename2nrows.txt','r').read().split('\n') if line])
	return df2n


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

def find_correct_np_file(datafn_all_rows_before,all_index):
	'''return filename of np array that contains the all_index, and number of data rows before this file.
	all_index is the index of the epoch in the h5 file of all data file combined.
	the all index is used to find the block data file that contains this epoch.
	'''
	for i,line in enumerate(datafn_all_rows_before):
		if i < len(datafn_all_rows_before) -1:
			if datafn_all_rows_before[i+1][1] > all_index >= line[1]:
				# print('found file',all_index,line[1],datafn_all_rows_before[i+1],i)
				return line
		elif  22844424 > all_index >= line[1]:
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
	output = np.zeros((nrows,2500))
	start_index = 0
	for i,f in enumerate(filenames_and_indices):
		print(f,i,len(filenames_and_indices))
		d = np.load(f)
		temp = d[filenames_and_indices[f],:]
		output[start_index:start_index+temp.shape[0],:] = temp
		start_index += temp.shape[0]
	return output

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
	fout = open('/Users/u050158/storage/martijn_sync/clean_part_progress.txt','a')
	fout.write(f+','+ time.asctime( time.localtime(time.time()) ) +'\n')
	fout.close()

def make_all_clean_parts():
	'''Makes 100 clean part data files. Artifact parts were loaded from h5 file (clean part are 14X bigger) and were
	prohibitevily slow. Load data from original block data files speeds it up to 1 hour per file instead of 4-6.'''
	datafn2nrows = load_dict_datafn2nrows()
	datafn_all_rows_before = make_list_datafn_all_rows_before(datafn2nrows)
	clean_fn = glob.glob(path.artifact_training_data + 'PART_INDICES/indices_clean*.npy')
	for i,f in enumerate(clean_fn):
		print(f,i,len(clean_fn))
		write_update(f)
		indices = np.load(f)
		filenames_and_indices = make_filenames_and_indices(indices,datafn_all_rows_before)
		output = make_part_file(filenames_and_indices)
		output_filename = f.replace('indices','data')
		np.save(output_filename,output)
		write_update(f)

