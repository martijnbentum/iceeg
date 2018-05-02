import copy 
import experiment as e 
import glob
import numpy as np
import os
import path
import random
import sklearn
import time
import tables
import utils
import windower


def make_matrix(fo, add_pp_info = False,normalize_data = True,save_directory =None,make_data= True,start_pp =1,overwrite = False):
	print('making artifact training np matrices per block, with default 1 sec and 99 perc overlap')
	if save_directory == None: save_directory = path.channel_artifact_training_data
	print('save directory:',save_directory)
	if make_data: fout = open(path.bak+ 'channel_nrows_per_block-p','w')
	nrows = 0

	for i in range(start_pp,49):
		p = e.Participant(i,fid2ort = fo)
		p.add_all_sessions()
		for s in p.sessions:
			for b in s.blocks:
				if os.path.isfile(path.channel_artifact_training_data+ windower.make_name(b) +'_data.npy') and not overwrite:
					# check whether windowed data is already present
					continue
				if not os.path.isfile(path.eeg100hz + windower.make_name(b) +'.npy'):
					# check whether downsampled data is present to load
					continue
				print(windower.make_name(b))
				d = load_100hz_numpy_block(windower.make_name(b))
				if b.start_marker_missing or b.end_marker_missing:
					w = windower.Windower(b,nsamples= d.shape[1], sf = 100,window_overlap_percentage = .99)
				else:
					w = windower.Windower(b,sf = 100,window_overlap_percentage = .99)
				f = windower.block2channel_fn_annotation(w.b,path.channel_artifacts_clean)
				if f == 0: 
					print('skipping:',w.name,'NO ANNOTATION')
					continue # if there is no annotation file skip
				print('processing:',w.name,w.fn_annotation)
				w.make_channel_ca_info_matrix(add_pp_info = add_pp_info)
				if make_data:
					d = remove_channels(d)
					d = windower.window_data(d,w.windows['sf100'],flatten=True,normalize= normalize_data,cut_off = 300)
					rows = d.shape[0]
					nrows += rows
					fout.write(w.name + '\t' + str(rows) + '\n')
					print (d.shape, w.info_matrix.shape[0])
					assert d.shape[0] == w.info_matrix.shape[0]
					d = insert_target_channel_rows(d,nchannels=26,kernel_size=6)
					np.save(save_directory+ w.name + '_data',d)
				np.save(save_directory+ w.name + '_info',w.info_matrix)

	if make_data:
		fout.write(path.bak+ 'all_blocks\t'+str(nrows)+'\n')
		fout.close()

def make_channel_nrows_per_block():
	'''creates a list of the nrows in windowed block file save in channel_artifact_training_data.'''
	fout = open(path.data+ 'channel_nrows_per_block','w')
	fn = glob.glob(path.channel_artifact_training_data + '*info*')
	filename2row = {}
	for f in fn:
		name = f.split('/')[-1]
		d = np.load(f)
		filename2row[name] = d.shape[0]
		fout.write(name + '\t' + str(d.shape[0]) + '\n')
	fout.close()
	return filename2row
		
def load_dict_datafn2nrows():
	'''loads a dictionary that transelates windowed block filenames to nrows.'''
	df2n = dict([[line.split('\t')[0],int(line.split('\t')[1])] for line in open(path.data + 'channel_nrows_per_block','r').read().split('\n') if line])
	return df2n
		

def align_info_filenames_with_data_filename(fnd,fns):
	aligned_fni = []
	for f in fnd:
		done = False
		for fs in fns:
			if f.split('/')[-1].replace('_data.npy','') == fs.split('/')[-1].replace('.index_info.npy',''): 
				print(f.split('/')[-1],fs.split('/')[-1])
				if done:print('OOOOOOPS',f,fs,aligned_fni[-1])
				aligned_fni.append(fs)
				done = True
	return aligned_fni


def make_list_datafn_all_rows_before(datafn2nrows):
	datafn_all_rows_before =[] 
	all_rows_before = 0
	for k in datafn2nrows:
		datafn_all_rows_before.append([k,  all_rows_before])
		all_rows_before += datafn2nrows[k]
	return datafn_all_rows_before

def find_correct_np_file(datafn_all_rows_before,all_index):
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
	return all_index - nrows_before

def make_part_file(filenames_and_indices):
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
	fout = open('/Users/u050158/storage/martijn_sync/clean_part_progress.txt','a')
	fout.write(f+','+ time.asctime( time.localtime(time.time()) ) +'\n')
	fout.close()

def make_all_clean_parts():
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


		

def make_hf5_matrix(filename = 'training.h5',dimension_data = 2500, dimension_info =10,save_data =True,save_info=True,use_snippet_info = False):
	print('creating artifact training dataset and saving to:',filename)
	fnd = glob.glob(path.artifact_training_data+ 'pp*data.npy')
	fni = glob.glob(path.artifact_training_data+ 'pp*info.npy')

	fns = glob.glob(path.snippet_annotation+ 'pp*info.npy')
	print(len(fns))
	fns = align_info_filenames_with_data_filename(fnd,fns)

	print(len(fnd),len(fni),len(fns))

	if use_snippet_info: fni = fns

	for i,f in enumerate(fnd):
		print(i,f,len(fnd),fni[i], 'asserting info and data files are correctly ordered.')
		if use_snippet_info:
			assert f.split('/')[-1].strip('_data.npy') == fni[i].split('/')[-1].strip('.index_info.npy')
		else: assert f.strip('data.npy') == fni[i].strip('info.npy')
	if save_data: save_hf5(fn = fnd, prefix = 'data', filename = filename, dimension = dimension_data)
	if save_info: 
		if use_snippet_info: save_hf5(fn = fni, prefix = 'info-snippet', filename = filename, dimension = dimension_info)
		else: save_hf5(fn = fni, prefix = 'info', filename = filename, dimension = dimension_info)
	
def save_hf5(fn, prefix, filename,dimension):
	print('saving to:',prefix + filename)
	fout = tables.open_file(path.artifact_training_data+ prefix + '_'+filename, mode = 'w')
	atom = tables.Float64Atom()
	array_c = fout.create_earray(fout.root,'data',atom,(0,dimension))
	for i,f in enumerate(fn):
		print(i,f,len(fn), 'saving file to hf5')
		# pp_info = filename2pp_info(f)
		array_c.append(np.load(f))
	fout.close()

def save_partial_hf5(orig,indices,filename,dimension = 2500):
	print('saving to:', path.artifact_training_data + filename)
	indices.sort()
	indices_list = []
	nrows = (int(len(indices) /1000))
	start,end = 0,0
	for r in range(1,nrows+1):
		start = end
		end += 1000
		print(start,end,'bla')
		indices_list.append(indices[start:end])
	indices_list.append(  indices[r*1000:] )
	print(nrows,len(indices_list))
	with tables.open_file(path.artifact_training_data+ filename, mode = 'w') as fout:
		atom = tables.Float64Atom()
		array_c = fout.create_earray(fout.root,'data',atom,(0,dimension))
		for i,indices in enumerate(indices_list):
			print(i,'nepochs',len(indices),'total',len(indices_list), 'saving file to hf5')
			d = orig.root.data[i,:]
			print(d.shape)
			array_c.append(d)
	

def load_bad_pp():
	exptype2int = {'o':1,'k':2,'ifadv':3}
	annot2int = {'clean':0,'garbage':1,'unk':2,'drift':3,'other':4}

	bad_pp = [line.strip().split('\t') for line in open(path.data + 'bad_pp_artifact_training.txt').read().split('\n') if line]
	bad_pp = [line[:-1] + list(map(int,line[-1].split(','))) for line in bad_pp]
	bad_pp = [[int(line[0]), exptype2int[line[1]], line[2]] for line in bad_pp]
	return bad_pp

def find_bad_pp_indices(info,bad_pp = []):
	if bad_pp == []:
		bad_pp = load_bad_pp()
	bad_pp_indices = []
	for pp in bad_pp:
		indices = np.where(info[:,-3] == pp[0])[0]
		bad_indices = []
		for i in indices:
			if list(info[i,-3:]) == pp:
				bad_indices.append(i)
		print(pp,info[bad_indices,-3:],len(bad_indices))
		bad_pp_indices.extend(bad_indices)
	return bad_pp_indices
	


def make_ds(data = None,info = '',bad_pp_indices = [],perc_cutoff = 0.9):
	if info != np.ndarray:
		dinfo = np.load(path.artifact_training_data + 'info_training.npy')
	else: dinfo = info
	if data == None:
		data_f = path.artifact_training_data+ 'data_training.h5'
		print('Loading file pointer:',data_f)
		data = tables.open_file(data_f, mode = 'r')
	if bad_pp_indices == str: 
		bad_pp_indices = list(map(int,open(bad_pp_indices).read().split('\n')))
	elif bad_pp_indices == []:
		bad_pp_indices = find_bad_pp_indices(info)
	all_indices = np.arange(dinfo.shape[0])
	artifacts_indices,temp,semi_artifact_indices = [],[],[]
	for i in [1,2,3]:
		# 1,2,3 correspond to garbage unk and drift
		temp.extend(np.where(dinfo[:,i] == 1)[0])
	for i in temp:
		if np.nonzero(dinfo[i,:] > perc_cutoff)[0].shape[0] == 2:
			artifacts_indices.append(i)
		else: semi_artifact_indices.append(i)
	artifacts_indices = list(set(artifacts_indices))
	remove_indices = bad_pp_indices + artifacts_indices + semi_artifact_indices
	clean_indices = list(np.setdiff1d(all_indices,remove_indices))
	clean_indices_selection = random.sample(clean_indices,len(artifacts_indices))
	return data, dinfo, clean_indices, clean_indices_selection,artifacts_indices, semi_artifact_indices,bad_pp_indices

def load_info():
	info = np.load(path.artifact_training_data + 'info_training.npy')
	return info

def load_bad_pp_indices():
	bad_pp_indices = list(map(int,open(path.data + 'bad_pp_indices.txt').read().split('\n')))
	return bad_pp_indices

def load_ds():
	data_f = path.artifact_training_data+ 'data_training.h5'
	print('Loading file pointer:',data_f)
	data = tables.open_file(data_f, mode = 'r')
	artifact_indices = list(map(int,open(path.data + 'artifact_indices.txt').read().split('\n')))
	clean_indices = list(map(int,open(path.data + 'clean_indices.txt').read().split('\n')))
	clean_indices_selection = list(map(int,open(path.data + 'clean_indices_selection.txt').read().split('\n')))
	return data, artifact_indices, clean_indices, clean_indices_selection

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


def load_clean_info(f = None):
	if f == None: f = path.artifacts + 'clean_training_info.txt'
	clean_info = []
	t = [line.split('\t') for line in open(f).read().split('\n') if line]
	for line in t:
		line[0] = int(line[0])
		line[1] = utils.exptype2int[line[1]]
		if len(line[-1].split(',')) > 1:
			temp = []
			for n in line[-1].split(','):
				temp.append(line[:-1] + [int(n)])
			clean_info.extend(temp)
		else:
			line[-1] = int(line[-1]) 
			clean_info.append(line)
	return clean_info

def filename2pp_info(f):
	f.replace('.','_')
	temp = f.split('/')[-1].split('_')
	pp_id = int(temp[0].strip('pp'))
	exp_type = utils.exptype2int[temp[1].strip('exp-')]
	bid = int(temp[2].strip('bid-'))



def insert_target_channel_rows(data,tc_value = 0.0,kernel_size = 6, nchannels = 26):
	'''Insert the target channel at regular intervals in the training/test sample.
	The target channel is inserted in such a way that the kernel always sees the target channel once.
	Only at the position where the target channel is originally present does the kernel see
	the target channel twice.
	
	data 		data read in from channel_artifact_training_data, each line contains one sample
				with (default) 26 channels and 100 timepoints.
	tc_value 	the index of the target channel
	kernel_size height of the kernel
	nchannels 	number of channels in a sample
	'''
	
	indices = list(range(0,nchannels,kernel_size-1))
	for i,m in enumerate(data):
		m = np.reshape(m,[nchannels,-1])
		tch = [tc_value] * m.shape[1]
		d =np.reshape(np.insert(m,indices,tch,axis=0),[1,-1])
		if i == 0: output = d[:]
		else: output = np.concatenate((output,d[:]))
	return output
