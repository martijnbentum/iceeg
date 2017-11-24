import copy
import experiment as e
import glob
import path
import numpy as np
import os
import sklearn
import tables
import utils
import windower


def make_matrix(fo, add_pp_info = False):
	print('making artifact training np matrices per block, with default 1 sec and 90 perc overlap')
	fout = open('nrows_per_block','w')
	nrows = 0

	for i in range(1,49):
		p = e.Participant(i,fid2ort = fo)
		p.add_all_sessions()
		for s in p.sessions:
			for b in s.blocks:
				if os.path.isfile(path.artifact_training_data+ windower.make_name(b) +'.npy'):
					# check whether windowed data is already present
					continue
				if not os.path.isfile(path.eeg100hz + windower.make_name(b) +'.npy'):
					# check whether downsampled data is present to load
					continue
				print(windower.make_name(b))
				d = load_100hz_numpy_block(windower.make_name(b))
				if b.start_marker_missing or b.end_marker_missing:
					w = windower.Windower(b,nsamples= d.shape[1], sf = 100)
				else:
					w = windower.Windower(b,sf = 100)
				if w.fn_annotation == 0: 
					print('skipping:',w.name,'NO ANNOTATION')
					continue # if there is no annotation file skip
				print('processing:',w.name,w.fn_annotation)
				w.make_info_matrix(add_pp_info = add_pp_info)
				d = remove_channels(d)
				d = windower.window_data(d,w.windows['sf100'],flatten=True,normalize= True)
				# d = unit_norm(d)
				# d = normalize_numpy_matrix(d)
				rows = d.shape[0]
				nrows += rows
				fout.write(w.name + '\t' + str(rows) + '\n')
				print (d.shape, w.info_matrix.shape[0])
				assert d.shape[0] == w.info_matrix.shape[0]
				np.save(path.artifact_training_data+ w.name + '_data',d)
				np.save(path.artifact_training_data+ w.name + '_info',w.info_matrix)

	fout.write('all_blocks\t'+str(nrows)+'\n')
	fout.close()

def make_hf5_matrix(filename = 'training.h5',dimension_data = 2600, dimension_info =10,save_data =True,save_info=True):
	print('creating artifact training dataset and saving to:',filename)
	fnd = glob.glob(path.artifact_training_data+ '*data.npy')
	fni = glob.glob(path.artifact_training_data+ '*info.npy')
	for i,f in enumerate(fnd):
		print(i,f,len(fnd),fni[i], 'asserting info and data files are correctly ordered.')
		assert f.strip('data.npy') == fni[i].strip('info.npy')
	if save_data: save_hf5(fn = fnd, prefix = 'data', filename = filename, dimension = dimension_data)
	if save_info: save_hf5(fn = fni, prefix = 'info', filename = filename, dimension = dimension_info)
	
def save_hf5(fn, prefix, filename,dimension):
	print('saving to:',prefix + filename)
	fout = tables.open_file(path.artifact_training_data+ prefix + '_'+filename, mode = 'w')
	atom = tables.Float64Atom()
	array_c = fout.create_earray(fout.root,'data',atom,(0,dimension))
	for i,f in enumerate(fn):
		print(i,f,len(fn), 'saving file to hf5')
		pp_info = filename2pp_info(f)
		array_c.append(np.load(f))
	fout.close()


def normalize_numpy_matrix(d):
	# normalize such that min = 0 and max =1
	mi = np.min(d,axis = 1)
	ma = np.max(d,axis = 1)
	dt = (d.transpose() - mi) / (ma - mi)
	return dt.transpose()

def unit_norm(x):
	return sklearn.preprocessing.normalize(x,axis=1)

def load_100hz_numpy_block(name):
	return np.load(path.eeg100hz + name + '.npy')

def remove_channels(d):
	ch_names = open(path.data + 'channel_names.txt').read().split('\n')
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
	temp = f.split('/')[-1].split('_')
	pp_id = int(temp[0].strip('pp'))
	exp_type = utils.exptype2int[temp[1].strip('exp-')]
	bid = int(temp[2].strip('bid-'))
