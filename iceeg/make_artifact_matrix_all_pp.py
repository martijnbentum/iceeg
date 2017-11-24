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

#Create np matrices for each pp and each block with a moving window of 1 sec and a step of 0.1 sec
#The matrices will be used to make predictions based on the current model 
# 	to extend number of artifacts
# 	to exclude bad data from ICA
# 	importantly the first batch of annotated data is not present


def make_matrix(fo, add_pp_info = False):
	print('making artifact data np matrices per block, with default 1 sec and 90 perc overlap')
	fout = open('nrows_per_block','w')
	nrows = 0

	for i in range(1,49):
		p = e.Participant(i,fid2ort = fo)
		p.add_all_sessions()
		for s in p.sessions:
			for b in s.blocks:
				if os.path.isfile(path.artifact_data_all_pp+ windower.make_name(b) +'.npy'):
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
				print('processing:',w.name,w.fn_annotation)
				d = remove_channels(d)
				d = windower.window_data(d,w.windows['sf100'],flatten=True,normalize= True)
				rows = d.shape[0]
				nrows += rows
				fout.write(w.name + '\t' + str(rows) + '\n')
				print (d.shape )
				np.save(path.artifact_data_all_pp+ w.name + '_data',d)

	fout.write('all_blocks\t'+str(nrows)+'\n')
	fout.close()


def load_100hz_numpy_block(name):
	return np.load(path.eeg100hz + name + '.npy')

def remove_channels(d):
	'''remove eeg channels, by default the reference and eog channels. 
	Also deleted Fp2 and F8 to have an even number of channels for cnn training.
	'''
	ch_names = open(path.data + 'channel_names.txt').read().split('\n')
	remove_ch = ['VEOG','HEOG','TP10_RM','STI 014','LM','F8','Fp2']
	print('removing channels:',remove_ch)
	ch_mask = [n not in remove_ch for n in ch_names]
	ch_names= [n for n in ch_names if not n in remove_ch]
	d= d[ch_mask,:]
	nchannels = len(ch_names)
	print('remaining channels:',ch_names,'nchannels:',nchannels,'data shape:',d.shape)
	return d
