import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import path
import pickle
import progressbar as pb
import utils


'''Create a dataset based on the pmn_datasets word data for a gate. The relevant EEG is extracted
and average for a specific time window.
The code should be changed in such a way that one grand dataset can be created that
incorporates cross entropy for all gates, because the eeg data does not change.

create a plot dict based on the pmn_datasets word data for a gate, The relevant EEG data is average
over word tokens per channel.
There is code to create a plot dict for all gates not sure whether this works.

plot the plot dict
'''

fnd = glob.glob(path.pmn_datasets + '110/WORDS/*')

def f2word_number(f):
	'''extract word_number from a filename.'''
	return f.split('/')[-1].split('_')[0]

def make_pp_dict():
	'''Maps participant id to directory name of eeg word files for PMN.'''
	return dict([[f.split('PP')[-1],f] for f in glob.glob(path.pmn_words + 'PP*')])
	

def make_word2eeg_dict():
	'''create a dictionary that maps participant_id and word number to an eeg epoch of a word.
	.ch is file containing which channels are present
	.npy is numpy matrix with eeg data
	'''
	word2eeg = {}
	pp2dir = make_pp_dict()
	for k in pp2dir.keys():
		fn = glob.glob(pp2dir[k] + '/*.npy')
		for f in fn:
			wn = f2word_number(f)
			word2eeg[k + '_'+ wn] = f
	return word2eeg

def extract_trial_info(f):
	'''extract info of the word corresponding the eeg numpy matrix.'''
	# word.fid + '_' + word.sid + '_' + str(word.chunk_number) + '_' + str(word.word_number) + '_' + str(word.pos.sentence_number) + '_' + word.pos.token_number +'_' + word.word_utf8_nocode_nodia()
	fid = f.split('/')[-1].split('_')[1]
	sid = f.split('/')[-1].split('_')[2]
	w = f.split('/')[-1].split('_')[-3]
	exp = f.split('/')[-1].split('_')[-2]
	cw = f.split('_')[-1].split('.')[0]
	cid,wn,sn,tn= f.split('/')[-1].split('_')[-7:-3]
	return fid,sid,w,exp,cw,cid,wn,sn,tn
	
def load_np_ch(f):
	'''load a specific numpy eeg data matrix for a specific eeg pmn word.'''
	fch = f.replace('.npy','.ch')
	d = np.load(f)
	ch = open(fch).read().split('\n')
	return d, ch

def load_eeg_ch(f, channel):
	'''load a specific eeg channel from eeg pmn word.'''
	d, channels = load_np_ch(f)
	return extract_channel(d,channels,channel)

def extract_channel(d,channels,channel):
	'''extract a specific channel from the eeg numpy matrix.'''
	if channel in channels: i = channels.index(channel)
	else:return 'channel not found'  
	return d[i]
		
def average(channel, average_type = 'baseline'):
	'''compute the average over eeg data for baseline or pmn.'''
	if average_type == 'baseline':return np.mean(channel[100:300])
	elif average_type == 'pmn':return np.mean(channel[450:650])

def load_header():
	return open(path.pmn_datasets + 'header').read().rstrip('\n').split('\t')

def make_dataset_gate(gate = '110',w2e = None,overwrite = False,test = False):
	if w2e == None:w2e = make_word2eeg_dict()
	filename = path.pmn_datasets + 'pmn_dataset_'+gate+'_eeg.dataset'
	if not overwrite and os.path.isfile(filename):
		print(filename,'already exists, use overwrite to overwrite file')
		return None, None
	fnd = glob.glob(path.pmn_datasets + gate +'/WORDS/*')
	if test: fnd = fnd[:100]
	output, failed = [],[]
	bar = pb.ProgressBar()
	bar(range(len(fnd)))
	for i,f in enumerate(fnd):
		bar.update(i)
		o,h,f = pmn_dataword2eeg(f,w2e)
		output.extend(o)
		failed.extend(f)
	output = '\n'.join([h] + output)
	save(output,filename)
	return output,failed


def save(output,filename):
	'''save the dataset.'''
	with open(filename,'w') as fout:
		fout.write(output)
		
def check_match(fnp,pmnd):
	'''Check whether the pmn_dataset word (contains entropy)  matches 
	with eeg pmn word (conains eeg data).

	fnp 	 the numpy eeg data matrix filename
	pmnd 	 the pmn_dataset word filename
	'''
	fid, sid,w,exp,cw,cid,wn,sn,tn= extract_trial_info(fnp)
	header = load_header()
	items = fid, sid,w,exp,cw
	labels = 'fid,sid,word,exp,content_word'.split(',')
	for i,item in enumerate(items):
		if labels[i] == 'content_word': 
			item = 'False' if item == 'nw' else 'True'
		if item.lower() != pmnd[header.index(labels[i])].lower():
			print(item, pmnd[header.index(labels[i])],'should be equal',labels[i])
			with open(path.pmn_datasets + 'mismatch_failures','a') as fout:
				fout.write(fnp +'\n')
			return False
	return True
		


def pmn_dataword2eeg(f,w2e = None):	
	'''matches eeg data to the pmn dataword and extracts eeg averages for baseline and pmn.
	f 		filename of the pmn_dataword
	'''
	if w2e == None:w2e = make_word2eeg_dict()
	pmn_ch = utils.pmn_channel_set()
	pmnd = open(f).read().rstrip('\n').split('\t')
	word_number = f2word_number(f)
	failed = []
	ch_name2d= {}
	output = []
	header = load_header()
	# header = 'pp,fid,sid,word,exp,content_word,cid,wn,sn,tn'.split('\t') + header
	header = ['pp'] + header
	for ch in pmn_ch:
		header.extend([ch +'_baseline',ch + '_pmn'])
	for i in range(1,49):
		k = str(i) + '_' +word_number
		if k not in w2e.keys(): 
			failed.append(k)
			continue
		fnp = w2e[k]
		if not check_match(fnp,pmnd): continue
		temp = ['pp'+str(i)] + pmnd[:]  
		for ch in pmn_ch:
			chd = load_eeg_ch(fnp,channel= ch)
			if type(chd) == str: temp.extend(['NA','NA'])
			else:
				temp.append( average(chd,'baseline') )
				temp.append( average(chd,'pmn') )
		output.append('\t'.join(map(str,temp)))
	return output, '\t'.join(header), failed



def make_plot_gate(gate = '190',content_word=False,w2e = None,overwrite = True,test = False,save = True,only_function = False):
	'''Create a dictionary with averages for pmn plotting.'''
	if w2e == None:w2e = make_word2eeg_dict()
	fnd = glob.glob(path.pmn_datasets + gate +'/WORDS/*')
	if test: fnd = fnd[:100]
	n_dict, v_dict= {},{}
	failed = []
	bar = pb.ProgressBar()
	bar(range(len(fnd)))
	for i,f in enumerate(fnd):
		bar.update(i)
		n_dict,v_dict,failed = pmn_dataword2ploteeg(f,content_word,w2e,n_dict,v_dict,failed,only_function)
	if overwrite and save and not test:
		cw = 'cross_entropy1' if not content_word else 'cross_entropy-cw'
		if only_function: cw = 'cross_entropy-nw'
		save_plot_dict(n_dict,gate,'n',cw)
		save_plot_dict(v_dict,gate,'v',cw)
	return n_dict,v_dict,failed

def pmn_dataword2ploteeg(f,content_word,w2e = None,n_dict = {},v_dict = {},failed = [],only_function= False):
	'''extract eeg data for a specific pmn_dataword'''
	if w2e == None:w2e = make_word2eeg_dict()
	pmn_ch = utils.pmn_channel_set()
	pmnd = open(f).read().rstrip('\n').split('\t')
	header = load_header()
	if only_function and content_word and pmnd[header.index('content_word')] == 'True': return n_dict,v_dict,failed
	elif content_word and pmnd[header.index('content_word')] == 'False': return n_dict,v_dict,failed
	value = float(pmnd[header.index('cross_entropy')])
	if value < 1: value_type = 'low'
	elif 2.8 > value > 1: value_type = 'mid'
	else: value_type = 'high'

	word_number = f2word_number(f)
		
	for i in range(1,49):
		k = str(i) + '_' +word_number
		if k not in w2e.keys(): 
			failed.append(k)
			continue
		fnp = w2e[k]
		fch = fnp.replace('.npy','.ch')
		d = np.load(fnp)
		channels = open(fch).read().split('\n')
		for ch in pmn_ch:
			k = value_type + '_' + ch
			if k not in n_dict:
				n_dict[k] = 0
				v_dict[k] = np.zeros(1300)
			# chd = load_eeg_ch(fnp,channel=ch)
			chd = extract_channel(d,channels,ch)
			if type(chd) == str: continue
			else:
				n_dict[k] += 1
				v_dict[k] += chd
				
	return n_dict,v_dict, failed


def pmn_all(content_word=False,w2e = None,only_function= False):
	'''Create a plot dict for all gates.'''
	if w2e == None:w2e = make_word2eeg_dict()
	gates = [str(g) for g in range(110,660,20) if len(glob.glob(path.pmn_datasets + str(g) +'/WORDS/*')) == 49019]
	values = 'ud_entropy,entropy,cross_entropy,surprisal,ud_surprisal,cow_index_word,ud_index_word'.split(',')
	pmn_ch = utils.pmn_channel_set()
	pmnd = open(f).read().rstrip('\n').split('\t')
	header = load_header()
	if only_function and content_word and pmnd[header.index('content_word')] == 'True': return n_dict,v_dict,failed
	elif content_word and pmnd[header.index('content_word')] == 'False': return n_dict,v_dict,failed
	value = float(pmnd[header.index('cross_entropy')])
	if value < 1: value_type = 'low'
	elif 2.8 > value > 1: value_type = 'mid'
	else: value_type = 'high'

	word_number = f2word_number(f)
		
	for i in range(1,49):
		k = str(i) + '_' +word_number
		if k not in w2e.keys(): 
			failed.append(k)
			continue
		fnp = w2e[k]
		fch = fnp.replace('.npy','.ch')
		d = np.load(fnp)
		channels = open(fch).read().split('\n')
		for ch in pmn_ch:
			k = value_type + '_' + ch
			if k not in n_dict:
				n_dict[k] = 0
				v_dict[k] = np.zeros(1300)
			# chd = load_eeg_ch(fnp,channel=ch)
			chd = extract_channel(d,channels,ch)
			if type(chd) == str: continue
			else:
				n_dict[k] += 1
				v_dict[k] += chd
	return n_dict,v_dict, failed

def save_plot_dict(plot_dict,gate,dict_type,identifier = ''):
	'''Save plot_dict in pmn_dataset directory.'''
	if identifier != '' and identifier[-1] != '_': identifier += '_'
	fout = open(path.plot_dicts+ identifier + 'plot_'+gate+'_'+dict_type+'.dict','wb')
	pickle.dump(plot_dict,fout,-1)
	fout.close()

def load_plot_dict(gate,dict_type,identifier = ''):
	'''Load the plot dictionary.'''
	if identifier != '' and identifier[-1] != '_': identifier += '_'
	filename = path.plot_dicts+ identifier + 'plot_' + gate +'_'+ dict_type + '.dict'
	if not os.path.isfile(filename): 
		print(filename,'does not exists')
		return
	fin = open(filename,'rb')
	plot_dict= pickle.load(fin)
	fin.close()
	return plot_dict 

def load_vn_dicts(gate,identifier=''):
	'''load dict with eeg averages and the number of tokens for those averages.'''
	n_dict = load_plot_dict(gate,'n',identifier)
	v_dict = load_plot_dict(gate,'v',identifier)
	return n_dict,v_dict

def pmn_selection_channel_set():
	'''The relevant channels for the PMN. Work in progress.'''
	return 'F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,T8'.split(',')

def select_values(values,channels):
	'''extract values from a plot dict based on a channel set
	values 		? dict with key (hig low cross entropy) eeg average mapping ?
	channels 	? list of channels that need to be extracted ?
	not called in file.
	'''
	output,ov = {},{}
	n = {}
	for ch in channels:
		for k in values.keys():
			if ch in k:
				key = k.split('_')[0]
				if key not in output.keys(): 
					output[key] = values[k]
					n[key] = 1
				else: 
					output[key] += values[k]
					n[key] += 1
	for k in output.keys():
		ov[k] = output[k] /n[k]
	return ov
	

def cap_order(channels = []):
	'''create a dictionary with the indices of a channel in a matrix that reflects the position on
	the scalp.
	'''
	co = 'F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,CP5,CP1,CP2,CP6,P7,P3,Pz,P4,P8,O1,O2'.split(',')
	cod = {}
	cod['f'] = 'F7,F3,Fz,F4,F8'.split(',')
	cod['fc'] = 'FC5,FC1,FC2,FC6'.split(',')
	cod['c'] = 'T7,C3,Cz,C4,T8'.split(',')
	cod['cp'] = 'CP5,CP1,CP2,CP6'.split(',')
	cod['p'] = 'P7,P3,Pz,P4,P8'.split(',')
	cod['o'] = 'O1,O2'.split(',')
	adjust_index = 1
	rows,cols = 6,5
	for row_k in ['f','fc','c','cp','p','o']:
		if row_k == 'c':adjust_index+=1
		if row_k == 'p':adjust_index+=1
		if row_k == 'o':adjust_index+=1
		for channel in cod[row_k]:
			cod[channel] = (rows,cols,co.index(channel)+adjust_index)
	if channels == []: return co,cod
	else:pass # adjust cod depending on channels passed?
	
	

def plot(n_dict,v_dict,gate = '190',identifier = 'cross_entropy1'): 
	'''plots pmn epochs per channel, not yet tested (reconstructed).
	'''
	co, cod = cap_order()
	values = {}
	nd, vd = load_vn_dicts(gate,identifier)
	x = np.arange(-300,1000)
	
	for k in vd.keys():
		values[k] = vd[k] / nd[k]

	pmn_ch = utils.pmn_channel_set()
	for ch in pmn_ch:
		rows,cols,index = cod[ch]
		plt.subplot(rows,cols,index)
		plt.plot(x,values['low_'+ch]-np.mean(values['low_'+ch][:300]),color='blue')
		plt.plot(x,values['mid_'+ch]-np.mean(values['mid_'+ch][:300]),color='orange')
		plt.plot(x,values['high_'+ch]-np.mean(values['high_'+ch][:300]),color='red')
		plt.ylim(0.4,-0.4)
		plt.grid()
		plt.axis('off')
		plt.legend(('low-'+ch,'mid-'+ch,'high-'+ch),fontsize = 'x-small')
		plt.axhline(linewidth=1,color='black')
		plt.axvline(linewidth=1,linestyle='-',color='black')
		plt.axvline(150,color='tomato',linewidth=1,linestyle='--')
		plt.axvline(350,color='tomato',linewidth=1,linestyle='--')
	

def plot_article(gate = '190',identifier = 'cross_entropy1'): 
	'''plots pmn epochs per channel, not yet tested (reconstructed).
	'''
	co, cod = cap_order()
	values = {}
	nd, vd = load_vn_dicts(gate,identifier)
	x = np.arange(-300,1000)
	
	for k in vd.keys():
		values[k] = vd[k] / nd[k]

	channels = 'F7,F8,FC5,FC6,T7,T8'.split(',')
	rows,cols = 3, 2
	pmn_ch = utils.pmn_channel_set()
	for ch in channels:
		index = channels.index(ch) + 1
		plt.subplot(rows,cols,index)
		plt.plot(x,values['low_'+ch]-np.mean(values['low_'+ch][:300]),color='blue')
		plt.plot(x,values['mid_'+ch]-np.mean(values['mid_'+ch][:300]),color='orange')
		plt.plot(x,values['high_'+ch]-np.mean(values['high_'+ch][:300]),color='red')
		plt.ylim(0.35,-0.35)
		plt.axis('off')
		plt.grid()
		plt.annotate(ch,xy= (-350,-0.2))
		if ch == 'T7':
			plt.annotate('-300',xy= (-350,0.13))
			plt.annotate('1000',xy= (900,0.13))
			plt.annotate('-0.3',xy= (-180,-0.3))
			plt.annotate(' 0.3',xy= (-180,0.35))
			plt.legend(('low','mid','high'), bbox_to_anchor=(1,1.3),fontsize='small',loc = 1)
		plt.axhline(linewidth=1,color='black')
		plt.axvline(linewidth=1,linestyle='-',color='black')
		plt.axvline(150,color='tomato',linewidth=1,linestyle='--')
		plt.axvline(350,color='tomato',linewidth=1,linestyle='--')
