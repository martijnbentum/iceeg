import copy
import experiment as e
import glob
import numpy as np
import pandas as pd
import path
import utils
import xml_handler

'''Create channel xml files with only artifact clean 
all overlapping bad_epochs are combined (if they are garbage unk).
epochs are stiched together if they overlap or less then 500 ms apart
all remaining data is clean 
'''

def get_xml_files():
	'''Load all xml filenames from the artifacts folder.'''
	fn = glob.glob(path.bad_channels+ '*.xml')
	return fn

def get_bad_channels(filename):
	'''Load the xml file and create the bad_epochs from the contents.'''
	x = xml_handler.xml_handler(filename = filename)
	x.load_xml()
	bad_channels= x.xml2bad_channels()
	return bad_channels

def get_all_bad_channels():
	'''Get all bad_epochs from all xml files.'''
	fn = get_xml_files()
	bad_channels= []
	for f in fn:
		bad_channels.extend( get_bad_channels(f) )
	print('bad_channels',len(bad_channels))
	print('xml files',len(fn))
	return bad_channels


def make_duration_list(bad_channels= [], annotation = ''):
	'''List all durations of the bad_epochs.'''
	duration_list = []
	for bc in bad_channels:
		if bc.annotation == annotation: 
			duration_list.append(bc.duration /1000)
	return duration_list


def select_artifacts(f, selection = ['unk','garbage','all']):
	'''Select only those bad_channels that are unk garbage.'''
	bad_channels= get_bad_channels(f)
	bad_channels.sort()
	artifacts = [bc for bc in bad_channels if bc.annotation in selection]
	return artifacts

def stitch_artifacts(artifacts, distance = 500):
	'''Create 2d vectors of bad_epoch indices that are less then or equal to the distance apart.'''
	stiches = []
	for i,bc in enumerate(artifacts):
		if i > len(artifacts)-2: pass
		elif 0 <= artifacts[i+1].st_sample - bc.et_sample <= distance:
			# print(bc.start.x,bc.end.x)
			# print(artifacts[i+1].start.x, artifacts[i+1].end.x,)
			# print(i)
			# print('-'*8)
			stiches.append([i,i+1])
	return stiches

def stitch_stiches(stiches_indices):
	'''Combine consecutive indices.
	stiches indices contains 2d vectors with indices corresponding to bad_epochs that should be combined.
	if the indices form a chain [[19,20],[20,21]], they are stiched together -> [[19,21]].'''
	stiches = copy.deepcopy(stiches_indices)
	# print('start',stiches)
	i = 0
	stiched = False
	while True:
		if len(stiches) == 1: break
		if len(stiches) == 2:
			if stiches[0][1] == stiches[1][0]: 
				stiches = [[stiches[0][0],stiches[1][1]]]
			break
		if i >= len(stiches) -1:
			if not stiched: break
			stiched = False
			i = 0
		if stiches[i][1] == stiches[i+1][0]:
			new_stich = [stiches[i][0],stiches[i+1][-1]]
			stiches[i] = new_stich
			stiches.pop(i+1)
			if len(stiches) == 1: break
			i = 0
			stiched = True
		i += 1
	return stiches

def combine_artifacts(artifacts,stiches):
	'''Combine bad_channels that overlap or are less than 500 samples apart.'''
	indices = [range(s[0],s[1]+1) for s in stiches]
	indices = [x for y in indices for x in y]
	new_artifacts = []
	for s in stiches:
		bc = copy.deepcopy(artifacts[s[0]])
		if bc.start.x > artifacts[s[1]].start.x: bc.start = artifacts[s[1]].start
		if bc.end.x < artifacts[s[1]].end.x: bc.end = artifacts[s[1]].end
		bc.set_info()
		if hasattr(bc,'epoch_ids') and getattr(bc,'epoch_ids') != 'NA': bc.epoch_ids += ',' + ','.join([str(bc.epoch_id) for bc in artifacts[s[0]:s[1]+1]])
		else: bc.epoch_ids = ','.join([bc.epoch_id for bc in artifacts[s[0]:s[1]+1]])
		bc.annotation = 'artifact'
		bc.color = 'blue'
		new_artifacts.append(bc)
	# print(len(new_artifacts))
	for i in range(len(artifacts)):
		if i not in indices:
			# print(i,end=',')
			a = artifacts[i]
			a.annotation = 'artifact'
			a.color = 'blue'
			new_artifacts.append(a)
	# print(len(new_artifacts))
	new_artifacts.sort()
	return new_artifacts

def bad_channel2block(bc,fo = None):
	'''Return block object that correspond to the bad_epoch.'''
	p = e.Participant(bc.pp_id,fid2ort = fo)
	p.add_session(bc.exp_type)
	s = getattr(p,'s' + bc.exp_type)
	return getattr(s, 'b' + str(bc.bid))

def find_overlap_artifacts(artifacts):
	'''Find indices of bad_epochs that overlap with each other.'''
	overlap_indices = []
	for i,a in enumerate(artifacts):
		if i > len(artifacts) -2: pass
		elif utils.compute_overlap(a.st_sample,a.et_sample,artifacts[i+1].st_sample,artifacts[i+1].et_sample) > 0:
			overlap_indices.append([i,i+1])
	return overlap_indices

			
def combine_overlaps(artifacts):
	'''Combine all artifacts that overlap with each other.
	multiple passes could be needed to catch all overlapp epochs.'''
	a = copy.deepcopy(artifacts)
	i = 0
	while True:
		print(i)
		i += 1
		o = find_overlap_artifacts(a)
		if o == []: break
		ost = stitch_stiches(o)
		a = combine_artifacts(a,ost)
	return a


def make_new_clean_epoch(ep_id,start,end,bc,default):
	'''make a bad_epoch object for cleaan sections between artifacts.
	should exclude the bad pp sections, call it other.'''
	clean_epoch = copy.deepcopy(bc)
	clean_epoch.epoch_id = ep_id 
	clean_epoch.start.x = start
	clean_epoch.end.x = end
	clean_epoch.set_info()
	clean_epoch.annotation = default
	clean_epoch.color = 'white'
	return clean_epoch


def add_clean_epochs(artifacts, default,b, minimal_duration = 500):
	'''Add clean epochs for stretches between bad_epochs.'''
	artifacts.sort()
	epochs = []
	for i, a in enumerate(artifacts):
		# print(i,'--',a,b.duration_sample,a.et_sample,b.duration_sample-a.et_sample)
		if i == 0 and a.st_sample >=  minimal_duration: 
			# print('add start artifact')
			ep_id = '0.' + str(a.epoch_id)
			epochs.append(make_new_clean_epoch(ep_id,0,a.st_sample-1,a,default))
		if i != len(artifacts) -1: 
			ep_id = str(a.epoch_id) + '.' + str(artifacts[i+1].epoch_id)
			start = a.et_sample + 1
			end = artifacts[i+1].st_sample 
			epochs.append(make_new_clean_epoch(ep_id,start,end,a,default))
		if i == len(artifacts) -1 and minimal_duration <= b.duration_sample - a.et_sample: 
			# print('add end artifact')
			ep_id = str(a.epoch_id) + '.0'
			epochs.append(make_new_clean_epoch(ep_id,a.et_sample,b.duration_sample,a,default))
	# print(artifacts,999999999)
	artifacts.extend(epochs)
	artifacts.sort()
	# for i,a in enumerate(artifacts):
		# print(i,a)
	return artifacts
			

def check_artifacts(artifacts,b,default, minimal_duration = 500):
	for i,bc in enumerate(artifacts):
		if i < len(artifacts) -1:
			if not minimal_duration <= artifacts[i+1].st_sample - bc.et_sample: 
				print('PROBLEM')
				print(artifacts[i+1].st_sample,bc.et_sample)
				print(bc,i)
				print
				print(artifacts[i+1])
			# assert minimal_duration <= artifacts[i+1].st_sample - bc.et_sample 
	artifacts = add_clean_epochs(artifacts,default,b, minimal_duration = minimal_duration)
	for i,bc in enumerate(artifacts):
		if bc.annotation == default:
			if not bc.duration >= minimal_duration: print(i,bc)
			# assert bc.duration >= 500



def compute_artifact_clean_duration():
	exptype2int = {'o':1,'k':2,'ifadv':3}
	fn_xml = glob.glob(path.artifacts_clean + '*')
	fout = open('artifacts-channel_duration.txt','w')
	for f in fn_xml:
		print(f)
		xml_artifacts = xml_handler.xml_handler(filename = f)
		xml_artifacts.load_xml()
		xml_artifacts.xml2bad_channels()
		artifacts = xml_artifacts.bad_channels
		bc = artifacts[0]
		pp_id = bc.pp_id
		exp_type = exptype2int[bc.exp_type]
		bid = bc.bid
		clean = sum(make_duration_list(artifacts,'clean'))
		artifact = sum(make_duration_list(artifacts,'artifact'))
		other = sum(make_duration_list(artifacts,'other'))
		print(f,clean,artifact,other)
		fout.write(','.join(map(str,[pp_id,exp_type,bid,clean,artifact,other])) + '\n')
	fout.close()


def load_artifact_clean_duration():
	names = ['pp_id','exp_type','bid','clean','artifact','other']
	duration = pd.read_csv(filepath_or_buffer='artifacts_duration.txt',names =names)
	d1 = duration[duration.other == 0]
	d1.assign(ratio = d1.clean/ (d1.artifact+1))


def handle_channel(all_artifacts,b,ch_name,bc = None,fo = None,default = 'clean'):
	'''artifacts should contain artifacts from a specfic block.
	artifacts 		list of bad channel object 
	b 				block correponding to the artifact annotations
	ch_name 		the channel name to create artifacts from
	fidort 			data to speed up loading artifacts
	default 		the default annotation of the channel (i.e. not artifact)
	'''
	artifacts = []
	for a in all_artifacts:
		if a.channel == ch_name:
			 artifacts.append(a)
	print(all_artifacts)

	if len(artifacts) == 0: 
		if bc == None: bc = copy.deepcopy(all_artifacts[0])
		print(bc)
		bc.channel = ch_name
		bc.start.x = 0 
		bc.end.x = b.duration_sample
		bc.set_info()
		bc.epoch_id = '0.' + bc.epoch_id
		bc.color = 'white'
		bc.annotation = default
		artifacts = [bc]
	elif len(artifacts) == 1 and artifacts[0].annotation == 'all':
		artifacts[0].annotation = 'artifact'
		artifacts[0].end.x = b.duration_sample
		artifacts[0].set_info()
	else:
		artifacts.sort()
		artifacts = combine_overlaps(artifacts)
		stiches = stitch_artifacts(artifacts)
		stiches = stitch_stiches(stiches)
		artifacts = combine_artifacts(artifacts,stiches)
	
	check_artifacts(artifacts,b,default)
	for a in artifacts:
		a.block_et_sample = b.et_sample
	return artifacts


def run(fo = None,start_index = 0):
	'''Create new xml files with two classes of bad_epochs: clean and artifact. 
	If block is part of bad_pp_artifact training clean is changed with other
	all epochs are non overlapping and clean epochs are minimally 500ms in length.'''
	exptype2int = {'o':1,'k':2,'ifadv':3}
	fn = get_xml_files()
	all_clean, all_artifact = 0,0
	pp = {}
	default = 'clean'
	channels = utils.load_selection_ch_names()

	for i,f in enumerate(fn[start_index:]):
		print(i,f)
		bad_channels= get_bad_channels(f)
		bc = copy.deepcopy(bad_channels[0])
		b = bad_channel2block(bc,fo)

		all_artifacts = select_artifacts(f)
		artifacts = []
		for ch in channels:
			if len(all_artifacts) == 0:
				bc = copy.deepcopy(bad_channels[0])
				artifacts.extend(handle_channel(all_artifacts,b,ch,bc,fo))
			else:
				artifacts.extend(handle_channel(all_artifacts,b,ch,None,fo))
			
		print(f,'\n',artifacts,'\n')
		'''
		clean = sum(make_duration_list(artifacts,default))
		artifact = sum(make_duration_list(artifacts,'artifact'))
		a = artifacts[0]
		if str(a.pp_id) + 'clean' not in pp.keys(): pp[str(a.pp_id)+'clean'] = [clean]
		else: pp[str(a.pp_id)+'clean'].append(clean)
		if str(a.pp_id) + 'artifact' not in pp.keys(): pp[str(a.pp_id)+'artifact'] = [artifact]
		else: pp[str(a.pp_id)+'artifact'].append(artifact)
		if artifact == 0: artifact = 1
		if str(a.pp_id) + 'ratio' not in pp.keys(): pp[str(a.pp_id)+'ratio'] = [clean/artifact]
		else: pp[str(a.pp_id)+'ratio'].append(clean/artifact)
		print(i,f,clean,artifact,clean/artifact)
		all_clean += clean
		all_artifact += artifact
		print('_'*9)
		'''
		x = xml_handler.xml_handler(bad_channels=artifacts,filename = path.channel_artifacts_clean + f.split('/')[-1])
		x.bad_channels2xml()
		x.write()
	return artifacts	
	# print(all_clean,all_artifact,all_clean/all_artifact)
	# return pp


def compute_nchannel_artifacts_clean(threshold = 0.5):
	fn = glob.glob(path.channel_artifact_training_data + '*info*')
	fout = open(path.data + 'artifact_clean_channel.txt','w')
	fout.close()
	for f in fn:
		name = f.split('/')[-1]
		print(name)
		info = np.load(f)
		info = (info >= 0.5).astype(int)

		all_columns= np.where(sum(info) == info.shape[0])[0]
		temp = np.delete(info,all_columns, axis = 1)
		nrows = str(info.shape[0])
		bad_rows = str(len(list(set(np.where(temp >= 0.5)[0]))))
		perc_bad_rows = str(round((int(bad_rows) / int(nrows)),3))

		artifact = str(np.sum(info))
		artifact_ch = ' '.join(map(str,sum(info)))
		clean = str(info.size - np.sum(info))
		perc_artifact = str(round((np.sum(info) / info.size),3))
		fout = open(path.data + 'artifact_clean_channel.txt','a')
		fout.write('\t'.join([name,nrows,bad_rows,perc_bad_rows,clean,artifact,perc_artifact,artifact_ch]) + '\n')
		fout.close()
		

def load_artifact_clean_channel():
	t = [line.split('\t') for line in open(path.data + 'artifact_clean_channel.txt').read().split('\n') if line]
	for line in t:
		for i in [1,2,4,5]:
			line[i] = int(line[i])
		for i in [3,6]:
			line[i] = float(line[i])
		line[-1] = list(map(int,line[-1].split(' ')))
	return t

def compute_annotation_duration():
	fn = glob.glob(path.channel_artifacts_clean + '*xml')
	all_duration = 0
	for f in fn:
		bc = get_bad_channels(f)
		temp = bc[0]
		all_duration += int(temp.block_et_sample) - int(temp.block_st_sample)
	return all_duration


def index2nbad_channels():
	fn = glob.glob(path.channel_artifact_training_data + '*info*')
	fout = open(path.data + 'index2nbad_channels.txt','w')
	fout.close()
	output = []
	for f in fn:
		name = f.split('/')[-1]
		print(name)
		info = np.load(f)
		ind = np.where(info>=0.5)[0]
		temp = []
		for i in range(info.shape[0]):
			temp.append(str(len(np.where(ind == i)[0])))
			output.append(int(temp[-1]))
		fout = open(path.data + 'index2nbad_channels.txt','a')
		fout.write('\n'.join(temp) +'\n')
		fout.close()
	return output

	
		
