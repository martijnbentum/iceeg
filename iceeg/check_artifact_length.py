import copy
import experiment as e
import glob
import pandas as pd
import path
import utils
import xml_handler

'''Create xml files with only artifact clean and other annotation.
all overlapping bad_epochs are combined (if they are garbage unk or drift).
epochs are stiched together if they overlap or less then 500 ms apart
all remaining data is clean or other (if the block is in the bad_pp list.
'''

def get_xml_files():
	'''Load all xml filenames from the artifacts folder.'''
	fn = glob.glob(path.artifacts + '*.xml')
	return fn

def get_bad_epochs(filename):
	'''Load the xml file and create the bad_epochs from the contents.'''
	x = xml_handler.xml_handler(filename = filename)
	x.load_xml()
	bad_epochs = x.xml2bad_epochs()
	return bad_epochs

def get_all_bad_epochs():
	'''Get all bad_epochs from all xml files.'''
	fn = get_xml_files()
	bad_epochs = []
	for f in fn:
		bad_epochs = get_bad_epochs(f)
	print('bad_epochs',len(bad_epochs))
	print('xml files',len(fn))
	return bad_epochs


def make_duration_list(bad_epochs = [], annotation = ''):
	'''List all durations of the bad_epochs.'''
	duration_list = []
	for be in bad_epochs:
		if be.annotation == annotation: 
			duration_list.append(be.duration /1000)
	return duration_list


def select_artifacts(f):
	'''Select only those bad_epochs taht are unk drift garbage.'''
	bad_epochs = get_bad_epochs(f)
	bad_epochs.sort()
	artifacts = [be for be in bad_epochs if be.annotation in ['unk','drift','garbage']]
	return artifacts

def stitch_artifacts(artifacts, distance = 500):
	'''Create 2d vectors of bad_epoch indices that are less then or equal to the distance apart.'''
	stiches = []
	for i,be in enumerate(artifacts):
		if i > len(artifacts)-2: pass
		elif 0 <= artifacts[i+1].st_sample - be.et_sample <= distance:
			# print(be.start.x,be.end.x)
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
		# print(i,len(stiches)-2,'bla',stiches)
		if len(stiches) == 1: break
		if len(stiches) == 2:
			if stiches[0][1] == stiches[1][0]: 
				stiches = [[stiches[0][0],stiches[1][1]]]
			break
		if i >= len(stiches) -1:
			# print('blie')
			if not stiched: break
			stiched = False
			i = 0
		if stiches[i][1] == stiches[i+1][0]:
			# print('blio')
			new_stich = [stiches[i][0],stiches[i+1][-1]]
			stiches[i] = new_stich
			stiches.pop(i+1)
			if len(stiches) == 1: break
			i = 0
			stiched = True
		i += 1
	return stiches

def combine_artifacts(artifacts,stiches):
	'''Combine bad_epochs that overlap or are less than 500 samples apart.'''
	indices = [range(s[0],s[1]+1) for s in stiches]
	indices = [x for y in indices for x in y]
	# print(indices,len(indices))
	new_artifacts = []
	for s in stiches:
		be = copy.deepcopy(artifacts[s[0]])
		if be.start.x > artifacts[s[1]].start.x: be.start = artifacts[s[1]].start
		if be.end.x < artifacts[s[1]].end.x: be.end = artifacts[s[1]].end
		be.set_info()
		if hasattr(be,'epoch_ids') and getattr(be,'epoch_ids') != 'NA': be.epoch_ids += ',' + ','.join([str(be.epoch_id) for be in artifacts[s[0]:s[1]+1]])
		else: be.epoch_ids = ','.join([be.epoch_id for be in artifacts[s[0]:s[1]+1]])
		be.annotation = 'artifact'
		be.color = 'blue'
		new_artifacts.append(be)
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

def bad_epoch2block(be,fo = None):
	'''Return block object that correspond to the bad_epoch.'''
	p = e.Participant(be.pp_id,fid2ort = fo)
	p.add_session(be.exp_type)
	s = getattr(p,'s' + be.exp_type)
	return getattr(s, 'b' + str(be.bid))

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


def make_new_clean_epoch(ep_id,start,end,be,default):
	'''make a bad_epoch object for cleaan sections between artifacts.
	should exclude the bad pp sections, call it other.'''
	clean_epoch = copy.deepcopy(be)
	clean_epoch.epoch_id = ep_id 
	clean_epoch.start.x = start
	clean_epoch.end.x = end
	clean_epoch.set_info()
	clean_epoch.annotation = default
	clean_epoch.color = 'white'
	return clean_epoch


def add_clean_epochs(artifacts, default,fo = None, minimal_duration = 500):
	'''Add clean epochs for stretches between bad_epochs.'''
	artifacts.sort()
	epochs = []
	b = bad_epoch2block(artifacts[0],fo)
	for i, a in enumerate(artifacts):
		# print(i,'--',a,b.duration_sample,a.et_sample,b.duration_sample-a.et_sample)
		if i == 0 and a.st_sample >=  minimal_duration: 
			# print('add start artifact')
			ep_id = '0.' + str(a.epoch_id)
			epochs.append(make_new_clean_epoch(ep_id,0,a.st_sample,a,default))
		if i != len(artifacts) -1: 
			ep_id = str(a.epoch_id) + '.' + str(artifacts[i+1].epoch_id)
			start = a.et_sample 
			end = artifacts[i+1].st_sample 
			epochs.append(make_new_clean_epoch(ep_id,start,end,a,default))
		if i == len(artifacts) -1 and minimal_duration <= b.duration_sample - a.et_sample: 
			# print('add end artifact')
			ep_id = str(a.epoch_id) + '.0'
			if hasattr(a,'block_et_sample'): end = a.block_et_sample - a.block_st_sample
			else: end = b.duration_sample
			epochs.append(make_new_clean_epoch(ep_id,a.et_sample,end,a,default))
	# print(artifacts,999999999)
	artifacts.extend(epochs)
	artifacts.sort()
	# for i,a in enumerate(artifacts):
		# print(i,a)
	return artifacts
			

def check_artifacts(artifacts,fo,default, minimal_duration = 500):
	for i,be in enumerate(artifacts):
		if i < len(artifacts) -1:
			if not minimal_duration <= artifacts[i+1].st_sample - be.et_sample: 
				print(artifacts[i+1].st_sample,be.et_sample)
				print(be,i)
				print
				print(artifacts[i+1])
			# assert minimal_duration <= artifacts[i+1].st_sample - be.et_sample 
	artifacts = add_clean_epochs(artifacts,default,fo = fo, minimal_duration = minimal_duration)
	for i,be in enumerate(artifacts):
		if be.annotation == default:
			if not be.duration >= minimal_duration: print(i,be)
			# assert be.duration >= 500


def load_bad_pp():
	exptype2int = {'o':1,'k':2,'ifadv':3}

	bad_pp = [line.strip().split('\t') for line in open(path.data + 'bad_pp_artifact_training.txt').read().split('\n') if line]
	bad_pp = [line[:-1] + list(map(int,line[-1].split(','))) for line in bad_pp]
	bad_pp = [[int(line[0]), exptype2int[line[1]], line[2]] for line in bad_pp]
	return bad_pp

def compute_artifact_clean_duration():
	exptype2int = {'o':1,'k':2,'ifadv':3}
	bad_pp = load_bad_pp()
	fn_xml = glob.glob(path.artifacts_clean + '*')
	fout = open('artifacts_duration.txt','w')
	for f in fn_xml:
		print(f)
		xml_artifacts = xml_handler.xml_handler(filename = f)
		xml_artifacts.load_xml()
		xml_artifacts.xml2bad_epochs()
		artifacts = xml_artifacts.bad_epochs
		be = artifacts[0]
		pp_id = be.pp_id
		exp_type = exptype2int[be.exp_type]
		bid = be.bid
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



def run(fo = None):
	'''Create new xml files with two classes of bad_epochs: clean and artifact. 
	If block is part of bad_pp_artifact training clean is changed with other
	all epochs are non overlapping and clean epochs are minimally 500ms in length.'''
	exptype2int = {'o':1,'k':2,'ifadv':3}
	bad_pp = load_bad_pp()
	fn = get_xml_files()
	all_clean, all_artifact = 0,0
	pp = {}

	for i,f in enumerate(fn):
		print(i,f)
		bad_epochs = get_bad_epochs(f)
		be = copy.deepcopy(bad_epochs[0])
		b = bad_epoch2block(be,fo)

		#check whether block is part of bad_pp -> non annotated data is not clean
		if [b.pp_id,exptype2int[b.exp_type],b.bid] in bad_pp: default = 'other'
		else: default = 'clean'

		artifacts = select_artifacts(f)
		if len(artifacts) == 0: 
			be.start.x = 0 
			be.end.x = b.duration_sample
			be.set_info()
			be.epoch_id = '0.' + be.epoch_id
			be.color = 'white'
			be.annotation = default
			artifacts = [be]
		else:
			artifacts = combine_overlaps(artifacts)
			stiches = stitch_artifacts(artifacts)
			stiches = stitch_stiches(stiches)
			artifacts = combine_artifacts(artifacts,stiches)
		
		check_artifacts(artifacts,fo,default)
		for a in artifacts:
			a.block_et_sample = b.et_sample

		if default == 'other': clean = 0
		else: clean = sum(make_duration_list(artifacts,default))
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
		x = xml_handler.xml_handler(artifacts)
		x.bad_epochs2xml()
		x.save(path.artifacts_clean + f.split('/')[-1])
	print(all_clean,all_artifact,all_clean/all_artifact)
	return pp
