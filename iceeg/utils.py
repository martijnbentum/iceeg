import experiment as e
import numpy as np
import path

'''general functions, not specific to an object

There are likely many functions now defined on objects that should be here
Work In Progress
'''
all_block_names = open(path.data + 'all_block_names.txt').read().split('\n')

exptype2explanation_dict = {'o':'read-aloud-books','k':'News-broadcast','ifadv':'spontaneous-dialogue'}

def name2pp_id(name):
	'''Extract pp id from name (windower.make_name(b)).'''
	return int(name.split('_')[0].strip('pp'))

def name2exp_type(name):
	'''Extract exp type from name (windower.make_name(b)).'''
	return name.split('_')[1].strip('exp-')

def name2bid(name):
	'''Extract block id from name (windower.make_name(b)).'''
	return int(name.split('_')[2].strip('bid-'))

def name2block(name, fo = None):
	'''Based on the name made by the windower object, create and return the block object.'''
	pp_id = name2pp_id(name)
	exp_type = name2exp_type(name)
	bid = name2bid(name)

	p = e.Participant(pp_id,fid2ort = fo)
	p.add_session(exp_type)
	s = getattr(p,'s'+exp_type)
	return getattr(s,'b'+str(bid))
	

def bad_epoch2block(be,fo = None):
	'''Return block object that correspond to the bad_epoch.'''
	p = e.Participant(be.pp_id,fid2ort = fo)
	p.add_session(be.exp_type)
	s = getattr(p,'s' + be.exp_type)
	return getattr(s, 'b' + str(be.bid))

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

def load_ch_names():
	return open(path.data + 'channel_names.txt').read().split('\n')

def load_100hz_numpy_block(name):
	return np.load(path.eeg100hz + name + '.npy')

exptype2int = {'o':1,'k':2,'ifadv':3}
annot2int = {'clean':0,'garbage':1,'unk':2,'drift':3,'other':4}

def make_attributes_available(obj, attr_name, attr_values,add_number = True,name_id = '',verbose = False):
	'''make attribute available on object as a property
	For example if attr_name is 'b' attr_value(s) can be accessed as: .b1 .b2 .b3 etc.

	Keywords:
	obj = the object the attributes should be added to
	attr_name = is the name the attributes should accessed by (see above)
	attr_values = list of values (e.g. a list of block objects)
	'''
	if type(attr_values) != list:
		# values should be provided in a list
		print('should be a list of value(s), received:',type(attr_values))
		return 0
	if len(attr_values) == 0:
		# Check for values
		print('should be a list with at leat 1 item, received empty list',attr_values)
		return 0

	# Make property name
	if add_number:
		# Add a number to the property name: .b1,.b2 etc.
		if verbose:
			print('Will add a number to:',attr_name,' for each value 1 ... n values')
		if name_id != '':
			print('Number is added to property, name id:',name_id,' will be ignored')
		if len(attr_values) > 1:
			property_names = [attr_name +str(i) for i in range(1,len(attr_values)+ 1)]
		else: property_names = [attr_name + '1']

	elif len(attr_values) > 1:
		print('add_number is False: you should only add one value otherwise you will overwrite values')
		return 0

	else:
		# Add name_id to property name
		if hasattr(obj,attr_name + name_id):
			print('object already had attribute:',attr_name,' will overwrite it with new value')
			print('Beware that discrepancies between property:', attr_name, ' and list of objects could arise')
			print('e.g. .pp1 could possibly not correspond to .pp[0]')
		property_names = [attr_name+name_id]

	# add value(s) to object 
	[setattr(obj,name,attr_values[i]) for i,name in enumerate(property_names)]

	#Add list of attribute names to object
	pn = 'property_names'
	if not attr_name.endswith('_'): pn = '_' + pn

	if hasattr(obj,attr_name + pn) and not add_number:
		# if no number the list of attribute names could already excist
		getattr(obj,attr_name + pn).extend(property_names)
	else:
		# otherwise create the list
		setattr(obj,attr_name + pn,property_names)

	if verbose:
		print('set the following attribute names:')
		print(' - '.join(property_names))


def make_events(start_end_sample_number_list):
	'''Make np array compatible with MNE EEG toolkit.

	assumes a list of lists with column of samplenumbers and a column of ids  int

	structure:   samplenumber 0 id_number
	dimension 3 X n_events.
	WORK IN PROGRESS
	'''
	if set([len(line) for line in start_end_sample_number_list]):
		return np.asarray(output)	


def get_path_blockwavname(register, blockwav_name ):
	'''Return wavname corresponding to register and blockwav_name.

	blockwav_name is the filename of the experiment audio file.
	'''
	print(register,blockwav_name)
	if register == 'spontaneous_dialogue':
		path = '/Users/Administrator/storage/EEG_study_ifadv_cgn/IFADV/'
	elif register== 'read_aloud_stories':
		path = '/Users/Administrator/storage/EEG_study_ifadv_cgn/comp-o/'
	elif register == 'news_broadcast':
		path = '/Users/Administrator/storage/EEG_study_ifadv_cgn/comp-k/'
	else: raise Exception('Unknown register:',register)

	blockwav_path = path + blockwav_name
	return blockwav_path


def get_path_fidwav(register, fid):
	'''Return wavname corresponding to register and file id.'''
	if register == 'spontaneous_dialogue':
		path = '/Users/Administrator/storage/EEG_study_ifadv_cgn/IFADV/'
	elif register== 'read_aloud_stories':
		path = '/Users/Administrator/storage/cgn_audio/comp-o/nl/'
	elif register == 'news_broadcast':
		path = '/Users/Administrator/storage/CGN/comp-k/'
	else: raise Exception('Unknown register:',register)

	fn = glob.glob(path + fid + '*')
	if len(fn) == 1: wavname = fn[0]
	else:
		print('Could not find:',fn,' in:',path)
		return ''
	return path + wavname


def get_start_end_times_relative2blockwav(b, item, sf=1000):
	'''Return start end times of item in relation to blockwav.

	Samplenumbers are relative to start experimental audio file
	for both comp-o and comp-k multiple corpus audiofiles were used
	to create the experimental audio file.

	sample frequency = 1000
	'''
	start_block = b.st_sample
	start_time = item.st_sample
	
	end_time = item.et_sample

	start_sec = (start_time - start_block) / sf 
	end_sec = (end_time - start_block) / sf
	
	return start_sec,end_sec


def extract_audio(b, item, filename = 'default_audio_chunk'):
	'''Extract part from audio file.

	part is specified by item, can be word, chunk or sentence
	block info is needed to find times relative to onset experimental audio file.
	
	wave currently has has a namespace clash with local chunk
	wave imports chunk and my local chunk takes precedence.
	'''
	import sys
	save_path = sys.path[:]
	sys.path.remove('')
	import wave
	sys.path = save_path

	if not filename.endswith('.wav'): filename += '.wav'
	wavname = get_path_blockwavname(b.register, b.wav_filename)
	start,end = get_start_end_times_relative2blockwav(b, item)
	print('Audio name:',wavname,'start/end:',start,end)

	audio = wave.open(wavname,'rb')
	framerate = audio.getframerate()
	nchannels = audio.getnchannels()
	sampwidth = audio.getsampwidth()

	audio.setpos(start * framerate)
	chunk = audio.readframes(int((end-start) * framerate))

	chunk_audio = wave.open(filename,'wb')
	chunk_audio.setnchannels(nchannels)
	chunk_audio.setsampwidth(sampwidth)
	chunk_audio.setframerate(framerate)
	chunk_audio.writeframes(chunk)
	chunk_audio.close()
	print('Extracted from:',wavname,'start/end:',start,end,'written to:',filename)
	del wave








		
