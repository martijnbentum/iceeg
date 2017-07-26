import blinks
import glob
import path
import pickle
import random
import numpy as np

'''Loads all blink files from the blink folder and all manually classified blinks
blink files are pickled object created with the blink.py module
classification files are made with manually_classify_blinks.py, they are
text files with participant and experiment information and sample number with
respect to start recording and start block.  
'''

fn = glob.glob(path.blinks + '*.blinks')
fn_class = glob.glob(path.blinks + '*.classification')
i = 0

output = np.zeros([334644,1000])
info = np.zeros([334644,6],dtype = int)
exp2int = dict = {'ifadv':1,'o':2,'k':3}


def find_classification_file(f):
	'''Check wheter there is a classification file and return filename if there is.
	'''
	cf = f.strip('.blinks')
	for f in fn_class:
		if '_'.join(f.split('_')[:-2]) == cf:
			return f
	
def extract_class(f):
	'''Extract sample number from block and file onset and class code.'''
	if f.endswith('.blinks'): f = find_classification_file(f)
	if f == None: 
		print('no file found')
		return 0
	d = [line.split('\t')[-3:] for line in open(f).read().split('\n') if line]
	d = [list(map(int,line)) for line in d]
	for c in d:
		if c[-1] == 2: c[-1] = 0
		elif c[-1] == 0:c[-1] = -1
	print('returning classification data')
	return d

def normalize(epoch):
	'''Normalize block between 0 and 1. Add a small number to prevend zero devision
	and maybe problems with automatic clasification.
	'''
	if len(epoch) == 0: 
		epoch = np.zeros([1000]) + 0.00000001
	else:
		epoch = (epoch - min(epoch)) + 0.00000001 
		epoch = epoch / max(epoch)
	return epoch

for f in fn:
		'''Loop through all blink files and add the epochs to output and
		participant and experiment and class info to the info object.
		class info is not always present ~150 files manually classified out of
		~1500. Set class columns to -9
		sample number from file onset is also set to -9 if file is not classified,
		should change this, because i need it to create the right output once
		i automatically classified all blinks.
		'''
		print('extracting data from:',f)
		fin = open(f,'rb')
		b = pickle.load(fin)
		classes = extract_class(f)
		ppid = int(b.fn.split('_')[0].lstrip('pp'))
		exp_name = b.fn.split('_')[1]
		if '1' in exp_name: exp_name = 'k'
		exp = exp2int[exp_name]
		marker = int(b.fn.split('_')[-1].strip('.blinks'))
		print([ppid,exp,marker])
		print(i)
		print('___')
		if classes != 0:
			print('classes is not 0',i)
			assert len(classes) == len(b.peaks)
		for j,p in enumerate(b.peaks):
			epoch = normalize(b.veog[p-500:p+500])
			if p < 500:
				output[i,:len(epoch)] = epoch
			elif p > len(b.veog) - 500:
				output[i,1000-len(epoch):] = epoch
			else:
				output[i] = epoch
			if classes != 0:
				# print('putting in data',classes[j])
				assert classes[j][-3] == p
				info[i] = [ppid,exp,marker,p,classes[j][-2],classes[j][-1]]
			else:
				info[i] = [ppid,exp,marker,p,-9,-9]
			i += 1

		
'''Use numpy.save to save files to disk (pickle does not work with files > 2GB'''
fout_output = path.data + 'blinks_np_array1000_data'
fout_info = path.data + 'blinks_np_array1000_info'

np.save(fout_output,output)
np.save(fout_info, info)

