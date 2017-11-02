import data_stats 
import glob
import load_all_ort
import path
import pickle
import random
import numpy as np

'''Loads all data_stats files from the data_stats folder and all manually classified blinks
blink files are pickled object created with the blink.py module
classification files are made with manually_classify_blinks.py, they are
text files with participant and experiment information and sample number with
respect to start recording and start block.  
'''
class statsarray:
	'''Object to create a numpy matrix containing all data and info from datastats objects and  xml labeling files'''

	def __init__(self,coder = 'martijn',ncolumns_output = None, nrows_output = None, fo = None,output_fn = 'data_stats_lpc'):		
		'''Create statsarray based on all blocks that are labeled by a specific labeler
		coder 				name labeler
		ncolumns_output 	preset ncolumns to save computing time, other wise calculate based on data
		nrows_output 		preset nrows to save computing time, other wise calculate based on data
		fo 					provide fid2ort a datastructure needed to load blocks, can be provided to skip 15 sec loading time
		output_fn 			name of the data and info matrix
		'''

		if fo == None: fo = load_all_ort.load_fid2ort()
		self.fn_data_stats = glob.glob(path.data_stats + '*.data_stats')
		self.fn_annotation= glob.glob(path.artifacts+ 'martijn*.xlm')

		if ncolumns == None:
			#Calculate n columns data matrix
			ds = data_stats.load_data_stats(fn = fn_data_stats[0], fo = fo)
			self.nchannels = len(ds.ch_names)
			self.ncoefficients = len(ds.lpc_coefficients[0,0])
			self.ncolumns = nchannels * ncoefficients
			print('ncolumns output matrix: nchannels',nchannels,'* ncoefficients',ncoefficients, '=',ncolumns)
		else: ncolumns = ncolumns_output

		#Calculate n rows data matrix
		if nrows == None:
			total_nsnippets = 0
			for f in fn_data_stats:
				ds = data_stats.load_data_stats(fn = f,fo = fo)
				total_nsnippets += ds.nsnippets
			self.nrows = total_nsnippets
			print('nrows is:',nrows,'sum of all windows of all blocks of all participants')
		else: nrows = nrows_output


		self.output = np.zeros([nrows,ncolumns])
		self.info = np.zeros([nrows,6])
		self.exp2int = {'ifadv':1,'o':2,'k':3}
		self.annot2int = {'clean':0,'garbage':1,'unk':2,'drift':3,'other':4}


	def load_data(self)
		'''Load xml and datastats files, based on the annotated files'''
		for f in self.fn_data_stats:
				print('extracting data from:',f)
				ds = data_stats.load_data_stats(fn = f,fo = self.fo)
				exp_type = self.exp2int[ds.exp_type]
				marker = int(f.split('.')[0].split('m')[-1])
				annotation_fn = find_annotation_file(ds.pp_id,ds.exp_type,ds.bid, self.coder)
				if annotation_fn == 0: break
				annot = xml_handler.xml_handler(annotation_fn)
				annotations = annot.xml2bad_epochs() 
			
	def save_array(self):
		'''Use numpy.save to save files to disk (pickle does not work with files > 2GB'''
		fout_output = path.data + 'blinks_np_array1000_data'
		fout_info = path.data + 'blinks_np_array1000_info'

		np.save(fout_output,output)
		np.save(fout_info, info)


def find_annotation_file(pp_id,exp_type,bid, coder = 'martijn'):
	'''Check wheter there is a annotation file and return filename if there is.
	'''
	f = path.artifacts + coder +'_pp' + str(pp_id) + '_exp-' + exp_type + '_bid-' +str(bid)
	if os.path.isfile(f): return f
	else: 
		print('file:',f,'not found')
		return 0
	

def normalize(d):
	'''Normalize data between 0 and 1. Add a small number to prevend zero devision
	and maybe problems with automatic clasification.
	'''
	if len(d) == 0: 
		d = np.zeros([1000]) + 0.00000001
	else:
		d = (d - min(d)) + 0.00000001 
		d = epoch / max(d)
	return d


def compute_overlap(start_a,end_a,start_b, end_b):
	'''Also defined in the datastat class, unsure what is the best location
	computes n samples a overlaps with b
	'''
	if end_a < start_a:
		raise ValueError('first interval is invalid, function assumes increasing intervals',start_a,end_a)
	if end_b < start_b:
		raise ValueError('second interval is invalid, function assumes increasing intervals',start_b,end_b)
	if end_b < start_a or start_b > end_a: return 0 # b is completely before or after a
	elif start_a == start_b and end_a == end_b: return end_a - start_a # a and b are identical
	elif start_b < start_a: # first statement already removed b cases completely before a
		if end_b < end_a: return end_b - start_a # b starts before a and ends before end of a	
		else: return end_a - start_a # b starts before a and ends == or after end of a
	elif start_b < end_a: # first statement already romve b cases completely after a
		if end_b > end_a: return end_a - start_b # starts after start of a and ends == or after end of a
		else: return end_b - start_b  # b starts after start of a and ends before end of a #
	else:  print('error this case should be impossible')
