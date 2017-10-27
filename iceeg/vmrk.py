'''vmrk module loads and process BrainVision marker files

vmrk module contains the vmrk class
'''

import glob
import numpy as np
import path
import re

class vmrk:
	'''Class to aggregate information about markers and sample numbers.

	.marker2samplen a dict that maps marker number to sample number 
		this aligns onset experimental audio file to EEG data
	.events (obsolete) makes a np array compatible with MNE for all markers
	'''

	def __init__(self,pp_id = None,exp_type = None,verbose = False):
		'''Load and process BrainVision marker files

		Keywords:
		pp_id = participant number int
		exp_type = experimental type (o/k/ifadv) reflect register of audio file str
		path = default = parent directory
		'''
		self.verbose = verbose
		if self.verbose:
			print('loading vmrk with:\t',pp_id,exp_type,path.data)
		self.pp_id = pp_id
		self.exp_type = exp_type
		self.read_vmrk()
		self.vmrk2events()# creates an np array events from 1 or more vmrk lists
		self.vmrk2dict() # creates a dict with markers as keys and sample numbers as values
		self.set_markers()
		self.set_missing()


	def __str__(self):
		m = 'vmrk filename:\t\t' + str(self.vmrk_fn) + '\n'
		return m


	def find_vmrk_filename(self):
		'''find vmrk filename based on pp_id and exp type, sets to none if it fails
		will set n eeg recordings according to n matches it finds
		sometimes 1 EEG session was recorded in multiple file (because of battery failure)
		'''
		pp_id = str(self.pp_id)
		fn = glob.glob(path.eeg + '/pp*' + pp_id + '_' \
			+ self.exp_type + '*.vmrk')
		matches = []
		for f in fn:
			temp = re.match(r".*pp0*("+pp_id+")_.*\.vmrk", f)
			if temp != None:
				matches.append(temp.string)
		# handle filename based on the number of matching files found
		if len(matches) == 0:
			print('Could not find vmrk file with pp_id',self.pp_id,fn)
			self.vmrk_fn = None
			self.n_eeg_recordings = 0
		elif len(matches) == 1:
			self.vmrk_fn = matches[0]
			self.n_eeg_recordings = 1
		else:
			self.vmrk_fn = matches
			self.n_eeg_recordings = len(matches)


	def read_vmrk(self):
		'''read in the marker file and deal with multiple recordings in 1 session'''
		if not self.pp_id or not self.exp_type:
			print("need pp_id and exp_type")
			self.vmrk_fn= None
			self.session = None
			self.vmrk = None
			return None
		self.find_vmrk_filename()						
		if self.verbose:
			print('loading vmrk: \t\t',self.vmrk_fn)
		if self.n_eeg_recordings == 1:
			temp = [line.split(',') for line in open(self.vmrk_fn).read().split('\n')]
			self.vmrk= [line for line in temp if line[0][:len('Mk')] == 'Mk']
		elif self.n_eeg_recordings > 1:
			self.vmrk = []
			for vmrk_fn in self.vmrk_fn:
				temp = [line.split(',') for line in open(vmrk_fn).read().split('\n')]
				vmrk= [line for line in temp if line[0][:len('Mk')] == 'Mk']
				self.vmrk.append( vmrk )
		else:
			self.vmrk = None
			

	def make_events(self,vmrk):
		'''Create an np array that is compatible with MNE from vmrk list. 

		vmrk is the marker file split on lines and tabs 
		Return np array of dimension: number_of_markers X 3
		columns: sample number, 0, marker 
		'''
		if not self.pp_id or not self.exp_type or not self.vmrk:
			self.events = None
		output = []
		for line in vmrk:
			if line[1]:
				event_id = int(line[1].strip('S').strip(' '))
				sample_number = int(line[2])
				output.append([sample_number,0,event_id])
		np_out = np.asarray(output)	
		return np_out


	def vmrk2events(self):
		'''Create an np array events from 1 or more vmrk lists.'''
		if not self.vmrk:
			print('Could not create events array, vmrk == None')
			return None
		if self.verbose:
			print('Creating events array:\t sample_number , 0 , marker')
		if self.n_eeg_recordings == 1:
			self.events = self.make_events(self.vmrk)
		else:
			self.events = []
			self.events = [self.make_events(vmrk) for vmrk in self.vmrk]
			

	def vmrk2dict(self):
		'''Create a dict with markers as keys and sample numbers as values.'''
		if not self.vmrk:
			print('Could not create marker dict, self.vmrk == None')
			return None
		if self.verbose:
			print("Creating marker2samplen:\t marker -> sample number")
		if self.n_eeg_recordings == 1:
			events = self.events.tolist() 
			self.marker2samplen = dict([[l[2],l[0]] for l in events])
			self.marker2vmrk_fn = dict([[l[2],self.vmrk_fn] for l in events])
		else:
			self.marker2samplen = {} 
			self.marker2vmrk_fn = {}
			for i,events in enumerate(self.events):
				events = events.tolist()
				self.marker2samplen.update( dict([[l[2],l[0]] for l in events]) )
				self.marker2vmrk_fn.update( dict([[l[2],self.vmrk_fn[i]] for l in events]) )
				 

	def set_markers(self):
		'''Make sets of start and end markers based on type of experiment.'''
		if self.exp_type == 'k':
			self.smarkers = set(range(10,220,10))
			self.emarkers = set(range(11,221,10))
		if self.exp_type == 'o':
			self.smarkers = set(range(10,80,10))
			self.emarkers = set(range(11,81,10))
		if self.exp_type == 'ifadv':
			self.smarkers = set(range(10,70,10))
			self.emarkers = set(range(11,71,10))


	def set_missing(self):
		'''Make a lists of missing start and end markers.'''
		self.missing_smarkers = self.smarkers - set(self.marker2samplen.keys())
		self.nmissing_smakers = len(self.missing_smarkers)
		self.missing_emarkers = self.emarkers - set(self.marker2samplen.keys())
		self.nmissing_emakers = len(self.missing_emarkers)
		self.all_markers_present = self.nmissing_smakers == self.nmissing_emakers == 0



