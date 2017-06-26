import glob
import numpy as np
import re
PATH = '../'

class vmrk:
	def __init__(self,pp_id = None,exp_type = None,path = None):
		print('loading vmrk with:\t',pp_id,exp_type,path)
		if not path: self.path = PATH
		else: self.path = path
		self.pp_id = pp_id
		self.exp_type = exp_type
		self.path = path
		self.read_vmrk()
		self.vmrk2events()# creates an np array events from 1 or more vmrk lists
		self.vmrk2dict() # creates a dict with markers as keys and sample numbers as values


	def __str__(self):
		m = '\nVMRK OBJECT\n'
		m += 'vmrk filename:\t\t' + str(self.vmrk_fn) + '\n'
		# fields = ('pp_id exp_type vmrk events marker_dict duration').split(' ')
		# m += 'FIELDS:\n' + '\t'.join(fields)
		return m


	def find_vmrk_filename(self):
		# finds the filename based on pp_id and exp type, sets to none if it fails
		# will set n eeg recordings according to n matches it finds
		# sometimes 1 session was recorded in multiple file (because of battery failure)
		pp_id = str(self.pp_id)
		fn = glob.glob(self.path + 'EEG/pp*' + pp_id + '_' \
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
		# reads in the marker file and deals with multiple recording in 1 session
		if not self.pp_id or not self.exp_type:
			print("need pp_id and exp_type")
			self.vmrk_fn= None
			self.session = None
			self.vmrk = None
			return None
		self.find_vmrk_filename()						
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
		# creates an np array from vmrk list
		# called by vmrk2events expects raw vmrk list
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
		# creates an np array events from 1 or more vmrk lists
		if not self.vmrk:
			print('Could not create events array, vmrk == None')
			return None
		print('Creating events array:\t sample_number , 0 , marker')
		if self.n_eeg_recordings == 1:
			self.events = self.make_events(self.vmrk)
		else:
			self.events = []
			self.events = [self.make_events(vmrk) for vmrk in self.vmrk]
			

	def vmrk2dict(self):
		# creates a dict with markers as keys and sample numbers as values
		if not self.vmrk:
			print('Could not create marker dict, self.vmrk == None')
			return None
		print("Creating marker2samplen:\t marker -> sample number")
		if self.n_eeg_recordings == 1:
			events = self.events.tolist() 
			self.marker2samplen = dict([[l[2],l[0]] for l in events])
		else:
			self.marker2samplen = {} 
			for events in self.events:
				events = events.tolist()
				self.marker2samplen.update( dict([[l[2],l[0]] for l in events]) )
				 
