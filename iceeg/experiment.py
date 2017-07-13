import copy
import load_all_ort 
import session
import utils

class Experiment:
	'''Experiment object holds all participants of the experiment.'''

	def __init__(self,pp_ids= range(1,49),add_all_pp=False, multi_thread = False):
		'''Load information about participants, linking audio to EEG data.

		Keywords:
		pp_ids = range(1-48), participant numbers can be adapted to only include not excluded participants
		fid2ort = None, dictionary to map file id (from corpus) to ort object (transcription info)
		make_fid2ort=True, whether to make a new fid2ort object
		add_all_pp = False, whether to immediatly load all participants
		multi_thread = False, wether to use multi threading to load participants
		'''
		self.pp_ids = pp_ids
		self.pp = []


	def add_all_participants(self):
		'''Add all 48 participants.'''
		self.pp= []
		for pp_id in self.pp_ids:
			self.add_participant(pp_id = pp_id)


	def add_participant(self,pp_id = 1):
		'''Add a participant to the experiment object.

		You can pass a fid2ort dictionary and specify whether it should deepcopy it
		Copying the fid2ort takes 18 seconds. It is important to use a unique
		fid2ort for each participant.
		'''
		print('adding participant: '+str(pp_id))
		self.pp.append(Participant(pp_id))
		utils.make_attributes_available(self,'pp',[self.pp[-1]],False,str(pp_id))

	

class Participant:
	def __init__(self,pp_id = 1,fid2ort = None):
		'''Aggregate experiment data of one participant.

		Keywords:
		pp_id = participant number  int
		'''
		self.pp_id = pp_id
		if fid2ort != None: 
			assert type(fid2ort) == dict
			print('Beware that if you reuse fid2ort, sample information if overwritten.')
			self.fid2ort = fid2ort
		else: self.fid2ort = load_all_ort.load_fid2ort()
		self.exp_types = ['o','k','ifadv']
		self.sessions = []
		self.nwords = 0

	def __str__(self):
		pass

	def add_all_sessions(self):
		'''Add all experimental sessions.
		Each session a participant heard speech from a differen register: 
		o: Read aloud stories k: News broadcast ifadv: Spontaneous dialogues.'''
		self.sessions = []
		for exp_type in self.exp_types:
			self.add_session(exp_type = exp_type)
	
	def add_session(self,exp_type = 'o'):
		'''Add a experimental session to the participant object .'''
		self.sessions.append(session.Session(self.pp_id,exp_type,self.fid2ort))
		utils.make_attributes_available(self,'s',[self.sessions[-1]],False,exp_type)
		self.nwords += self.sessions[-1].nwords
			

		

'''
		nrecord = self.sessions[-1].vmrk.n_eeg_recordings
		setattr(self,'nrecordings_'+exp_type,nrecord)
		setattr(self,'nblock_'+exp_type, self.session[-1].nblock)
'''
