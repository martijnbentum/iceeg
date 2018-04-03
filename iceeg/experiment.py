import copy
import load_all_ort 
import session
import utils
import windower

class Experiment:
	'''Experiment object holds all participants of the experiment.'''

	def __init__(self,pp_ids= range(1,49),add_all_pp=False, multi_thread = False, fid2ort = None):
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
		self.sessions = []
		self.blocks = []
		self.nwords = 0
		if add_all_pp:
			self.add_all_participants(fid2ort = fid2ort)

	def __str__(self):
		m = 'n participants:\t' + str(len(self.pp)) + '\n'
		m += 'n words:\t' + str(self.nwords) + '\n'
		return m

	def __repr__(self):
		return 'experiment-object:\tn participants ' + str(len(self.pp)) + '\tnwords: ' + str(self.nwords)


	def add_all_participants(self, fid2ort = None):
		'''Add all 48 participants.'''
		self.pp= []
		for pp_id in self.pp_ids:
			self.add_participant(pp_id = pp_id, fid2ort = fid2ort)


	def add_participant(self,pp_id = 1, fid2ort = None):
		'''Add a participant to the experiment object.

		You can pass a fid2ort dictionary 
		It is important to use a unique
		fid2ort for each participant if you want to use accurate timesample data.
		'''
		print('adding participant: '+str(pp_id))
		self.pp.append(Participant(pp_id, fid2ort = fid2ort))
		utils.make_attributes_available(self,'pp',[self.pp[-1]],False,str(pp_id))

	def add_all_sessions(self):
		'''Add all session to each pp object.'''
		for p in self.pp:
			p.add_all_sessions()
			self.sessions.extend(p.sessions)
			self.blocks.extend(p.blocks)
		self.calc_nwords()

	def add_session(self, session_name = 'o'):
		'''Add a session to each pp object.'''
		for p in self.pp:
			p.add_session(session_name)
			self.sessions.extend(p.sessions)
			self.blocks.extend(p.blocks)
		self.calc_nwords()

	def calc_nwords(self):
		self.nwords = 0
		for p in self.pp:
			self.nwords += p.nwords


	def all_names(self):
		self.names = []
		for b in self.blocks:
			self.names.append(windower.make_name(b))
			

	

class Participant:
	def __init__(self,pp_id = 1,fid2ort = None):
		'''Aggregate experiment data of one participant.

		Keywords:
		pp_id = participant number  int
		'''
		self.pp_id = pp_id
		if fid2ort != None: 
			assert type(fid2ort) == dict
			print('Beware that if you reuse fid2ort, sample information is overwritten.')
			self.fid2ort = fid2ort
		else: self.fid2ort = load_all_ort.load_fid2ort()
		self.exp_types = ['o','k','ifadv']
		self.sessions = []
		self.nwords, self.nartifacts, self.total_duration, self.total_artifact_duration,self.artifact_perc = 0,0,0,0,0.0


	def __str__(self):
		m = 'pp-id:\t\t\t' + str(self.pp_id) + '\n'
		if hasattr(self,'sifadv'):  m += 'sifadv:\t\t\tavailable\n'
		else:  m += 'sifadv:\t\t\tNA\n'
		if hasattr(self,'sk'):  m += 'sk:\t\t\tavailable\n'
		else:  m += 'sk:\t\t\tNA\n'
		if hasattr(self,'so'):  m += 'so:\t\t\tavailable\n'
		else:  m += 'so:\t\t\tNA\n'
		m += 'nwords:\t\t\t' + str(self.nwords) + '\n'
		m += 'nartifacts\t\t'+str(self.nartifacts) + '\n'
		m += 'total dur\t\t'+str(int(self.total_duration)) + '\n'
		m += 'total artifact dur\t'+str(int(self.total_artifact_duration)) + '\n'
		m += 'artifact_perc\t\t'+str(round(self.artifact_perc,3)) + '\n'
		return m

	def __repr__(self):
		return 'participant-object:\tpp ' + str(self.pp_id) + '\t\tnwords ' + str(self.nwords) + '\tartifact perc: ' + str(round(self.artifact_perc,3))
		

	def add_all_sessions(self):
		'''Add all experimental sessions.
		Each session a participant heard speech from a differen register: 
		o: Read aloud stories k: News broadcast ifadv: Spontaneous dialogues.'''
		self.sessions = []
		self.blocks = []
		for exp_type in self.exp_types:
			self.add_session(exp_type = exp_type)
			self.blocks.extend(self.sessions[-1].blocks)
	
	def add_session(self,exp_type = 'o'):
		'''Add a experimental session to the participant object .'''
		self.sessions.append(session.Session(self.pp_id,exp_type,self.fid2ort))
		utils.make_attributes_available(self,'s',[self.sessions[-1]],False,exp_type)
		self.nwords += self.sessions[-1].nwords
		self.nartifacts += self.sessions[-1].nartifacts
		self.total_duration += self.sessions[-1].total_duration
		self.total_artifact_duration += self.sessions[-1].total_artifact_duration
		if self.total_duration > 0:
			self.artifact_perc = self.total_artifact_duration / self.total_duration
		else: self.artifact_perc = 0
		
		
			

		

'''
		nrecord = self.sessions[-1].vmrk.n_eeg_recordings
		setattr(self,'nrecordings_'+exp_type,nrecord)
		setattr(self,'nblock_'+exp_type, self.session[-1].nblock)
'''

