import experiment as e
import math
import numpy as np
import path
import progressbar as pb
import utils

wtd = utils.load_dict_wordtype2freq() 

class dataset():
	'''Object to create a dataset for statistical analysis based on dataset type.
	only defined type for now is n400, define others to extract the relevant eeg data.
	Exclude bad data
	'''
	def __init__(self, dataset_type = 'n400',participant = 'all', filename = '',nprevious = 0):
		'''Create object to hold data, it works based to make it per participant and save each and combine afterwards.

		dataset_type 		type that defines the eeg data that is extracted
		participant 		all, list of number or number (participant numbers range(1,49)
		'''
		self.dataset_type = dataset_type
		self.filename = path.datasets + filename + '_' + self.dataset_type + '.dataset' 
		self.participant = participant
		self.nprevious = nprevious
		self.pp, self.blocks, self.bad_blocks = [], [], []
		self.datawords, self.excluded_words = [], []
		self.nwords, self.nexcluded_words = 0, 0
		self.excluded_blocks = []
		if dataset_type == 'n400': self.channel_set = utils.n400_channel_set()
		else: self.channel_set = utils.load_selection_ch_names()
		self.made = False
		self.saved = False

	def __repr__(self):
		m = 'dataset\tnwords: ' + str(self.nwords)+ '\tnexcluded words: ' 
		m += str(self.nexcluded_words) + '\ttype: ' + self.dataset_type
		m += '\tmade: ' + str(self.made) + '\tsaved: ' + str(self.saved)
		m += '\tfilename: ' + self.filename
		return m

	def __str__(self):
		return self.__repr__().replace('\t','\n')

	def _load_participant(self,number):
		'''Load a specific participant, participant numbers 1 - 48.'''
		self.current_p = e.Participant(number)
		self.current_p.add_all_sessions()
		self.pp.append(self.current_p)


	def _handle_blocks(self):
		'''Handle all blocks of participant.
		checks whether a block is usable (otherwise skipped).
		'''
		chs, dst = self.channel_set, self.dataset_type
		for session in self.current_p.sessions:
			for b in session.blocks:
				if not hasattr(b,'xml') or not b.xml.usability in ['great','ok','mediocre']: 
					self.excluded_blocks.append(b)
					continue
				db = datablock(b,chs,dst,nprevious = self.nprevious)
				self.datawords.extend(db.datawords)
				self.excluded_words.extend(db.excluded_words)


	def _handle_particpants(self):
		'''Handles participants (if number only 1) (if list all numbers in list) (if all, all participants).
		best to create dataset per participant and save each and combine afterwards.
		'''
		for i in self.pp_numbers:
			self._load_participant(i)
			self._handle_blocks()
		self.nwords = len(self.datawords)
		self.nexcluded_words = len(self.excluded_words)

	def make(self, save = False):
		'''Actually create the dataset and optionally save it.'''
		if self.participant == 'all':self.pp_numbers= list(range(1,49))
		elif type(self.participant) == int: self.pp_numbers = [self.participant]
		elif type(self.participant) == list: self.pp_numbers = self.participant
		else: 
			print('participant should be: all, list of int, int (number of participant')
			print(p.participant,'\n','handling all participants')
			self.pp_numbers = list(range(1,49))
		self._handle_particpants()
		self.made = True
		if save: self.save()
		else: self.saved = False

	def save(self):
		'''Save the dataset.'''
		if len(self.datawords) == 0: 
			self.output = ['no_data']
			self.filename += '_no_data'
		else:
			header, values = self.datawords[0].make_all_line()
			self.output = ['\t'.join(header)]
			for dw in self.datawords:
				header, values = dw.make_all_line()
				self.output.append('\t'.join(values))
		with open(self.filename,'w') as fout:
			fout.write('\n'.join(self.output))
		self.saved = True
			


class datablock():
	'''Object that creates the information for the dataset on block level (see block.py).
	EEG data is organized in blocks which is accessible in the block object, part of a session part of participant
	each session is an experiment and each block is an audiofile, each block contains word objects
	which contain information for eeg registration and other word information (frequency surprisal usability)
	'''
	def __init__(self, b, chs, dst, nprevious = 0):
		if not hasattr(b,'xml'): 
			print('block does not have usability information, not append to dataset',b.name)
			return 
		if not hasattr(b,'raw'): b.load_eeg_data()
		self.b = b
		self.data = self.b.raw[:][0] * 10 **6
		self.channel_set = chs
		self.dataset_type = dst
		self.nprevious = nprevious
		self.extract_info()
		self.b.unload_eeg_data()
		delattr(self,'data')
		self.nwords = len(self.datawords) 
		self.nexcluded_words = len(self.excluded_words)


	def __repr__(self):
		m = 'datablock\twords: '+ str(self.nwords)
		m +='\texcluded: '+str(self.nexcluded_words)
		m += '\t'+self.b.name
		return m

	def __str__(self):
		m = 'datablock\n'
		for k in self.__dict__.keys():
			try:value = str(self.__dict__[k])
			except: continue
			if len(value) > 100: value = value[:80] + '...'
			m += k + '\t' + value + '\n'
		return m


	def extract_info(self):
		if self.dataset_type == 'n400': self._make_eeg_baseline_n400()
		else: 
			print('you requested',self.dataset_type,'this is not implemented')
			self._make_eeg_baseline_n400()


	def _select_channels(self):
		self.bad_channels = self.b.raw.info['bads']
		self.selected_channels = [ch for ch in self.channel_set if ch not in self.bad_channels] 
		all_ch = utils.load_ch_names()
		self.channels_indices = [i for i,ch in enumerate(all_ch) if ch in self.selected_channels]
		self.channel_mask= [ch in self.selected_channels for ch in all_ch]
		self.nselected_channels = len(self.selected_channels)
		self.nbad_channels = len(self.bad_channels)


	def _make_eeg_baseline_n400(self):
		self._select_channels()
		self.datawords = []
		self.excluded_words = []
		bar = pb.ProgressBar()
		bar(range(len(self.b.words)))
		print('handling block:',self.b.name)
		for i,w in enumerate(self.b.words):
			bar.update(i)
			if not w.usable or not hasattr(w,'pos') or not w.pos.content_word:
				self.excluded_words.append(self.b.name + '_wi-'+str(i))
				continue
			dw  = dataword(w,self,i,nprevious = self.nprevious)
			if dw.ok:
				self.datawords.append(dw)
			else:self.excluded_words.append(self.b.name + '_wi-'+str(i))



class dataword():
	'''Create a line in the dataset based on a word in the experiment. (see word.py)
	the word object contains info about the eeg registration and other info (frequency surprisal usability).
	'''
	def __init__(self,w,db,index,keep_traces = False, prev = False, b = None, nprevious = 0):
		'''Dataword object that holds info for a line in the dataset
		w 		word object (word.py)
		db 		datablock (contains block (block.py))
		index 	word index in block 0 = first word
		keep... whether to keep the extract eeg datablock (if true make sure you have large ram)
		prev 	each word has a history of preceding words that is also a dataword object, however these words are prev true 
		b 		optionally provide block object (also present on datablock object), used for prev datawords
		'''
		self.ok = True
		self.w = w
		self.db = db 
		if b == None: self.b = self.db.b
		else: self.b = b
		self.nprevious = nprevious
		self.index = index
		self.keep_traces = keep_traces
		self.word = self.w.word_utf8_nocode_nodia().lower()
		self.word_in_block= str(index + 1)
		self.prev = prev
		self._extract_info()
	

	def __repr__(self):
		m = 'dataword\t'+self.word+'\t'+self.register + '\t' + self.word_in_block
		if not self.prev:
			m += 'threshold_ok: ' + str(self.threshold_ok)
		else: m = 'prev '+m
		return m

	def __str__(self):
		output = []
		for item in zip(self.header(),self.values()):
			output.append(':\t'.join(item))
		return '\n'.join(output)


	def _extract_eeg(self):
		'''Extract the eeg data from the block
		The eeg data has been check for artefacts (stretches and channels)
		Eye related artefacts are removed with ICA
		This function will threshold the EEG data at +/- 75 mu volt (excludes the data if it exceeds).
		WIP: make extract_eeg dataset_type dependend - only n400 implemented (channel set / time window)
		'''
		w = self.w
		d = self.db.data

		self.stepoch = w.st_epoch - w.sample_offset 
		self.etepoch = w.et_epoch - w.sample_offset 
		epoch = d[self.db.channel_mask,self.stepoch:self.etepoch] 
		if self.stepoch < 0 or self.etepoch > d.shape[1]: 
			self.ok = False
			return False
		self.threshold_ok = str(np.min(epoch) > -75 and np.max(epoch) < 75)

		self.stn400 = w.st_sample + 300 - w.sample_offset 
		self.etn400 = w.st_sample + 500 - w.sample_offset 

		n400 = d[self.db.channel_mask,self.stn400:self.etn400] 
		self.n400_trace = sum(n400) / self.db.nselected_channels
		self.n400_avg=  sum(self.n400_trace) / self.n400_trace.shape[0]

		self.stbaseline= w.st_sample - 150 - w.sample_offset 
		self.etbaseline= w.st_sample - w.sample_offset 

		baseline= d[self.db.channel_mask,self.stbaseline:self.etbaseline] 
		self.baseline_trace = sum(d[self.db.channel_mask,] / self.db.nselected_channels)
		self.baseline_avg=  sum(self.n400_trace) / self.baseline_trace.shape[0]

		self.n400 = str(self.n400_avg)
		self.baseline = str(self.baseline_avg)
		self.word_onset = str(self.w.st_sample)
		self.sample_offset = str(self.w.sample_offset)
		if not self.keep_traces:
			delattr(self,'n400_trace')
			delattr(self,'baseline_trace')


	def _extract_logprob(self):
		'''Extract the logprob computed by SRILM for cow (general web corpus or register specific corpus).
		cow corpus is a large collection of web text (~5 billion words)
		register specific corpus, text similar to the speech style of the experiment (experimental text not part of corpus)
		SML trained on cow
		SML trained on register specific corpus and subsequently a version is made that is an interpolation
		of cow and register specific corpus, this version is used
		surprisal value of a word given the preceding words given a SLM (cow / register specific)
		'''
		if not hasattr(self.w,'ppl'): 
			self.logprob = 'na'
			self.logprob_register = 'na'
			self.logprob_other = 'na'
			self.logprob_cache= 'na'
			self.ok = False
			print(self.w,'no logprob')
		elif '-inf' in self.w.ppl.word_line: 
			self.logprob = '-10'
			self.logprob_register = '-10'
			self.logprob_other= '-10'
			self.logprob_cache= '-10'
		else: 
			self.logprob = str(self.w.ppl.logprob)
			self.logprob_register = str(self.w.ppl.logprob_register)
			self.logprob_other = str(self.w.ppl.logprob_other1)
			self.logprob_cache= str(self.w.ppl.logprob_cache)
		

	def _extract_frequency(self):
		'''Frequency is based on the cow corpus.'''
		if self.word in wtd.keys():
			self.freq_raw = int(wtd[self.word]) + 1
			self.freq = str(self.freq_raw)
			self.freq_log = str(math.log(self.freq_raw))
		else:
			self.freq_raw, self.freq_log, self.freq = 'na','na','na'
			print(self.word, 'not in dictionary',self.__repr__())


	def _extract_info(self):
		'''set relevant information.'''
		if not self.prev: 
			self._extract_eeg()
			self.rejected_channels = ','.join(self.db.bad_channels)
			self.selected_channels = ','.join(self.db.selected_channels)
			self.nch = str(self.db.nselected_channels)
			self.nch_rm = str(self.db.nbad_channels)
		if not self.ok: return
		self._extract_logprob()
		self.pp_id = str(self.b.pp_id)
		self.exp = str(self.b.exp_type)
		self.bid = str(self.b.bid)
		self.fid = self.w.fid
		self.sid = self.w.sid
		self.register = self.w.register
		self.word_duration = str(self.w.duration_sample)
		self.word_in_sentence = str(self.w.word_number + 1)
		if hasattr(self.w,'pos'):
			self.sentence_in_block = str(self.w.pos.sentence_number + 1)
			self.content_word = str(self.w.pos.content_word)
		else: self.sentence_in_block, self.content_word = 'na','na'
		self.usable = str(self.w.usable)
		self.name = self.b.name
		if self.nprevious > 0: self._set_previous_words()
		else: self.nprevious_words = '0'
		self._extract_frequency()

		
	def _set_previous_words(self):
		'''Set information of the preceding words, can be usefull for analysis.'''
		if not self.ok or self.prev: return
		start_index = self.index - self.nprevious
		if start_index < 0: start_index = 0
		self.previous_words = self.db.b.words[start_index:self.index]
		self.nprevious_words = str(len(self.previous_words))
		index = start_index
		self.previous_dw = []
		for w in self.previous_words:
			self.previous_dw.append(dataword(w,None,index,prev = True,b =self.b))
			index += 1
			

	def values(self, word = 'target'):
		'''Extract the values given the header to make the dataset line.'''
		h = self.header(word)
		return [self.__dict__[name] for name in h]
		

	def header(self,word = 'target'):
		'''Set the header of the values that need to be extracted from the dataword.'''
		if word == 'target':
			h = 'word,baseline,n400,freq,freq_log,pp_id,exp,bid,word_duration'
			h += ',logprob,logprob_register,logprob_other,logprob_cache'
			h += ',nch,nch_rm,rejected_channels,selected_channels,content_word'
			h += ',usable,threshold_ok,word_in_block,word_in_sentence,sentence_in_block'
			h += ',fid,sid,register,word_onset,sample_offset,name,nprevious_words'
			return h.split(',')
		if 'prev' in word:
			h = 'word,freq,freq_log,pp_id,exp,word_duration,content_word,word_in_block'
			return h.split(',') 
		
		 
	def make_all_line(self):
		'''Make a line that includes all info from a dataword and the three preceding words.'''
		if self.prev: return False
		header = self.header()
		values = self.values()
		if self.nprevious == 0: return header, values
		prev_h, prev_v = [], []
		n = len(self.previous_dw)
		for i,w in enumerate(self.previous_dw):
			prev_h.append(['prev' + str(n-i) + cn for cn in w.header('prev')])
			prev_v.append(w.values('prev'))
		temp_h, temp_v = [], []
		for i in range(self.nprevious-n):
			temp_h.append(['prev' + str(self.nprevious-i) + cn for cn in  self.header('prev')])
			temp_v.append(['na'] * len(temp_h[-1]))
		if len(temp_h) > 0:
			prev_h = temp_h + prev_h
			prev_v = temp_v + prev_v
		[header.extend(ph) for ph in prev_h]
		[values.extend(pv) for pv in prev_v]
		return header, values
		 

	

