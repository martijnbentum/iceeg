import experiment

class pmn_data():
	'''Holds all words of all experiments with relevant information for PMN experiment.
	'''
	def __init__(self,p = None,  history_length = 4):
		'''Hold all words of all experiments with PMN information (phonological mismatch negativity)
		p 		participant object with all sessions loaded p = e.Participant(1) p.add_all_sessions()
		his... 	number of previous words added to the pmn word (to determine entropy) 
		'''
		self.p = p
		self.historty_length = history_length
		self.pmn_words = []
		if not p: 
			self.p = experiment.Participant(1)
			self.p.add_all_sessions()
		self.make_data()

	def make_data(self):
		'''Create pmn_word object for each word.'''
		for session in self.p.sessions:
			for block in session.blocks:
				for i,word in enumerate(block.words):
					start = i - self.historty_length
					if start < 0 : start = 0
					precontext = block.words[start:i]
					precontext = [w.word_utf8_nocode for w in precontext]
					self.pmn_words.append(pmn_word(word,precontext))

	def make_chunk_speaker_file(self,offset = 0.5, small = False):
		'''Create lists with information for KALDI.
		offset 		n seconds after word onset
		small 		whether to create a limited set of examples (n = 100) for testing
		output:
		self.speaker 		speaker followed by corresponding filenames (speakers).
		self.chunks 			chunk name, filename, start, end
		'''
		self.speaker2f = {}
		self.chunks = []
		self.speakers = []
		self.small = small
		for i,w in enumerate(self.pmn_words):
			self.chunks.append(w.format(i+1,offset=offset))
			if w.word.sid not in self.speaker2f.keys():
				self.speaker2f[w.word.sid] = [self.chunks[-1].split(' ')[0]]
			else: self.speaker2f[w.word.sid].append(self.chunks[-1].split(' ')[0])
			if small and i+1 == 100:
				break
		for s in self.speaker2f.keys():
			self.speakers.append(' '.join([s+'-S0'] + self.speaker2f[s]))
			

	def save(self):
		'''Save KALDI helper files created with make_chunk_speaker_file().'''
		fc = 'chunks.txt'
		if self.small: fc = 'chunks_small.txt'
		fs = 'speakers.txt'
		if self.small: fs = 'speakers_small.txt'
		with open(fc,'w') as fout: fout.write('\n'.join(self.chunks))
		with open(fs,'w') as fout: fout.write('\n'.join(self.speakers))
		

class pmn_word():
	'''Holds information about each word necessary for the PMN experiment.'''
	def __init__(self,word,precontext):
		'''Holds information about each word necessary for the PMN experiment.
		word 		word object (see word.py)
		precontext 	preceding words (as strings)
		'''
		self.word = word
		self.precontext = precontext
		self.word_str = word.word_utf8_nocode
		self.fid = word.fid
		self.start = word.st
		self.t_zero= word.st
		self.start_str = str(self.start)
		self.pc_string = ','.join(precontext)

	def __repr__(self):
		return '\t'.join([self.word_str,self.pc_string,self.start_str,self.fid])

	def __str__(self):
		m = 'pre context:\t\t' + self.pc_string
		m += 't zero:\t\t' + self.start_str
		m += 'file id:\t\t' + self.fid
		m += 'word:\t\t' + self.word
		return m
		
	def format(self,index, offset = 0.5):
		'''Create a line for the KALDI chunk file.'''
		return ' '.join([self.fid +'.'+str(index),self.fid,str(self.start),str(self.start + offset)])
					
					
		

