import experiment
import numpy as np

'''create lists that describe the word onsets, gates and filenames for each word in the experiment'''

class pmn_data():
	'''Holds all words of all experiments with relevant information for PMN experiment.
	'''
	def __init__(self,p = None,  history_length = 4):
		'''Hold all words of all experiments with PMN information (phonological mismatch negativity)
		p 		participant object with all sessions loaded p = e.Participant(1) p.add_all_sessions()
		his... 	number of previous words added to the pmn word (to determine entropy) 
		'''
		self.p = p
		self.history_length = history_length
		if not p: 
			self.p = experiment.Participant(1)
			self.p.add_all_sessions()
		self.make_data()
		self.add_index()

	def make_data(self):
		'''Create pmn_word object for each word.'''
		self.pmn_words = []
		for session in self.p.sessions:
			for block in session.blocks:
				for i,word in enumerate(block.words):
					start = i - self.history_length
					if start < 0 : start = 0
					precontext = block.words[start:i]
					precontext = [w.word_utf8_nocode_nodia() for w in precontext]
					self.pmn_words.append(pmn_word(word,precontext))

	def add_index(self):
		for i,w in enumerate(self.pmn_words):
			w.index = make_number(i+1,len(self.pmn_words))
		

	def make_chunk_speaker_file(self,offset = 0.5, small = False, exclude_words_shorter_than_offset = False,wlm = 2):
		'''Create lists with information for KALDI.
		offset 		n seconds after word onset
		small 		whether to create a limited set of examples (n = 100) for testing
		exclude...  default includes all words, if true exclude words that are to short
		wlm 		how many time should offset be longer to exclude word (if offset_dep... is true)
		output:
		self.speaker 		speaker followed by corresponding filenames (speakers).
		self.chunks 			chunk name, filename, start, end
		'''
		self.speaker2f = {}
		self.chunks = []
		self.speakers = []
		self.small = small
		self.offset = offset
		for i,w in enumerate(self.pmn_words):
			if exclude_words_shorter_than_offset and offset > w.word.duration * wlm: continue
			self.chunks.append(w.format(make_number(i+1,len(self.pmn_words)),offset=offset))
			if w.word.sid not in self.speaker2f.keys():
				self.speaker2f[w.word.sid] = [self.chunks[-1].split(' ')[0]]
			else: self.speaker2f[w.word.sid].append(self.chunks[-1].split(' ')[0])
			if small and i+1 == 100:
				break
		for s in self.speaker2f.keys():
			self.speakers.append(' '.join([s+'-S0'] + self.speaker2f[s]))
			

	def save(self):
		'''Save KALDI helper files created with make_chunk_speaker_file().'''
		fc = 'chunks'
		if self.small: fc = 'chunks_small'
		fc += '_' + str(int(1000 * self.offset)) +'.txt'
		fs = 'speakers' + '_' + str(int(1000 * self.offset)) +'.txt'
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
		self.word_str = word.word_utf8_nocode_nodia()
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
		m += 'word:\t\t' + self.word_str
		return m
		
	def format(self,index, offset = 0.5):
		'''Create a line for the KALDI chunk file.'''
		return ' '.join([self.fid +'.'+str(index),self.fid,str(self.start),str(self.start + offset)])
					
					
		
def make_number(i, ntokens):
	ndigits = len(str(ntokens)) + 1
	diff = ndigits - len(str(i))
	return '0'*diff + str(i)

def make_offset(pmn= None,offset = 0.11, exclude_words_shorter_than_offset = False, wlm = 2):
	if pmn == None: pmn = pmn_data()
	t = exclude_words_shorter_than_offset
	pmn.make_chunk_speaker_file(offset = offset, exclude_words_shorter_than_offset = t, wlm = wlm)
	pmn.save()
	return pmn

def make_all_offset(pmn = None, offset_start = 0.11, offset_step = 0.02, offset_end = 0.66, ewsto = True, wlm = 2):
	if pmn == None: pmn = pmn_data()
	offsets = np.arange(offset_start,offset_end,offset_step)
	print('making files for',len(offsets),'offsets')
	for offset in offsets:
		t = ewsto if offset > 0.11 else False
		make_offset(pmn=pmn,offset=offset,exclude_words_shorter_than_offset = t,wlm = wlm)
	return pmn
		
		
	
