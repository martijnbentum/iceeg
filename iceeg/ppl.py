import path

class word2ppl:
	'''Converts SRILM ngram output (debug 2) to an object with probablistic information.'''
	def __init__(self,exp = 'o'):
		'''Converts SRILM ngram output (debug 2) to an object with probablistic information.
		exp 	sets the experiment expects the files sentence_info_[exp] sentence_strings_[exp]
				and ppl_[exp] to exists in the data directory: EEG_DATA_ifadv_cgn/
		'''
		self.exp = exp 
		self.read_files()
		self.words = []
		self.set_ppl_sentences()
		self._extract_info()


	def __repr__(self):
		return self.exp + '\tnsentence: '+ str(self.nsentences)+'\t\tnword: '+str(self.nwords)+'\toov:'+str(self.oov)

	def __str__(self):
		m = 'experiment:\t'+self.exp+ '\n'
		m += 'nsentences:\t'+str(self.nsentences)+ '\n'
		m += 'nwords:\t'+str(self.nwords)+ '\n'
		m += 'oov:\t'+str(self.oov)+ '\n'
		m += 'logprob:\t'+str(self.logprob)+ '\n'
		m += 'ppl_score:\t'+str(self.ppl_score)+ '\n'
		m += 'ppl1_score:\t'+str(self.ppl1_score)+ '\n'
	

	def read_files(self):
		'''OUTPUT:
		self.infos, holds sentence info, sentence_info file made with p.print_session (part of participant in experiment.py)
		self.strings, holds the string representation of each sentence
		self.ppls, holds the output of the SRILM ngram call
		'''
		exp = self.exp
		self.infos = [line.split('\t') for line in open(path.data +'sentence_info_' + exp,encoding = 'utf-8').read().split('\n')]
		self.strings = open(path.data+'sentence_strings_' + exp,encoding = 'utf8').read().split('\n')
		# should make ppl filename more genral
		self.ppls = open(path.data+'ppl_' + exp,encoding = 'utf-8').read().split('\n\n')


	def set_ppl_sentences(self):
		'''Create ppl_sentence objects.'''
		self.sentences = []
		for i,ppl_line in enumerate(self.ppls):
			if i == len(self.ppls) -1: 
				setattr(self,self.exp + '_ppl', ppl_line)
				break
			info_line = self.infos[i]
			s = self.strings[i]
			self.sentences.append(ppl_sentence(ppl_line,info_line,s,self.exp))
			self.words.extend(self.sentences[-1].words)

	def _extract_info(self):
		'''Retreive some aggregate stats about the sentences.'''
		self.nsentences = len(self.sentences)
		self.nwords = len(self.words)
		self.oov = int(self.ppls[-1].split('words, ')[-1].split(' OOVs')[0])
		self.zeroprobs = float(self.ppls[-1].split('\n')[1].split(' ')[0])
		self.logprob = float(self.ppls[-1].split('logprob= ')[-1].split(' ')[0])
		self.ppl_score= float(self.ppls[-1].split('ppl= ')[1].split(' ')[0])
		self.ppl1_score= float(self.ppls[-1].split('ppl1= ')[-1].strip())
		


class ppl_sentence:
	'''Holds information about each sentence, based on the SRILM ngram call output.'''
	def __init__(self,ppl_line,info_line,s,exp):
		'''PPL information for each sentence, hold ppl_word objects.
		ppl_line 	a string with all info related to 1 sentence (SRILM ngram output)
		info_line 	list with sentence info (checks ppl info)
		s 			sentence (string format)
		exp 		the session, o k or ifadv
		'''
		self.ppl_line = ppl_line
		self.ppl_list = ppl_line.split('\n')
		self.info = info_line
		assert self.info[-1] == self.ppl_list[0] == s
		self.sentence = s
		self.exp = exp
		self.sentence_number = self.info[1]
		self.sid = self.info[2]
		self._extract_sentence_ppl()
		self._extract_noovs()
		self.set_ppl_words()


	def __repr__(self):
		m =  str(self.sentence_number) + '\t'+ self.sid +'\t'+self.sentence+'\t'+ str(self.nwords)
		m += '\t' + self.exp 
		return m

	def __str__(self):
		m += 'sentence:\t'+self.sentence+ '\n'
		m += 's number:\t'+self.sentence_number+ '\n'
		m += 'id:\t\t' + self.sid+ '\n'
		m += 'logprob:\t\t' + str(self.logprob)+ '\n'
		m += 'ppl:\t\t' + str(self.ppl_score)+ '\n'
		m += 'ppl1:\t\t' + str(self.ppl1_score)+ '\n'
		m += 'experiment:\t'+self.exp+ '\n'

	def set_ppl_words(self):
		'''Create ppl_word object for each word in the sentence.'''
		self.words = []
		for i,word_line in enumerate(self.ppl_list[1:-2]):
			print(word_line)
			word = word_line.split('( ')[-1].split(' | ')[0]
			if word == '</s>': continue
			self.words.append(ppl_word(i,word_line,self))

	def _extract_sentence_ppl(self):
		'''Retreive some sentence ppl stats.'''
		self.logprob = self.ppl_line.split(' logprob= ')[-1].split(' ')[0]
		self.ppl_score, self.ppl1_score = self.ppl_line.split('ppl= ')[-1].split(' ppl1= ')
		try: self.ppl_score = float(self.ppl_score)
		except: pass
		try: self.ppl1_score = float(self.ppl1_score)
		except: pass
		try: self.logprob = float(self.logprob)
		except: pass

	def _extract_noovs(self):
		'''Count out of vocabulary words.'''
		self.oov = self.ppl_line.split(' words, ')[-1].split(' ')[0]
		self.nwords= self.ppl_line.split(' sentences, ')[-1].split(' ')[0]
		try: self.oov = int(self.oov)
		except: pass
		try: self.nwords= int(self.nwords)
		except: pass


class ppl_word:
	'''PPL info for a word based on SRILM ngram call output.'''
	def __init__(self,i,word_line,sentence):
		'''PPL info for a word based on SRILM ngram call output.
		i 		index, word number in sentence
		word... the line corresponding to the word in the SRILM output
		sent... the string representation of the sentence
		'''
		self.ppl_word = word_line.split('( ')[-1].split(' | ')[0]
		if self.ppl_word == '<unk>': self.unk = True
		else: self.unk = False
		self.index = i
		self.word_line = word_line
		self.ppl_sentence = sentence
		self.sentence = self.ppl_sentence.sentence
		self.word = self.sentence.split(' ')[i]
		self.word_id= '_'.join([sentence.sentence_number,sentence.sid,str(self.index),self.word])
		self.exp = sentence.exp
		self.history = self.word_line.split(' | ')[-1].split(') ')[0].strip(' ').split(' ')
		self._extract_metrics()


	def __repr__(self):
		m = self.word + '\t' + str(self.logprob) + '\t'+ str(self.ngram) 
		m += '\t' + self.exp + '\t'+' '.join(self.history) + '\t\t' + str(self.unk)
		return m

	def __str__(self):
		m = 'word:\t\t' + self.word + '\n'
		m += 'id:\t\t' + self.word_id + '\n'
		m += 'logprob:\t\t' + str(self.logprob)+ '\n'
		m += 'p:\t\t' + str(self.p)+ '\n'
		m += 'ngram:\t\t' + str(self.ngram)+ '\n'
		m += 'experiment:\t'+self.exp+ '\n'
		m += 'sentence:\t'+self.sentence+ '\n'
		m += 'history:\t'+' '.join(self.history) + '\n'

	def _extract_metrics(self):
		'''Retreive ppl information.'''
		if 'gram' in self.word_line:
			self.ngram = self.word_line.split('= [')[-1].split('gram')[0]
			self.oov = False
		else:
			self.ngram = 0
			self.oov = True
		self.p = self.word_line.split('] ')[-1].split(' ')[0]
		self.logprob = self.word_line.split(' [ ')[-1].split(' ]')[0]
		self.ngram = int(self.ngram)
		self.p= float(self.p)
		self.logprob= float(self.logprob)
		

