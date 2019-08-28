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
		self.set_word_index()


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
		self.ppls_register, contains the lm with cow interpolated with correct register lm
		self.ppls_other1/2, contains the lm with cow interpolated with wrong register
			the self.lm_other1/2 first part is the register lm used, other is the text used (corresponding to exp)
		'''
		exp = self.exp
		if exp == 'o': 
			self.lm = 'books_o_70'#'books_30'
			self.lm_other1 = 'news_o_70'#'autocues_30_other-o'
			self.lm_other2 = 'dialogues_o_87'#'compa_20_other-o'
			self.lm_cache = 'cache-general_o_64_5.out'
		if exp == 'k': 
			self.lm = 'news_k_70'#'autocues_30'
			self.lm_other1 = 'books_k_70'#'books_30_other-k'
			self.lm_other2 = 'dialogues_k_87'#'compa_20_other-k'
			self.lm_cache = 'cache-general_k_64_5.out'
		if exp == 'ifadv': 
			self.lm = 'dialogues_ifadv_87'#'compa_20'
			self.lm_other1 = 'news_ifadv_70'#'autocues_30_other-ifadv'
			self.lm_other2 = 'books_ifadv_70'#'books_30_other-ifadv'
			self.lm_cache = 'cache-general_ifadv_64_5.out'
		self.infos = [line.split('\t') for line in open(path.data +'sentence_info_' + exp,encoding = 'utf-8').read().split('\n')]
		self.strings = open(path.data+'sentence_strings_' + exp,encoding = 'utf8').read().split('\n')
		# should make ppl filename more genral
		self.ppls = open(path.data+'out_' + exp + '_10.ppl',encoding = 'utf-8').read().split('\n\n')
		self.ppls_register = open(path.data + 'specific_'+self.lm + '.out', encoding = 'utf-8').read().split('\n\n')
		self.ppls_other1= open(path.data + 'specific_'+self.lm_other1 + '.out', encoding = 'utf-8').read().split('\n\n')
		self.ppls_other2= open(path.data + 'specific_'+self.lm_other2 + '.out', encoding = 'utf-8').read().split('\n\n')
		self.ppls_cache= open(path.data + self.lm_cache, encoding = 'utf-8').read().split('\n\n')

	def set_ppl_sentences(self):
		'''Create ppl_sentence objects.'''
		self.sentences = []
		for i,ppl_line in enumerate(self.ppls):
			if i == len(self.ppls) -1: 
				setattr(self,self.exp + '_ppl', ppl_line)
				break
			info_line = self.infos[i]
			info_line_register = self.ppls_register[i]
			info_line_other1= self.ppls_other1[i]
			info_line_other2= self.ppls_other2[i]
			info_line_cache= self.ppls_cache[i]
			s = self.strings[i]
			self.sentences.append(ppl_sentence(ppl_line,info_line,s,self.exp,info_line_register,
			info_line_other1,info_line_other2,info_line_cache))
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
		
	def _make_blocks(self):
		self.blocks = []
		for i,s in enumerate(self.sentences):
			if i == 0: 
				bid = s.bid
				temp = [s]
			elif bid != s.bid:
				self.blocks.append(temp)
				temp = [s]
				bid = s.bid
			else: temp.append(s)
			if i == len(self.sentences) -1: self.blocks.append(temp) 

	def set_word_index(self):
		self._make_blocks()
		self.blocks_words = []
		for b in self.blocks:
			i = 0
			temp = []
			for s in b:
				for w in s.words:
					w.word_index = i
					i += 1
					temp.append(w)
			self.blocks_words.append(temp)

	def _match_block_word(self,b):
		if not hasattr(self,'blocks_words'): self.set_word_index()
		self.matched_block_word = False
		for bi,bppl in enumerate(self.blocks_words):
			mismatch = False
			for i,wppl in enumerate(bppl):
				if b.words[i].word_utf8_nocode_nodia().lower() != wppl.word: 
					mismatch = True
					# print(b.words[i].word_utf8_nocode_nodia(),wppl.word)
					break
			if not mismatch: 
				self.matched_block_word = bppl
				self.mbwi= bi
				break
		if not self.matched_block_word:
			print('could not match block by word')
			self.mbwi = False

	def _match_block_sentence(self,b):
		self.matched_block_sentence = False
		for bi,bs in enumerate(self.blocks):
			mismatch = False
			for i, s in enumerate(bs):
				if b.sentences[i].string_utf8_words_no_diacritics(True).lower() != s.sentence:
					mismatch = True
					# print(i,b.sentences[i],s.sentence)
					break
			if not mismatch:
				self.matched_block_sentence = bs
				self.mbsi = bi
				break
		if not self.matched_block_sentence:
			print('could not match block by sentence',b)
			self.mbws = False

	def add_ppl_to_words(self,b):
		self._match_block_word(b)
		self._match_block_sentence(b)
		if not self.matched_block_word and not self.matched_block_sentence:
			print('cannot add ppl to word without matched block.')
			return
		if self.matched_block_word and self.matched_block_sentence and self.mbwi != self.mbsi:
			print('found a different block  with sentence match and word match')
			return
		if self.matched_block_word:
			for i,wppl in enumerate(self.matched_block_word):
				if wppl.word != b.words[i].word_utf8_nocode_nodia().lower():
					raise ValueError('words do not match:',wppl.word,b.words[i],b)
				b.words[i].ppl = wppl
		elif self.matched_block_sentence:
			print('only sentence match between blocks and ppl')
			for i, sppl in enumerate(self.matched_block_sentence):
				if sppl.sentence != b.sentences[i].string_utf8_words_no_diacritics(True).lower():
					raise ValueError('sentences do not match:',sppl,b.sentences[i])
				for wi,word in enumerate(sppl.words):
					b.sentences[i].words[wi].ppl = word
					
		
				
		


class ppl_sentence:
	'''Holds information about each sentence, based on the SRILM ngram call output.'''
	def __init__(self,ppl_line,info_line,s,exp,info_line_register = '',
				info_line_other1 = '',info_line_other2 = '',info_line_cache=''):
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
		self.info_line_register = info_line_register
		self.ppl_register_list = info_line_register.split('\n')
		self.info_line_other1= info_line_other1
		self.ppl_other1_list = info_line_other1.split('\n')
		self.info_line_other2= info_line_other2
		self.ppl_other2_list = info_line_other2.split('\n')
		self.info_line_cache= info_line_cache
		self.ppl_cache_list = info_line_cache.split('\n')
		self.sentence_number = self.info[1]
		self.sid = self.info[2]
		self.bid = self.info[3]
		self._extract_sentence_ppl()
		self._extract_noovs()
		self.set_ppl_words()


	def __repr__(self):
		m =  str(self.sentence_number) + '\t'+ self.sid +'\t'+self.sentence+'\t'+ str(self.nwords)
		m += '\t' + self.exp 
		return m

	def __str__(self):
		m = 'sentence:\t'+self.sentence+ '\n'
		m += 's number:\t'+self.sentence_number+ '\n'
		m += 'id:\t\t' + self.sid+ '\n'
		m += 'logprob:\t\t' + str(self.logprob)+ '\n'
		m += 'ppl:\t\t' + str(self.ppl_score)+ '\n'
		m += 'ppl1:\t\t' + str(self.ppl1_score)+ '\n'
		m += 'experiment:\t'+self.exp+ '\n'
		return m

	def set_ppl_words(self):
		'''Create ppl_word object for each word in the sentence.'''
		self.words = []
		ppl_register_list = self.ppl_register_list[1:-2]
		ppl_other1_list = self.ppl_other1_list[1:-2]
		ppl_other2_list = self.ppl_other2_list[1:-2]
		ppl_cache_list = self.ppl_cache_list[1:-2]
		for i,word_line in enumerate(self.ppl_list[1:-2]):
			# print(word_line)
			word = word_line.split('( ')[-1].split(' | ')[0]
			word_line_register = ppl_register_list[i]
			word_line_other1= ppl_other1_list[i]
			word_line_other2= ppl_other2_list[i]
			word_line_cache= ppl_cache_list[i]
			if word == '</s>': continue
			self.words.append(ppl_word(i,word_line,self,word_line_register,word_line_other1,word_line_other2,word_line_cache))

	def _extract_sentence_ppl(self):
		'''Retreive some sentence ppl stats.'''
		self.logprob = self.ppl_line.split(' logprob= ')[-1].split(' ')[0]
		self.ppl_score, self.ppl1_score = self.ppl_line.split('ppl= ')[-1].split(' ppl1= ')

		self.logprob_register = self.info_line_register.split(' logprob= ')[-1].split(' ')[0]
		self.ppl_score_register, self.ppl1_score_register = self.info_line_register.split('ppl= ')[-1].split(' ppl1= ')

		self.logprob_other1= self.info_line_other1.split(' logprob= ')[-1].split(' ')[0]
		self.ppl_score_other1, self.ppl1_score_other1= self.info_line_other1.split('ppl= ')[-1].split(' ppl1= ')

		self.logprob_other2= self.info_line_other2.split(' logprob= ')[-1].split(' ')[0]
		self.ppl_score_other2, self.ppl1_score_other2= self.info_line_other2.split('ppl= ')[-1].split(' ppl1= ')

		self.logprob_cache= self.info_line_cache.split(' logprob= ')[-1].split(' ')[0]
		self.ppl_score_cache, self.ppl1_score_cache= self.info_line_cache.split('ppl= ')[-1].split(' ppl1= ')

		try: self.ppl_score = float(self.ppl_score)
		except: pass
		try: self.ppl1_score = float(self.ppl1_score)
		except: pass
		try: self.logprob = float(self.logprob)
		except: pass

		try: self.ppl_score_register = float(self.ppl_score_register)
		except: pass
		try: self.ppl1_score_register = float(self.ppl1_score_register)
		except: pass
		try: self.logprob_register = float(self.logprob_register)
		except: pass

		try: self.ppl_score_other1= float(self.ppl_score_other1)
		except: pass
		try: self.ppl1_score_other1= float(self.ppl1_score_other1)
		except: pass
		try: self.logprob_other1 = float(self.logprob_other1)
		except: pass

		try: self.ppl_score_other2= float(self.ppl_score_other2)
		except: pass
		try: self.ppl1_score_other2= float(self.ppl1_score_other2)
		except: pass
		try: self.logprob_other2 = float(self.logprob_other2)
		except: pass

		try: self.ppl_score_cache= float(self.ppl_score_cache)
		except: pass
		try: self.ppl1_score_cache= float(self.ppl1_score_cache)
		except: pass
		try: self.logprob_cache= float(self.logprob_cache)
		except: pass

	def _extract_noovs(self):
		'''Count out of vocabulary words.'''
		self.oov = self.ppl_line.split(' words, ')[-1].split(' ')[0]
		self.nwords= self.ppl_line.split(' sentences, ')[-1].split(' ')[0]

		self.oov_register = self.info_line_register.split(' words, ')[-1].split(' ')[0]
		self.nwords_register = self.info_line_register.split(' sentences, ')[-1].split(' ')[0]

		self.oov_other1= self.info_line_other1.split(' words, ')[-1].split(' ')[0]
		self.nwords_other1 = self.info_line_other1.split(' sentences, ')[-1].split(' ')[0]

		self.oov_other2= self.info_line_other2.split(' words, ')[-1].split(' ')[0]
		self.nwords_other2 = self.info_line_other2.split(' sentences, ')[-1].split(' ')[0]

		self.oov_cache= self.info_line_cache.split(' words, ')[-1].split(' ')[0]
		self.nwords_cache= self.info_line_cache.split(' sentences, ')[-1].split(' ')[0]

		try: self.oov = int(self.oov)
		except: pass
		try: self.nwords= int(self.nwords)
		except: pass

		try: self.oov_register = int(self.oov_register)
		except: pass
		try: self.nwords_register= int(self.nwords_register)
		except: pass

		try: self.oov_other1= int(self.oov_other1)
		except: pass
		try: self.nwords_other1= int(self.nwords_other1)
		except: pass

		try: self.oov_other2= int(self.oov_other2)
		except: pass
		try: self.nwords_other2= int(self.nwords_other2)
		except: pass

		try: self.oov_cache= int(self.oov_cache)
		except: pass
		try: self.nwords_cache= int(self.nwords_cache)
		except: pass


class ppl_word:
	'''PPL info for a word based on SRILM ngram call output.'''
	def __init__(self,i,word_line,sentence,word_line_register,word_line_other1,word_line_other2,word_line_cache):
		'''PPL info for a word based on SRILM ngram call output.
		i 		index, word number in sentence
		word... the line corresponding to the word in the SRILM output
		sent... the string representation of the sentence
		'''
		self.ppl_word = word_line.split('( ')[-1].split(' | ')[0]
		self.ppl_word_register = word_line_register.split('( ')[-1].split(' | ')[0]
		if self.ppl_word == '<unk>': self.unk = True
		else: self.unk = False
		self.index = i
		self.word_line = word_line
		self.word_line_register = word_line_register
		self.word_line_other1= word_line_other1
		self.word_line_other2= word_line_other2
		self.word_line_cache= word_line_cache
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
		m += 'logprob_register:\t\t' + str(self.logprob_register)+ '\n'
		m += 'logprob_other1:\t\t' + str(self.logprob_other1)+ '\n'
		m += 'logprob_other2:\t\t' + str(self.logprob_other2)+ '\n'
		m += 'logprob_cache:\t\t' + str(self.logprob_cache)+ '\n'
		m += 'p:\t\t' + str(self.p)+ '\n'
		m += 'ngram:\t\t' + str(self.ngram)+ '\n'
		m += 'experiment:\t'+self.exp+ '\n'
		m += 'sentence:\t'+self.sentence+ '\n'
		m += 'history:\t'+' '.join(self.history) + '\n'
		return m

	def _extract_metrics(self):
		'''Retreive ppl information.'''
		if 'gram' in self.word_line:
			self.ngram = self.word_line.split('= [')[-1].split('gram')[0]
			self.oov = False
		else:
			self.ngram = 0
			self.oov = True
		if 'cache' in self.word_line_cache:
			self.cache = float(self.word_line_cache.split('cache=')[-1].split(']')[0])
		else: self.cache = -1
		self.p = self.word_line.split('] ')[-1].split(' ')[0]
		self.logprob = self.word_line.split(' [ ')[-1].split(' ]')[0]
		self.ngram = int(self.ngram)

		self.p_register = self.word_line_register.split('] ')[-1].split(' ')[0]
		self.logprob_register = self.word_line_register.split(' [ ')[-1].split(' ]')[0]

		self.p_other1= self.word_line_other1.split('] ')[-1].split(' ')[0]
		self.logprob_other1= self.word_line_other1.split(' [ ')[-1].split(' ]')[0]

		self.p_other2= self.word_line_other2.split('] ')[-1].split(' ')[0]
		self.logprob_other2= self.word_line_other2.split(' [ ')[-1].split(' ]')[0]

		self.p_cache= self.word_line_cache.split('] ')[-1].split(' ')[0]
		self.logprob_cache= self.word_line_cache.split(' [ ')[-1].split(' ]')[0]

		self.p= float(self.p)
		self.logprob= float(self.logprob)
		
		self.p_register= float(self.p_register)
		self.logprob_register= float(self.logprob_register)

		self.p_other1= float(self.p_other1)
		self.logprob_other1= float(self.logprob_other1)

		self.p_other2= float(self.p_other2)
		self.logprob_other2= float(self.logprob_other2)

		self.p_cache= float(self.p_cache)
		self.logprob_cache= float(self.logprob_cache)
