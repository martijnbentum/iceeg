#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
import codecs
import gensim
import glob
import string

class Word:
	'''Structure with infomation  about word, POS, start/end times, corpus and register.

	whether it is the last word of a sentence, whether it has special marking (*)
	should add an exclude category: special marker, word stop list, etc?
	surprisal and frequency should be added
	'''

	def __init__(self, word,word_number, chunk_number, chunk_st,chunk_et,filename = None, fid = None,sid = None,cid = None,wid = None, st = None, et = None,corpus = None,register = None, chunk=None):
		'''Structure with infomation  about word, POS, start/end times, corpus and register.'''
		self.usable = True
		self.word = word
		self.word_number = word_number
		self.chunk_number= chunk_number
		self.chunk_st = chunk_st
		self.chunk_et = chunk_et
		self.status()
		self.filename = filename
		self.fid = fid
		self.sid = sid
		self.cid = cid
		self.st = st
		self.et = et
		if st and et: self.duration = round(et - st,3)
		else: self.duration = None
		self.add_info(filename,fid,sid,cid,wid,corpus = corpus,register = register)
		self.add_times()
		self.prev_word = None
		self.next_word = None 
		self.set_overlap()
		self.replace_diacritics()
		self.remove_special_code()
		self.pos_ok = False
		if self.corpus == 'CGN': self.set_overlap(False)
		self.chunk = chunk

	def __str__(self):
		a = ['word:\t\t'+self.word]
		a.append( 'word index:\t'+str(self.word_number) )
		a.append( 'start time:\t'+str(self.st) )
		a.append( 'end time:\t'+str(self.et) )
		a.append( 'duration:\t'+str(self.duration) )
		a.append( 'chunk index:\t'+str(self.chunk_number) )
		a.append( 'chunk st:\t'+str(self.chunk_st) )
		a.append( 'chunk et:\t'+str(self.chunk_et) )
		a.append( 'special:\t'+str(self.special_code))
		a.append( 'punctuation:\t'+str(self.punctuation))
		a.append( 'end of line:\t'+str(self.eol))
		a.append( 'file id:\t'+self.fid) 
		a.append( 'speaker id:\t'+self.sid) 
		a.append( 'overlap:\t'+str(self.overlap)) 
		a.append( 'overlap unk:\t'+str(self.overlap_unknown)) 
		a.append('corpus:\t\t'+str(self.corpus))
		a.append('register:\t'+str(self.register))
		a.append('usable:\t\t'+str(self.usable) + ' (non overlapping, time info present, no special marking, sample number > 0.)')
		if self.pos_ok:
			a.append('-'*30)
			a.append('POS tag INFO')
			a.append(self.pos.__str__())
		a.append('-'*30)
		a.extend(self.print_sample_info(aslist = True))
		return '\n'.join(a)
		
	def __repr__(self):
		if hasattr(self,'pos'):
			return self.word.ljust(25) + ' pos: ' + self.pos.pos_simple + '\tdur: ' + str(self.duration) + '\tregister: ' + self.register + '\tok: '+ str(self.usable)
		else:
			return self.word.ljust(25) + ' pos: ' + 'NA'+ '\tdur: ' + str(self.duration) + '\tregister: ' + self.register + '\tok: '+ str(self.usable)

	def __eq__(self,other):
		return self.st_sample == other.st_sample and self.et_sample == other.et_sample and self.word == other.word

	def status(self):
		'''Set word status 
		whether it has a special code, punctuation, or is the last word in a sentence
		(should add usable field based on a stoplist)
		'''
		self.special_code,self.punctuation,self.eol = False,False,False
		if '*' in self.word:
			self.special_code = True
			self.usable = False
		for p in string.punctuation.replace('*',''):
			if p in self.word:
				self.punctuation = True
		for eol in ['.','!','?']:
			if eol in self.word:
				self.eol = True

	def add_info(self,filename = None, fid = None,sid = None,cid = None,wid = None,st = None, et = None,corpus = None,register = None):
		'''Add information to the word, file speaker chunk and word id/index, 
		start and end time, corpus and register.
		'''
		if filename: self.filename = filename
		if fid != None: self.fid = fid
		if sid != None: self.sid = sid
		if cid != None: self.cid = cid
		if wid != None: self.wid = wid
		if st != None: self.st = st
		if et != None: self.et = et
		if corpus != None: self.corpus = corpus
		if register != None: self.register = register

	def add_prev_word(self,word = None):
		# unused, add last word
		self.prev_word = word

	def add_next_word(self,word = None):
		# unused, add next word
		self.next_word = word

	def set_overlap(self,overlap = None):
		'''Mark whether word overlaps in time with other words, and mark whether this has been checked.'''
		if overlap == None:
			self.overlap_unknown = True
			self.overlap = False
		else:
			self.overlap = overlap 
			self.overlap_unknown = False
			if self.overlap: self.usable = True

	def add_times(self,st = None, et= None):
		'''Adds start and end time.'''
		if st != None: self.st = st
		if et != None: self.et = et
		if self.st != None and self.et != None:
			self.duration = round(self.et - self.st,3)

	
	def replace_diacritics(self):
		'''Replace special codes in IFADV transcription with charachter with diacritics. 
		(Replaces the IFADV special diacritic codes with the utf8 equivalent.)
		'''
		rd = {'\e^':'ê','\e"':'ë',"\e'":'é',"\e`":'è','\i"':'ï',"\i'":'í','\\u"':'ü',"\\u'":'ú',"\\a'":'á',"\\a`":'à','\o"':'ö',"\o'":'ó','\c,':'ç'}
		temp = self.word
		for c in rd.keys():
			temp = temp.replace(c,rd[c])
		self.word_utf8 = temp


	def remove_special_code(self):
		'''Remove * and eol characters.'''
		if self.special_code:
			self.word_nocode = self.word.split('*')[0]
			self.word_utf8_nocode = self.word_utf8.split('*')[0]
		else:
			self.word_nocode = self.word
			self.word_utf8_nocode = self.word_utf8

		for eol in ['.','!','?']:
			self.word_nocode = self.word_nocode.replace(eol,'')
			self.word_utf8_nocode = self.word_utf8_nocode.replace(eol,'')
		
	def word_utf8_nocode_nodia(self,fix_apostrophe = True):
		if fix_apostrophe:
			return apply_fix_apostrophe(deaccent_text(self.word_utf8_nocode))
		else: deaccent_text(self.word_utf8_nocode)



	def make_pos_info(self):
		'''Add pos object info to the word __str__ function.'''
		if not self.pos_ok:
			print('no pos or problem with pos',self.fid,self.sid,self.chunk_number)
			return 0
		output = []
		p = self.pos
		output = [p.token,p.pos,p.lemma,p.probability_of_tag]
		output.extend([self.st,self.et,self.word,self.word_number+1,self.chunk_number+1])
		output = map(str,output)
		return '\t'.join(output)


	def seconds2sample(self,s):
		'''Return s*1000 if it is a int or float.'''
		# if type(s) == float or type(s) == int:
		return int(round(s*1000))
		# else:
			# print(s,'is not a number, returning as is')
			# return s

	def set_times_as_samples(self,offset = 0,baseline = -300,epoch_dur = 1000):
		'''Set start/end sample number of a word based on offset value
		
		Keywords:
		offset = offset in samples from onset marker sample number, default = 0  int
		baseline = length of baseline window - determines start of epoch, default = -300   int
		epoch_dur = length from onset word to end of epoch, default = 1000   int
		'''
		try:
			self.sample_offset = offset
			self.st_sample = self.seconds2sample(self.st) + offset
			self.et_sample = self.seconds2sample(self.et) + offset
			self.duration_sample = self.seconds2sample(self.duration)
			self.st_epoch = self.st_sample + baseline
			self.et_epoch = self.st_sample + epoch_dur
			self.baseline = baseline
			self.epoch_dur = epoch_dur
			if self.st_sample < 0: self.usable = False
		except:
			print(self.__str__())
			self.usable = False

	def print_sample_info(self,aslist = False):
		'''Print sample information of word if it is available.'''
		try:
			a = ['sample offset:\t\t'+str(self.sample_offset)]
			a.append( 'baseline:\t\t'+str(self.baseline) )
			a.append( 'epoch duration:\t\t'+str(self.epoch_dur) )
			a.append( 'start_sample:\t\t'+str(self.st_sample) )
			a.append( 'end sample:\t\t'+str(self.et_sample) )
			a.append( 'duration_sample:\t'+str(self.duration_sample) )
			a.append( 'start epoch:\t\t'+str(self.st_epoch) )
			a.append( 'end epoch:\t\t'+str(self.et_epoch) )
			if aslist: return a
			else: return '\n'.join(a)
		except:
			m = 'NA, use set_times_as_samples([offset],[baseline],[epoch_dur])'
			if aslist: return [m]
			else: return m
	
	def set_artifact(self,artifact):
		self.artifact_id = artifact.epoch_id
		self.usable = False

	def add_phoneme_word(self):
		self.phoneme_word = phoneme_word(self)
		

def deaccent_text(text):
    t = ''
    ac = 'áàèéêëúüçćîïíīìôöòóō'
    deaccented_chars = 0
    for i,ch in enumerate(text):
        if ch in ac:
            t += gensim.utils.deaccent(ch)
            deaccented_chars += 1
        else: t += ch
    return t


def make_apostrophe_dict():
	source = "z'n,d'rvan,d'r,'r,'tzelfde,'t,'ns,m'n,zo'n,'k,'m,'s,'rzelf,d'ruit,'n,da's,d'raan".split(',')
	goal = "zijn,daarvan,der,er,hetzelfde,het,eens,mijn,zoeen,ik,hem,is,haarzelf,daaruit,een,datis,daaraan".split(',')
	return dict(zip(source,goal))


def apply_fix_apostrophe(w):
	ad = make_apostrophe_dict()
	for k in ad.keys():
		if w == k: return ad[k]
	if w.endswith("'s"):return w.replace("'s",'s')
	return w



class phoneme_word():
	'''extracts phonemic transcription (generated by forced alignment) of a word
	contains a list of phoneme objects for the individual phonemes in .phonemes
	Assumes that the word object w has a chunk object (annot_chunk).
	'''
	def __init__(self,w):
		self.w = w
		self.phon_word_number = w.word_number
		self.phon_word_line = w.chunk.awd_phon_words[w.word_number]
		self.phon_word = self.phon_word_line[2]
		self.st, self.et = float(self.phon_word_line[0]), float(self.phon_word_line[-1])
		self.duration = round(self.et - self.st,4)
		self.add_phonemes()


	def __repr__(self):
		m='phon word\t'+self.phon_word+'\tdur: '+str(self.duration)+'\tnphon: '+str(self.nphonemes)
		return m

	def add_phonemes(self):
		'''Add a phoneme object for each phoneme in the word.
		phonemic transcription is forced aligned, so boundaries and phonemes are not necessarilly
		present in the auditory signal.
		'''
		self.phonemes = []
		for i,p in enumerate(self.w.chunk.awd_phon):
			if p[0] >= self.phon_word_line[0] and p[-1] <= self.phon_word_line[-1]:
				self.phonemes.append(phoneme(p,self,i))
		self.nphonemes = len(self.phonemes)

		

class phoneme():
	'''Phoneme in a word, contains start and end times, assumes sample info is set for the word.
	this means the phoneme object can only be created when the word object is part of a participant
	in the eeg experiment, since sample info is part of the eeg data.'''
	def __init__(self,p,pw,phoneme_number):
		self.phoneme_line = p
		self.pw = pw
		self.phoneme_number = phoneme_number
		self.phoneme = p[2]
		self.st, self.et = float(p[0]), float(p[-1])
		self.duration = round(self.et - self.st,4)
		self.usable = self.pw.w.usable
		self.set_times_as_samples(self.pw.w.sample_offset)


	def __repr__(self):
		m = 'phoneme\t' + self.phoneme + '\tdur: '+str(self.duration)+'\tword: '+self.pw.phon_word
		return m


	def seconds2sample(self,s):
		'''sample is the sample number in the eeg data.
		the sample frequency is 1000, therefore time is seconds * 1000.
		'''
		return int(round(s*1000))


	def set_times_as_samples(self,offset = 0,baseline = -150,epoch_dur = 300):
		'''Set start/end sample number of a phoneme based on offset value
		(copied from word class)	

		Keywords:
		offset = offset in samples from onset marker sample number, default = 0  int
		baseline = length of baseline window - determines start of epoch, default = -300   int
		epoch_dur = length from onset word to end of epoch, default = 1000   int
		'''
		try:
			self.sample_offset = offset
			self.st_sample = self.seconds2sample(self.st) + offset
			self.et_sample = self.seconds2sample(self.et) + offset
			self.duration_sample = self.seconds2sample(self.duration)
			self.st_epoch = self.st_sample + baseline
			self.et_epoch = self.st_sample + epoch_dur
			self.baseline = baseline
			self.epoch_dur = epoch_dur
			if self.st_sample < 0: self.usable = False
		except:
			print(self.__str__())
			self.usable = False


