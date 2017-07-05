#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
import codecs
import glob
import sid
import utils


class Ort:
	'''Create struture that holds start and end times and POS tag for all words in an audio file.

	Ort uses transcription from CGN and IFADV corpus
	ort holds sid (speakers object) in speakers, sid holds chunks and 
	sentences and they both hold words
	sentences are needed for pos tags and surprisal, surprisal needs to be added still, 
	as does frequency
	'''

	def __init__(self,fid = None,sid_name = 'spreker1', path = None,awd_path = None,corpus = 'IFADV', pos_path = None,register = 'spontaneous_dialogue',auto_set_paths = True):
		'''Load information about words in the audio

		Keywords:
		fid = file id of audio / transcription file in the corpus
		sid_name = speaker id in the corpus
		path/awd_path/pos_poth = location of ort (ortographic) awd (forced aligned) and pos transcription
		corpus = corpus of file (IFADV/CGN) default = IFADV
		register = type of speech (spontaneous_dialogue/news_broadcast/read_aloud_stories)
		auto_set_paths = set paths based on register and corpus
		'''
		if fid == None:
			fid = 'DVA13U'
			print('calling ort class with default file id: ',fid)
		print('creating ort with file id:',fid,' and speaker id:',sid_name)
		self.fid = fid
		self.auto_set_paths = True
		self.corpus = corpus
		self.path = path
		self.awd_path = awd_path
		self.pos_path = pos_path
		self.register = register
		self.set_paths()
		self.sids = []
		self.speakers_present = False
		self.nspeakers = 0
		self.speakers = [] 
		self.words = []
		self.sentences = []
		self.sentence_overlap_indices = []

		if sid:
			self.add_speaker(sid_name)

		utils.make_attributes_available(self,'speaker',self.speakers)

	def __str__(self):
		a = ['file id:\t' + self.fid ]
		a.append('speaker ids:\t'+'  '.join(self.sids))
		for speaker in self.speakers:
			a.append('-'* 50)
			a.append(speaker.__str__())
		return '\n'.join(a)

	def set_paths(self):
		'''Set location of the different transcription (ort/awd/pos).'''
		if not self.auto_set_paths: 
			print('not setting paths')
		elif self.corpus == 'IFADV':
			print('setting paths for IFADV')
			self.path = '../IFADV_ANNOTATION/ORT/'
			self.awd_path = '../IFADV_ANNOTATION/AWD/WORD_TABLES/'
			self.pos_path = 'POS_IFADV/FROG_OUTPUT/'
		elif self.corpus == 'CGN': 
			print('setting paths for CGN')
			self.path = '../TABLE_CGN2_ORT/' 
			self.awd_path = '../TABLE_CGN2_AWD/'
			if self.register == 'read_aloud_stories':self.pos_path='POS_O/FROG_OUTPUT/'
			elif self.register == 'news_broadcast':self.pos_path='POS_K/FROG_OUTPUT/'
			else:
				print('unknown register:',self.register)
				self.pos_path = 'POS_O/FROG_OUTPUT/'
				print('setting pos_path to comp-o:',self.pos_path)
		else:
			print('unknown corpus',self.corpus,'please set paths yourself')
		

	def add_speaker(self,sid_name):
		'''add a speaker, multiply speakers only in case of ifadv.'''
		self.speakers.append( sid.Sid(self.fid,sid_name,self.path,self.awd_path,self.corpus,self.pos_path,self.register) )
		self.sids.append(sid_name)
		self.speakers_present = True
		self.nspeakers += 1
		self.words.extend(self.speakers[-1].words)
		self.sentences.extend(self.speakers[-1].sentences)


	def check_overlap(self):
		'''check whether words overlap with other words only needed for ifadv'''
		if len(self.speakers) != 2:
			print('need 2 speaker to check for overlap (> 2 speakers not implemented')
			return 0
		s1,s2 = self.speakers
		for i,w1 in enumerate(s1.words):
			w1.set_overlap(overlap = False)
			for i2,w2 in enumerate(s2.words):
				w2.set_overlap(overlap = False)
				if w1.st and w1.et and w2.st and w2.et:
					if (w1.st >= w2.st and w1.st < w2.et) \
						or (w1.et >= w2.st and w1.et < w2.et):
						if i not in s1.word_overlap_indices:
							s1.word_overlap_indices.append(i)
						if i2 not in s2.word_overlap_indices:
							s2.word_overlap_indices.append(i2)

		for i in s1.word_overlap_indices:
			s1.words[i].set_overlap(overlap = True)
		for i in s2.word_overlap_indices:
			s2.words[i].set_overlap(overlap = True)

		for s in self.speakers:
			for i,c in enumerate(s.chunks):
				c.check_overlap()
				if c.overlap and not c.overlap_unknown:
					s.chunk_overlap_indices.append(i)

			for j,se in enumerate(s.sentences):
				se.check_overlap()
				if se.overlap and not se.overlap_unknown:
					self.sentence_overlap_indices.append(j)

		s1.n_chunk_overlap = len(s1.chunk_overlap_indices)
		s1.n_word_overlap = len(s1.word_overlap_indices)
		s2.n_chunk_overlap = len(s2.chunk_overlap_indices)
		s2.n_word_overlap = len(s1.word_overlap_indices)


	def make_sentences(self):
		'''Makes sentences for each speaker (words between eol markers . ! ?)

		Sentence structure is needed for POS tagging and language modelling.
		'''
		for s in self.speakers:
			s.make_sentences()


	def print_sentences(self):
		'''Print out all sentences for each speaker into a text file 
		needed to do FROG pos tagging and language modelling
		'''
		for s in self.speakers:
			s.print_utf8_sentences()

	def create_pos_files(self):
		'''Not relevant for EEG processing 
		Create ifadv pos files similar to the ones on the site
		''' 
		output = []
		for speaker in self.speakers:
			for s in speaker.sentences:
				output.append('< file id: '+speaker.fid+' speaker id: '+speaker.sid+ ' sentence: '+str(s.sentence_number+1)+ ' start time: '+str(s.st) + ' end time:' + str(s.et) + ' >')
				for w in s.words:
					output.append(w.make_pos_info())	
		return output

