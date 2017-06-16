#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
import codecs
import glob
import string

class Word:
	def __init__(self, word,word_number, chunk_number, chunk_st,chunk_et,filename = None, fid = None,sid = None,cid = None,wid = None, st = None, et = None,corpus = None,register = None):
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
		if self.pos_ok:
			a.append('-'*30)
			a.append('POS tag INFO')
			a.append(self.pos.__str__())
		return '\n'.join(a)
		

	def status(self):
		self.special_code,self.punctuation,self.eol = False,False,False
		if '*' in self.word:
			self.special_code = True
		for p in string.punctuation.replace('*',''):
			if p in self.word:
				self.punctuation = True
		for eol in ['.','!','?']:
			if eol in self.word:
				self.eol = True

	def add_info(self,filename = None, fid = None,sid = None,cid = None,wid = None,st = None, et = None,corpus = None,register = None):
		if filename: self.filename = filename
		if fid: self.fid = fid
		if sid: self.sid = sid
		if cid: self.cid = cid
		if wid: self.wid = wid
		if st: self.st = st
		if et: self.et = et
		if corpus: self.corpus = corpus
		if register: self.register = register

	def add_prev_word(self,word = None):
		self.prev_word = word

	def add_next_word(self,word = None):
		self.next_word = word

	def set_overlap(self,overlap = None):
		if overlap == None:
			self.overlap_unknown = True
			self.overlap = False
		else:
			self.overlap = overlap 
			self.overlap_unknown = False

	def add_times(self,st = None, et= None):
		if st: self.st = st
		if et: self.et = et
		if self.st and self.et:
			self.duration = round(self.et - self.st,3)

	
	def replace_diacritics(self):
		rd = {'\e^':'ê','\e"':'ë',"\e'":'é',"\e`":'è','\i"':'ï',"\i'":'í','\\u"':'ü',"\\u'":'ú',"\\a'":'á',"\\a`":'à','\o"':'ö',"\o'":'ó','\c,':'ç'}
		temp = self.word
		for c in rd.keys():
			temp = temp.replace(c,rd[c])
		self.word_utf8 = temp


	def remove_special_code(self):
		if self.special_code:
			self.word_nocode = self.word.split('*')[0]
			self.word_utf8_nocode = self.word_utf8.split('*')[0]
		else:
			self.word_nocode = self.word
			self.word_utf8_nocode = self.word_utf8

		for eol in ['.','!','?']:
			self.word_nocode = self.word_nocode.replace(eol,'')
			self.word_utf8_nocode = self.word_utf8_nocode.replace(eol,'')

	def make_pos_info(self):
		if not self.pos_ok:
			print('no pos or problem with pos',self.fid,self.sid,self.chunk_number)
			# print(self.__str__())
			return 0
		output = []
		p = self.pos
		output = [p.token,p.pos,p.lemma,p.probability_of_tag]
		output.extend([self.st,self.et,self.word,self.word_number+1,self.chunk_number+1])
		output = map(str,output)
		return '\t'.join(output)

		output.append(self.st)
