#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
import chunk
import codecs
import glob
import sentence 
import match_words

class Sid:
	def __init__(self,fid = None,sid = 'spreker1',path = '../IFADV_ANNOTATION/ORT/',awd_path= '../IFADV_ANNOTATION/AWD/WORD_TABLES/'):
		if fid == None:
			fid = 'DVA13U'
			print('calling sid with default file id: ',fid)
		print('creating sid with file id:',fid,' and speaker id:',sid)
		self.fid = fid
		self.sid = sid
		self.path = path
		self.awd_path = awd_path
		self.awd_fn = glob.glob(awd_path + fid + '*' + 'Table')
		self.ort_fn = glob.glob(path + fid + '*' + sid)
		self.find_ort_filename()
		self.find_awd_filename()
		self.read_ort()
		self.read_awd()
		self.add_chunks()
		self.make_sentences()
		self.chunk_overlap_indices = []
		self.word_overlap_indices = []
		self.n_chunk_overlap = 0
		self.n_word_overlap = 0
		self.read_frog_pos_output()
		self.split_pos_text()
		self.add_pos_to_sentences() 

	def __str__(self):
		a = ['file id:\t' + self.fid ]
		a.append('speaker id:\t'+self.sid)
		a.append('filename:\t'+self.ort_filename)
		a.append('awd filename:\t'+self.awd_filename)
		a.append('Number of chunks:\t'+ str(self.nchunks))
		a.append('Number of words:\t'+ str(self.nwords))
		a.append('N sentences:\t\t' + str(self.nsentences))
		a.append('N overlap chunks:\t'+ str(self.n_chunk_overlap))
		a.append('N overlap words:\t'+ str(self.n_word_overlap))
		a.append('pos text ok:\t'+ str(self.pos_text_ok))
		a.append('npos sentence ok:\t'+ str(self.npos_sentences_ok))
		return '\n'.join(a)
	

	def find_ort_filename(self):
		for line in self.ort_fn:
			if self.fid in line and self.sid in line:
				self.ort_filename = line
				return line
		print('ort filename not found for: ',self.fid, self.sid)
		return None

	def find_awd_filename(self):
		for line in self.awd_fn:
			if self.fid in line: 
				self.awd_filename = line
				return line
		print('awd filename not found for: ',self.fid, self.sid)
		return None

	def read_ort(self):
		#read in a speaker specific table file (praat exported textgrid)
		self.ort_text = [line.split('\t') for line in codecs.open(self.ort_filename,'r','utf8').read().split('\n') if line]
		return self.ort_text

	def read_awd(self):
		self.awd_text = [line.split('\t') for line in codecs.open(self.awd_filename,'r','utf8').read().split('\n') if line and self.sid in line.split('\t')[1] ]
		for line in self.awd_text:
			line[0],line[-1] = float(line[0]),float(line[-1])
		return self.awd_text

	def remove_header(self,ort_text):
		h = ort_text[0]
		if h[0] =='tmin':
			return ort_text[1:] 
		else:
			return ort_text

	def find_awd_items_in_chunk(self,c):
		awd_items_in_chunk = []
		last = self.last_awd_index
		for i,line in enumerate(self.awd_text[self.last_awd_index:]):
			ok = False
			if line[0] >= c.st and line[-1] <= c.et: ok = True
			if line[0] < c.st and line[-1] > c.st: ok = True
			if line[0] < c.et and line[-1] > c.et: ok = True
			if ok:
				if line[2] not in ['sil','sp']:
					awd_items_in_chunk.append(line) 
				last = i
		self.last_awd_index = last
		return awd_items_in_chunk

	def add_chunks(self):
		# add chunk to the ort object, a chunk is a basic unit of orthografic
		# transcription and represented as an object consisting of words
		print('adding chunks to sid')
		ort_text = self.remove_header(self.ort_text)
		self.chunks = []
		self.nwords,self.nchunks = 0,0
		self.words = []
		self.last_awd_index= 0 
		for i,line in enumerate(ort_text):
			c = chunk.Chunk(line,i,self.ort_filename,self.fid,self.sid)
			awd_items_in_chunk = self.find_awd_items_in_chunk(c)
			c.add_awd_items_in_chunk(awd_items_in_chunk)
			c.match_awd2word()
			self.nwords += c.nwords
			self.nchunks += 1
			self.chunks.append(c)
			self.words.extend(c.words)


	def make_sentences(self):
		print('creating sentences, words between eols')
		self.sentences = []
		sentence_wl = []
		sentence_index = 0
		for w in self.words:
			sentence_wl.append(w)
			if w.eol:
				self.sentences.append(sentence.Sentence(sentence_wl,sentence_index))
				sentence_index += 1
				sentence_wl = []
		self.nsentences = len(self.sentences)


	def print_utf8_sentences(self,filename = 'sentences_utf8'):
		filename = filename + '_' + self.fid + '_' + self.sid + '.txt'
		print('saving all sentences to:',filename)
		output = []
		for s in self.sentences:
			output.append(s.string_utf8_words())
		fout = open(filename,'w')
		fout.write('.\n'.join(output))
		fout.close()
		

	def read_frog_pos_output(self):
		fn = glob.glob('POS_IFADV/*'+self.fid+'_'+self.sid+'*')
		if len(fn) != 1:
			print('did not find 1 filename, maybe no FROG POS output present')
			print('found the following using',self.fid,self.sid,':')
			print('file(s) found:',fn)
			self.pos_text_ok = False
			return 0
		print('reading FROG output pos tags, with',self.fid,self.sid,'\nfilename:',fn[0])
		self.pos_text_ok = True
		self.pos_filename = fn[0]
		self.pos_text = codecs.open(self.pos_filename,'r','utf8').read().split('\n')


	def split_pos_text(self):
		if self.pos_text_ok == False:
			print('pos tags not read in')
			self.npos_sentences_ok = False
			return 0
		print('splitting pos text into sentences')
		self.pos_sentences = []
		pos_wl = []
		for line in self.pos_text:
			if line == '' and pos_wl != []:
				self.pos_sentences.append(pos_wl)
				pos_wl = []
			elif line != '':
				pos_wl.append(line.split('\t'))
		if self.nsentences == len(self.pos_sentences): self.npos_sentences_ok = True
		else: self.npos_sentences_ok = False
				
				
	def add_pos_to_sentences(self):
		print('adding pos tags to words in sentences')
		if self.pos_text_ok == False:
			print('pos tags not read in')
			return 0
		if self.npos_sentences_ok == False:
			print('npos sentences do not equal number of sentences')
			return 0
		for i,s in enumerate(self.sentences):
			pos_tags = self.pos_sentences[i]
			s.add_pos_to_words(pos_tags)
					

